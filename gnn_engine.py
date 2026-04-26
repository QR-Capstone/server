"""
Web-structure GNN lane for phishing detection.

The GNN model builds a small graph from the target page instead of treating the
URL as a flat string:

    page -> domains / links / scripts / images / iframes / forms / inputs / brands

Each node receives a type-aware initial risk, then two message-passing rounds
propagate risk through the graph. The final page embedding is classified by a
trained logistic head saved in gnn_model.pkl.

No FastAPI/XGBoost/KoBERT code needs to know about this; the public API remains
GNN_Engine + predict_gnn.
"""
from __future__ import annotations

import math
import os
import pickle
import re
import socket
import ssl
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import Request, urlopen

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_KIND = "web_structure_gnn_phishing_v1"
ARTIFACT_VERSION = 2

DEFAULT_TIMEOUT = float(os.getenv("GNN_FETCH_TIMEOUT", "4.0"))
DEFAULT_MAX_BYTES = int(os.getenv("GNN_FETCH_MAX_BYTES", str(512 * 1024)))

PHISHING_WORDS = {
    "account",
    "auth",
    "bank",
    "billing",
    "cancel",
    "confirm",
    "credential",
    "login",
    "password",
    "pay",
    "payment",
    "secure",
    "signin",
    "support",
    "update",
    "verify",
    "wallet",
}

BRAND_WORDS = {
    "apple",
    "binance",
    "discord",
    "facebook",
    "github",
    "google",
    "instagram",
    "kakao",
    "metamask",
    "microsoft",
    "naver",
    "netflix",
    "paypal",
}

COMMON_SECOND_LEVEL_SUFFIXES = {
    "ac",
    "co",
    "com",
    "edu",
    "go",
    "gov",
    "ne",
    "net",
    "or",
    "org",
}

FEATURE_NAMES: List[str] = [
    "url_len",
    "host_len",
    "path_len",
    "dot_count",
    "hyphen_count",
    "digit_ratio",
    "entropy",
    "is_https",
    "path_depth",
    "token_count",
    "phish_word_ratio",
    "brand_word_ratio",
    "html_fetched",
    "fetch_failed",
    "status_bad",
    "final_domain_changed",
    "redirect_count",
    "graph_node_count",
    "graph_edge_count",
    "internal_link_ratio",
    "external_link_ratio",
    "external_resource_ratio",
    "form_count",
    "external_form_ratio",
    "password_input_ratio",
    "iframe_ratio",
    "script_ratio",
    "image_ratio",
    "brand_domain_mismatch",
    "empty_navigation_ratio",
    "page_risk_after_mp",
    "max_neighbor_risk",
    "mean_neighbor_risk",
    "risk_spread",
    "risky_edge_ratio",
    "domain_diversity",
]


def resolve_gnn_model_path(base_dir: str = _BASE_DIR) -> str:
    if p := os.getenv("GNN_MODEL_PATH"):
        return p
    if p := os.getenv("OPQR_MODEL_PATH"):
        return p
    return os.path.join(base_dir, "gnn_model.pkl")


def resolve_gnn_features_path(base_dir: str = _BASE_DIR) -> str:
    if p := os.getenv("GNN_FEATURES_PATH"):
        return p
    if p := os.getenv("OPQR_FEATURES_PATH"):
        return p
    return os.path.join(base_dir, "gnn_model_features.pkl")


def default_gnn_paths() -> Tuple[str, str]:
    return resolve_gnn_model_path(), resolve_gnn_features_path()


def _normalize_url(raw_url: str) -> str:
    u = (raw_url or "").strip()
    if not u:
        raise ValueError("empty url")
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _cap(value: float, scale: float) -> float:
    return min(max(float(value) / scale, 0.0), 1.0)


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = float(len(text))
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _host_parts(host: str) -> List[str]:
    return [p for p in host.lower().split(".") if p]


def _registered_domain(host: str) -> str:
    parts = _host_parts(host)
    if len(parts) < 2:
        return host.lower()
    if len(parts) >= 3 and parts[-2] in COMMON_SECOND_LEVEL_SUFFIXES:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _tokenize(text: str) -> List[str]:
    out: List[str] = []
    for tok in re.split(r"[^a-zA-Z0-9]+", text.lower()):
        if 2 <= len(tok) <= 32:
            out.append(tok)
    return out


def _url_tokens(url: str) -> List[str]:
    parsed = urlsplit(url)
    return _tokenize(" ".join([parsed.hostname or "", parsed.path or "", parsed.query or ""]))


def _lexical_features(url: str) -> Dict[str, float]:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    path = parsed.path or ""
    tokens = _url_tokens(url)
    digits = sum(1 for ch in url if ch.isdigit())
    phish_hits = sum(1 for t in tokens if t in PHISHING_WORDS)
    brand_hits = sum(1 for t in tokens if t in BRAND_WORDS)
    return {
        "url_len": _cap(len(url), 220.0),
        "host_len": _cap(len(host), 80.0),
        "path_len": _cap(len(path), 160.0),
        "dot_count": _cap(url.count("."), 10.0),
        "hyphen_count": _cap(url.count("-"), 12.0),
        "digit_ratio": _safe_ratio(digits, len(url)),
        "entropy": _cap(_shannon_entropy(url), 5.5),
        "is_https": 1.0 if parsed.scheme == "https" else 0.0,
        "path_depth": _cap(len([p for p in path.split("/") if p]), 10.0),
        "token_count": _cap(len(tokens), 24.0),
        "phish_word_ratio": _safe_ratio(phish_hits, len(tokens)),
        "brand_word_ratio": _safe_ratio(brand_hits, len(tokens)),
    }


class _StructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: List[str] = []
        self.images: List[str] = []
        self.scripts: List[str] = []
        self.iframes: List[str] = []
        self.forms: List[str] = []
        self.inputs: List[Dict[str, str]] = []
        self.text_chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        amap = {k.lower(): (v or "") for k, v in attrs}
        tag = tag.lower()
        if tag == "a" and amap.get("href"):
            self.links.append(amap["href"])
        elif tag == "img" and amap.get("src"):
            self.images.append(amap["src"])
        elif tag == "script" and amap.get("src"):
            self.scripts.append(amap["src"])
        elif tag == "iframe" and amap.get("src"):
            self.iframes.append(amap["src"])
        elif tag == "form":
            self.forms.append(amap.get("action", ""))
        elif tag == "input":
            self.inputs.append(
                {
                    "type": amap.get("type", "").lower(),
                    "name": amap.get("name", "").lower(),
                    "id": amap.get("id", "").lower(),
                    "placeholder": amap.get("placeholder", "").lower(),
                }
            )

    def handle_data(self, data: str) -> None:
        data = (data or "").strip()
        if data:
            self.text_chunks.append(data[:200])


@dataclass
class FetchedPage:
    requested_url: str
    final_url: str
    status: int
    html: str
    error: Optional[str]
    redirect_count: int


def fetch_page(url: str, timeout: float = DEFAULT_TIMEOUT, max_bytes: int = DEFAULT_MAX_BYTES) -> FetchedPage:
    normalized = _normalize_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
        )
    }
    try:
        req = Request(normalized, headers=headers)
        context = ssl.create_default_context()
        with urlopen(req, timeout=timeout, context=context) as resp:
            raw = resp.read(max_bytes)
            final_url = resp.geturl() or normalized
            status = int(getattr(resp, "status", 200) or 200)
            charset = resp.headers.get_content_charset() or "utf-8"
            html = raw.decode(charset, errors="replace")
            redirects = 1 if _registered_domain(urlsplit(normalized).hostname or "") != _registered_domain(urlsplit(final_url).hostname or "") else 0
            return FetchedPage(normalized, final_url, status, html, None, redirects)
    except HTTPError as e:
        body = ""
        try:
            body = e.read(max_bytes).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return FetchedPage(normalized, e.geturl() or normalized, int(e.code), body, str(e), 0)
    except (URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as e:
        return FetchedPage(normalized, normalized, 0, "", str(e), 0)


@dataclass
class WebGraph:
    page_url: str
    final_url: str
    status: int
    fetch_error: Optional[str]
    nodes: Set[str]
    edges: List[Tuple[str, str, str]]
    node_risk: Dict[str, float]
    domains: Set[str]
    brands: Set[str]
    counts: Dict[str, int]


def _node_domain(abs_url: str) -> str:
    return _registered_domain(urlsplit(abs_url).hostname or "")


def _add_edge(
    nodes: Set[str],
    edges: List[Tuple[str, str, str]],
    node_risk: Dict[str, float],
    src: str,
    rel: str,
    dst: str,
    risk: float,
) -> None:
    nodes.add(src)
    nodes.add(dst)
    edges.append((src, rel, dst))
    node_risk[dst] = max(node_risk.get(dst, 0.0), risk)


def build_web_graph(url: str, fetch: bool = True) -> WebGraph:
    normalized = _normalize_url(url)
    fetched = fetch_page(normalized) if fetch else FetchedPage(normalized, normalized, 0, "", "fetch_disabled", 0)
    base_url = fetched.final_url or normalized
    base_domain = _registered_domain(urlsplit(base_url).hostname or "")
    page = "page:target"
    nodes: Set[str] = {page}
    edges: List[Tuple[str, str, str]] = []
    node_risk: Dict[str, float] = {page: 0.0}
    domains: Set[str] = set()
    counts: Dict[str, int] = defaultdict(int)

    parser = _StructureParser()
    if fetched.html:
        try:
            parser.feed(fetched.html)
        except Exception:
            pass

    text = " ".join(parser.text_chunks[:80])
    tokens = set(_url_tokens(base_url) + _tokenize(text))
    brands = {t for t in tokens if t in BRAND_WORDS}
    base_parts = set(_host_parts(base_domain))

    for brand in brands:
        risk = 0.75 if brand not in base_parts else 0.05
        _add_edge(nodes, edges, node_risk, page, "mentions_brand", f"brand:{brand}", risk)
        counts["brand"] += 1

    def add_url_relation(raw: str, rel: str, node_prefix: str, external_risk: float) -> None:
        if not raw or raw.startswith(("javascript:", "mailto:", "tel:", "#")):
            counts["empty_nav"] += 1
            return
        abs_url = urljoin(base_url, raw)
        dom = _node_domain(abs_url)
        if not dom:
            return
        domains.add(dom)
        external = dom != base_domain
        risk = external_risk if external else 0.05
        dst = f"{node_prefix}:{dom}"
        _add_edge(nodes, edges, node_risk, page, rel, dst, risk)
        _add_edge(nodes, edges, node_risk, dst, "domain", f"domain:{dom}", risk)
        counts[f"{rel}_external" if external else f"{rel}_internal"] += 1

    for href in parser.links[:300]:
        add_url_relation(href, "links_to", "link", 0.35)
    for src in parser.images[:300]:
        add_url_relation(src, "loads_image", "image", 0.18)
    for src in parser.scripts[:200]:
        add_url_relation(src, "loads_script", "script", 0.42)
    for src in parser.iframes[:80]:
        add_url_relation(src, "embeds_iframe", "iframe", 0.65)
    for action in parser.forms[:80]:
        add_url_relation(action or base_url, "submits_to", "form", 0.85)

    password_inputs = 0
    suspicious_inputs = 0
    for inp in parser.inputs[:200]:
        blob = " ".join(inp.values())
        is_password = inp.get("type") == "password" or "password" in blob
        is_suspicious = is_password or any(word in blob for word in PHISHING_WORDS)
        if is_password:
            password_inputs += 1
        if is_suspicious:
            suspicious_inputs += 1
        risk = 0.75 if is_password else (0.45 if is_suspicious else 0.08)
        _add_edge(nodes, edges, node_risk, page, "has_input", f"input:{len(nodes)}", risk)
    counts["input"] = len(parser.inputs)
    counts["password_input"] = password_inputs
    counts["suspicious_input"] = suspicious_inputs
    counts["form"] = len(parser.forms)
    counts["link"] = len(parser.links)
    counts["image"] = len(parser.images)
    counts["script"] = len(parser.scripts)
    counts["iframe"] = len(parser.iframes)

    if not fetched.html and fetched.error:
        node_risk[page] = 0.25
    if fetched.status >= 400 or fetched.status == 0:
        node_risk[page] = max(node_risk[page], 0.25)

    return WebGraph(
        page_url=normalized,
        final_url=base_url,
        status=fetched.status,
        fetch_error=fetched.error,
        nodes=nodes,
        edges=edges,
        node_risk=node_risk,
        domains=domains,
        brands=brands,
        counts=dict(counts),
    )


def _message_pass(graph: WebGraph, rounds: int = 2) -> Dict[str, float]:
    neighbors: Dict[str, List[str]] = defaultdict(list)
    for src, _, dst in graph.edges:
        neighbors[src].append(dst)
        neighbors[dst].append(src)
    risk = {n: float(graph.node_risk.get(n, 0.0)) for n in graph.nodes}
    for _ in range(rounds):
        new_risk = dict(risk)
        for node in graph.nodes:
            ns = neighbors.get(node, [])
            if not ns:
                continue
            msg = sum(risk.get(n, 0.0) for n in ns) / len(ns)
            new_risk[node] = 0.55 * risk.get(node, 0.0) + 0.45 * msg
        risk = new_risk
    return risk


def _structure_features(graph: WebGraph) -> Dict[str, float]:
    counts = graph.counts
    risks = _message_pass(graph)
    page_risk = risks.get("page:target", 0.0)
    neighbor_risks = [risks.get(dst, 0.0) for src, _, dst in graph.edges if src == "page:target"]
    total_links = counts.get("links_to_external", 0) + counts.get("links_to_internal", 0)
    total_resources = (
        counts.get("loads_image_external", 0)
        + counts.get("loads_image_internal", 0)
        + counts.get("loads_script_external", 0)
        + counts.get("loads_script_internal", 0)
        + counts.get("embeds_iframe_external", 0)
        + counts.get("embeds_iframe_internal", 0)
    )
    external_resources = (
        counts.get("loads_image_external", 0)
        + counts.get("loads_script_external", 0)
        + counts.get("embeds_iframe_external", 0)
    )
    total_forms = max(1, counts.get("submits_to_external", 0) + counts.get("submits_to_internal", 0))
    final_changed = (
        _registered_domain(urlsplit(graph.page_url).hostname or "")
        != _registered_domain(urlsplit(graph.final_url).hostname or "")
    )
    base_domain = _registered_domain(urlsplit(graph.final_url).hostname or "")
    base_parts = set(_host_parts(base_domain))
    brand_mismatch = 1.0 if graph.brands and not graph.brands.intersection(base_parts) else 0.0
    risky_edges = sum(1 for _, _, dst in graph.edges if graph.node_risk.get(dst, 0.0) >= 0.5)

    return {
        "html_fetched": 1.0 if graph.fetch_error is None and bool(graph.edges or graph.counts) else 0.0,
        "fetch_failed": 1.0 if graph.fetch_error else 0.0,
        "status_bad": 1.0 if graph.status >= 400 or graph.status == 0 else 0.0,
        "final_domain_changed": 1.0 if final_changed else 0.0,
        "redirect_count": _cap(1 if final_changed else 0, 5.0),
        "graph_node_count": _cap(len(graph.nodes), 180.0),
        "graph_edge_count": _cap(len(graph.edges), 260.0),
        "internal_link_ratio": _safe_ratio(counts.get("links_to_internal", 0), total_links),
        "external_link_ratio": _safe_ratio(counts.get("links_to_external", 0), total_links),
        "external_resource_ratio": _safe_ratio(external_resources, total_resources),
        "form_count": _cap(counts.get("form", 0), 8.0),
        "external_form_ratio": _safe_ratio(counts.get("submits_to_external", 0), total_forms),
        "password_input_ratio": _safe_ratio(counts.get("password_input", 0), max(1, counts.get("input", 0))),
        "iframe_ratio": _safe_ratio(counts.get("iframe", 0), max(1, total_resources + total_links)),
        "script_ratio": _safe_ratio(counts.get("script", 0), max(1, total_resources + total_links)),
        "image_ratio": _safe_ratio(counts.get("image", 0), max(1, total_resources + total_links)),
        "brand_domain_mismatch": brand_mismatch,
        "empty_navigation_ratio": _safe_ratio(counts.get("empty_nav", 0), max(1, total_links + counts.get("empty_nav", 0))),
        "page_risk_after_mp": page_risk,
        "max_neighbor_risk": max(neighbor_risks) if neighbor_risks else 0.0,
        "mean_neighbor_risk": sum(neighbor_risks) / len(neighbor_risks) if neighbor_risks else 0.0,
        "risk_spread": (max(neighbor_risks) - min(neighbor_risks)) if len(neighbor_risks) > 1 else 0.0,
        "risky_edge_ratio": _safe_ratio(risky_edges, len(graph.edges)),
        "domain_diversity": _cap(len(graph.domains), 30.0),
    }


def feature_map_from_graph(graph: WebGraph) -> Dict[str, float]:
    features = _lexical_features(graph.page_url)
    features.update(_structure_features(graph))
    return features


def feature_map_for_url(url: str, fetch: bool = True) -> Dict[str, float]:
    normalized = _normalize_url(url)
    return feature_map_from_graph(build_web_graph(normalized, fetch=fetch))


def graph_feature_vector(url: str, fetch: bool = True) -> List[float]:
    features = feature_map_for_url(url, fetch=fetch)
    return [float(features.get(name, 0.0)) for name in FEATURE_NAMES]


def _structure_prior_logit(features: Dict[str, float]) -> float:
    """Convert graph-structure risk signals into a centered logit adjustment."""
    risk = 0.0
    risk += 1.40 * features.get("page_risk_after_mp", 0.0)
    risk += 1.15 * features.get("max_neighbor_risk", 0.0)
    risk += 0.95 * features.get("risky_edge_ratio", 0.0)
    risk += 0.90 * features.get("external_form_ratio", 0.0)
    risk += 0.70 * features.get("password_input_ratio", 0.0)
    risk += 0.75 * features.get("brand_domain_mismatch", 0.0)
    risk += 0.35 * features.get("final_domain_changed", 0.0)
    risk += 0.30 * features.get("iframe_ratio", 0.0)
    risk += 0.20 * features.get("fetch_failed", 0.0)
    risk -= 0.45 * features.get("internal_link_ratio", 0.0)
    risk -= 0.20 * features.get("is_https", 0.0)
    return max(-1.25, min(1.25, risk - 0.65))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _standardize(x: Sequence[float], means: Sequence[float], scales: Sequence[float]) -> List[float]:
    return [(v - m) / s for v, m, s in zip(x, means, scales)]


@dataclass
class WebStructureGNNModel:
    weights: List[float]
    bias: float
    means: List[float]
    scales: List[float]
    threshold: float
    metadata: Dict[str, Any]

    @property
    def feature_names(self) -> List[str]:
        return list(FEATURE_NAMES)

    def predict_proba_one(self, url: str, fetch: bool = True) -> float:
        fmap = feature_map_for_url(url, fetch=fetch)
        return self.predict_proba_from_features(fmap)

    def predict_proba_from_features(self, fmap: Dict[str, float]) -> float:
        vec = [float(fmap.get(name, 0.0)) for name in FEATURE_NAMES]
        z = _dot(self.weights, _standardize(vec, self.means, self.scales)) + self.bias
        z += float(self.metadata.get("structure_prior_weight", 2.4)) * _structure_prior_logit(fmap)
        return _sigmoid(z)

    def evidence(self, url: str) -> Dict[str, Any]:
        return self.evidence_from_graph(build_web_graph(url, fetch=True))

    def evidence_from_graph(self, graph: WebGraph) -> Dict[str, Any]:
        risks = _message_pass(graph)
        top = []
        for node, risk in sorted(risks.items(), key=lambda kv: kv[1], reverse=True):
            if node == "page:target":
                continue
            top.append({"node": node, "risk": round(float(risk), 4)})
            if len(top) >= 8:
                break
        return {
            "graph": "page->links/resources/forms/inputs/domains/brands",
            "status": graph.status,
            "fetch_error": graph.fetch_error,
            "final_url": graph.final_url,
            "nodes": len(graph.nodes),
            "edges": len(graph.edges),
            "top_risk_nodes": top,
            "counts": graph.counts,
        }


def train_web_structure_gnn_model(
    urls: Sequence[str],
    labels: Sequence[int],
    *,
    fetch_pages: bool = False,
    epochs: int = 900,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    threshold: float = 0.5,
) -> WebStructureGNNModel:
    if len(urls) != len(labels):
        raise ValueError("urls and labels length mismatch")
    if not urls:
        raise ValueError("empty training data")

    raw_vectors = [graph_feature_vector(url, fetch=fetch_pages) for url in urls]
    cols = list(zip(*raw_vectors))
    means = [sum(col) / len(col) for col in cols]
    scales = []
    for col, mean in zip(cols, means):
        var = sum((v - mean) ** 2 for v in col) / max(1, len(col) - 1)
        scales.append(math.sqrt(var) if var > 1e-12 else 1.0)
    vectors = [_standardize(v, means, scales) for v in raw_vectors]

    weights = [0.0 for _ in FEATURE_NAMES]
    bias = 0.0
    ys = [int(y) for y in labels]
    n = float(len(vectors))
    for _ in range(max(1, epochs)):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for x, y in zip(vectors, ys):
            err = _sigmoid(_dot(weights, x) + bias) - y
            for i, val in enumerate(x):
                grad_w[i] += err * val
            grad_b += err
        for i in range(len(weights)):
            grad_w[i] = grad_w[i] / n + l2 * weights[i]
            weights[i] -= learning_rate * grad_w[i]
        bias -= learning_rate * (grad_b / n)

    metadata = {
        "kind": MODEL_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "training_rows": len(urls),
        "fetch_pages_during_training": bool(fetch_pages),
        "structure_prior_weight": 2.4,
        "description": "Webpage structure graph with two-round message passing and logistic classifier.",
    }
    return WebStructureGNNModel(weights, bias, means, scales, threshold, metadata)


def train_web_structure_gnn_model_from_vectors(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    epochs: int = 900,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    threshold: float = 0.5,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> WebStructureGNNModel:
    if len(vectors) != len(labels):
        raise ValueError("vectors and labels length mismatch")
    if not vectors:
        raise ValueError("empty training data")
    raw_vectors = [[float(v) for v in row] for row in vectors]
    if any(len(row) != len(FEATURE_NAMES) for row in raw_vectors):
        raise ValueError(f"each vector must have {len(FEATURE_NAMES)} features")

    cols = list(zip(*raw_vectors))
    means = [sum(col) / len(col) for col in cols]
    scales = []
    for col, mean in zip(cols, means):
        var = sum((v - mean) ** 2 for v in col) / max(1, len(col) - 1)
        scales.append(math.sqrt(var) if var > 1e-12 else 1.0)
    train_vectors = [_standardize(v, means, scales) for v in raw_vectors]

    weights = [0.0 for _ in FEATURE_NAMES]
    bias = 0.0
    ys = [int(y) for y in labels]
    n = float(len(train_vectors))
    for _ in range(max(1, epochs)):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for x, y in zip(train_vectors, ys):
            err = _sigmoid(_dot(weights, x) + bias) - y
            for i, val in enumerate(x):
                grad_w[i] += err * val
            grad_b += err
        for i in range(len(weights)):
            grad_w[i] = grad_w[i] / n + l2 * weights[i]
            weights[i] -= learning_rate * grad_w[i]
        bias -= learning_rate * (grad_b / n)

    metadata = {
        "kind": MODEL_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "training_rows": len(raw_vectors),
        "fetch_pages_during_training": False,
        "trained_from_stored_features": True,
        "structure_prior_weight": 2.4,
        "description": "Webpage structure graph features captured at collection time.",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return WebStructureGNNModel(weights, bias, means, scales, threshold, metadata)


def save_gnn_artifact(model: WebStructureGNNModel, model_path: str, features_path: Optional[str] = None) -> None:
    artifact = {
        "kind": MODEL_KIND,
        "version": ARTIFACT_VERSION,
        "weights": model.weights,
        "bias": model.bias,
        "means": model.means,
        "scales": model.scales,
        "threshold": model.threshold,
        "metadata": model.metadata,
    }
    with open(model_path, "wb") as f:
        pickle.dump(artifact, f, protocol=4)
    if features_path:
        with open(features_path, "wb") as f:
            pickle.dump(FEATURE_NAMES, f, protocol=4)


def _artifact_to_model(artifact: Dict[str, Any]) -> WebStructureGNNModel:
    if artifact.get("kind") != MODEL_KIND:
        raise ValueError(f"unsupported_gnn_artifact:{artifact.get('kind')!r}")
    return WebStructureGNNModel(
        weights=[float(x) for x in artifact["weights"]],
        bias=float(artifact["bias"]),
        means=[float(x) for x in artifact["means"]],
        scales=[float(x) for x in artifact["scales"]],
        threshold=float(artifact.get("threshold", 0.5)),
        metadata=dict(artifact.get("metadata", {})),
    )


def _explain_gnn_load_error(exc: Exception) -> str:
    return (
        "Failed to load web-structure GNN artifact. Regenerate it with "
        "regenerate_gnn_model.py so gnn_model.pkl contains a "
        f"{MODEL_KIND} bundle. Detail: {type(exc).__name__}: {exc}"
    )


def load_gnn_model(
    model_path: str,
    feature_columns_path: Optional[str] = None,
) -> Tuple[WebStructureGNNModel, List[str]]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    try:
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        model = _artifact_to_model(artifact)
    except Exception as e:
        raise RuntimeError(_explain_gnn_load_error(e)) from e

    columns = list(FEATURE_NAMES)
    if feature_columns_path and os.path.isfile(feature_columns_path):
        try:
            with open(feature_columns_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, list) and loaded:
                columns = [str(c) for c in loaded]
        except Exception:
            columns = list(FEATURE_NAMES)
    return model, columns


def predict_gnn(
    model: WebStructureGNNModel,
    column_order: List[str],
    raw_url: str,
) -> Dict[str, Any]:
    url = _normalize_url(raw_url)
    fetch = os.getenv("GNN_FETCH_PAGE", "1") != "0"
    graph = build_web_graph(url, fetch=fetch)
    prob_mal = float(model.predict_proba_from_features(feature_map_from_graph(graph)))
    label = 1 if prob_mal >= model.threshold else 0
    out = {
        "url": url,
        "probability": round(prob_mal, 6),
        "label": label,
        "verdict": "malicious" if label == 1 else "benign",
        "model_type": MODEL_KIND,
    }
    if fetch:
        out["graph_evidence"] = model.evidence_from_graph(graph)
    return out


class GNN_Engine:
    """Webpage structure graph detector; paths default to repo root artifacts."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_columns_path: Optional[str] = None,
    ):
        self.model_path = model_path or resolve_gnn_model_path()
        self.feature_columns_path = feature_columns_path or resolve_gnn_features_path()
        self.model: Optional[WebStructureGNNModel] = None
        self.columns: Optional[List[str]] = None
        self._load_error: Optional[str] = None

        if not os.path.isfile(self.model_path):
            self._load_error = f"missing_model:{self.model_path}"
            return

        try:
            fc = self.feature_columns_path if os.path.isfile(self.feature_columns_path) else None
            self.model, self.columns = load_gnn_model(self.model_path, fc)
        except Exception as e:
            self.model = None
            self.columns = None
            self._load_error = str(e)

    @property
    def ok(self) -> bool:
        return self.model is not None and bool(self.columns)

    def analyze(self, url: str) -> Dict[str, Any]:
        if not self.ok or self.model is None or self.columns is None:
            return {
                "error": self._load_error or "model_not_loaded",
                "verdict": "unknown",
                "enabled": False,
            }
        try:
            return predict_gnn(self.model, self.columns, url)
        except Exception as e:
            return {
                "error": str(e),
                "verdict": "unknown",
                "enabled": True,
            }

    def analyze_legacy(self, url: str) -> Dict[str, Any]:
        out = self.analyze(url)
        if out.get("error"):
            return out
        p = float(out.get("probability", 0.0))
        mal = out.get("verdict") == "malicious"
        out = dict(out)
        out["target_url"] = out.get("url", url)
        out["danger_score"] = f"{p * 100:.2f}%"
        out["detection_result"] = "MALICIOUS" if mal else "SAFE"
        return out
