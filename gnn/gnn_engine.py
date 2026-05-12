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
import random
import re
import socket
import ssl
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from curl_cffi import requests as curl_requests
except Exception:  # pragma: no cover
    curl_requests = None  # type: ignore

try:
    import dns.resolver
except Exception:  # pragma: no cover
    dns = None  # type: ignore

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_IMPORT_ERROR = e
_NN_MODULE = nn.Module if nn is not None else object

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_KIND = "web_structure_torch_gnn_phishing_v2"
ARTIFACT_VERSION = 3

DEFAULT_TIMEOUT = float(os.getenv("GNN_FETCH_TIMEOUT", "4.0"))
DEFAULT_MAX_BYTES = int(os.getenv("GNN_FETCH_MAX_BYTES", str(512 * 1024)))
PUBLIC_DNS_SERVERS = ("1.1.1.1", "8.8.8.8")

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
    "ledger",
    "allegro",
    "chase",
    "crypto",
    "trezor",
    "youtube",  
    "amazon",   
    "twitter",  
    "samsung",  
}

SUSPICIOUS_TLDS = {".pro", ".top", ".xyz", ".icu", ".club", ".vip", ".live", ".click", ".cfd", ".ng", ".site",}

BRAND_DOMAIN_ALIASES = {
    "naver": {"naver", "pstatic"},
    "kakao": {"kakao", "daum"},
    "google": {"google", "gstatic", "googleusercontent"},
    "microsoft": {"microsoft", "live", "office", "windows"},
    "apple": {"apple", "icloud"},
    "chase": {"chase"},
     "crypto": {"crypto"},   
    "trezor": {"trezor"},
    "youtube": {"youtube", "youtu"},
    "amazon": {"amazon", "amazonaws"},
    "twitter": {"twitter", "twimg"},
    "samsung": {"samsung"},
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
    "suspicious_input_ratio",
    "credential_surface",
    "brand_capture_mismatch",
    "external_submission_risk",
    "iframe_ratio",
    "script_ratio",
    "image_ratio",
    "brand_domain_mismatch",
    "empty_navigation_ratio",
    "page_risk_after_mp",
    "relation_weighted_risk",
    "form_neighbor_risk",
    "input_neighbor_risk",
    "resource_neighbor_risk",
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
    parsed = urlsplit(u)
    host = parsed.hostname or ""
    if not host:
        return u
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        ascii_host = host
    netloc = ascii_host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username:
        userinfo = quote(parsed.username, safe="")
        if parsed.password:
            userinfo += ":" + quote(parsed.password, safe="")
        netloc = f"{userinfo}@{netloc}"
    path = quote(parsed.path or "", safe="/:%@!$&'()*+,;=-._~")
    query = quote(parsed.query or "", safe="=&?/:@!$'()*+,;%-._~")
    fragment = quote(parsed.fragment or "", safe="=&?/:@!$'()*+,;%-._~")
    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


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


def _brand_matches_domain(brand: str, host_parts: Set[str]) -> bool:
    allowed = BRAND_DOMAIN_ALIASES.get(brand, {brand})
    return bool(allowed.intersection(host_parts))


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
        "brand_word_ratio": _safe_ratio(brand_hits, len(tokens)),
        "suspicious_tld": 1.0 if any(host.endswith(tld) for tld in SUSPICIOUS_TLDS) else 0.0,
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
    fetch_method: str = "unknown"


def _decode_response_body(raw: bytes, headers: Dict[str, str], max_bytes: int) -> bytes:
    if headers.get("transfer-encoding", "").lower() == "chunked":
        out = bytearray()
        pos = 0
        while pos < len(raw) and len(out) < max_bytes:
            line_end = raw.find(b"\r\n", pos)
            if line_end < 0:
                break
            size_text = raw[pos:line_end].split(b";", 1)[0].strip()
            try:
                size = int(size_text, 16)
            except ValueError:
                break
            pos = line_end + 2
            if size == 0:
                break
            out.extend(raw[pos : pos + size])
            pos += size + 2
        return bytes(out[:max_bytes])
    return raw[:max_bytes]


def _parse_http_response(raw: bytes, max_bytes: int) -> Tuple[int, Dict[str, str], bytes]:
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1", errors="replace").split("\r\n")
    status = 0
    if lines:
        parts = lines[0].split()
        if len(parts) >= 2 and parts[1].isdigit():
            status = int(parts[1])
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    return status, headers, _decode_response_body(body, headers, max_bytes)


def _public_dns_ips(hostname: str) -> List[str]:
    ips: List[str] = []
    if dns is not None:
        for nameserver in PUBLIC_DNS_SERVERS:
            try:
                resolver = dns.resolver.Resolver(configure=False)
                resolver.nameservers = [nameserver]
                resolver.timeout = 2.0
                resolver.lifetime = 3.0
                answers = resolver.resolve(hostname, "A")
                ips.extend(str(answer) for answer in answers)
            except Exception:
                continue
    for nameserver in PUBLIC_DNS_SERVERS:
        try:
            result = subprocess.run(
                ["nslookup", hostname, nameserver],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except Exception:
            continue
        in_answer = False
        for line in (result.stdout or "").splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("name:"):
                in_answer = True
                continue
            if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", stripped):
                ips.append(stripped)
            elif in_answer and stripped.lower().startswith(("address:", "addresses:")):
                _, _, value = stripped.partition(":")
                value = value.strip()
                if re.match(r"^\d{1,3}(?:\.\d{1,3}){3}$", value):
                    ips.append(value)
    ordered: List[str] = []
    for ip in ips:
        if ip not in ordered:
            ordered.append(ip)
    return ordered


def _fetch_via_ip(
    url: str,
    ip: str,
    timeout: float,
    max_bytes: int,
    redirects_left: int = 3,
) -> FetchedPage:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    scheme = parsed.scheme or "http"
    port = parsed.port or (443 if scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query

    sock = socket.create_connection((ip, port), timeout=timeout)
    try:
        if scheme == "https":
            context = ssl.create_default_context()
            conn = context.wrap_socket(sock, server_hostname=host)
        else:
            conn = sock
        conn.settimeout(timeout)
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36\r\n"
            "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\r\n"
            "Accept-Language: ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii", errors="ignore")
        conn.sendall(req)
        chunks = bytearray()
        limit = max_bytes + 65536
        while len(chunks) < limit:
            data = conn.recv(65536)
            if not data:
                break
            chunks.extend(data)
    finally:
        try:
            sock.close()
        except Exception:
            pass

    status, headers, body = _parse_http_response(bytes(chunks), max_bytes)
    location = headers.get("location", "")
    if status in {301, 302, 303, 307, 308} and location and redirects_left > 0:
        next_url = urljoin(url, location)
        return _fetch_with_public_dns(next_url, timeout, max_bytes, redirects_left - 1)
    charset = "utf-8"
    content_type = headers.get("content-type", "")
    match = re.search(r"charset=([^;\s]+)", content_type, re.I)
    if match:
        charset = match.group(1)
    html = body.decode(charset, errors="replace")
    return FetchedPage(url, url, status, html, None, 0, "public_dns")


def _fetch_with_public_dns(
    url: str,
    timeout: float,
    max_bytes: int,
    redirects_left: int = 3,
) -> FetchedPage:
    host = urlsplit(url).hostname or ""
    if not host:
        return FetchedPage(url, url, 0, "", "public_dns:no_host", 0, "public_dns")
    ips = _public_dns_ips(host)
    last_error = "public_dns:no_ip"
    for ip in ips:
        try:
            return _fetch_via_ip(url, ip, timeout, max_bytes, redirects_left)
        except Exception as e:
            last_error = f"public_dns:{ip}:{type(e).__name__}:{e}"
    return FetchedPage(url, url, 0, "", last_error, 0, "public_dns")


def _fetch_with_curl_cffi(url: str, timeout: float, max_bytes: int) -> Optional[FetchedPage]:
    if curl_requests is None:
        return None
    try:
        response = curl_requests.get(
            url,
            impersonate="chrome116",
            timeout=timeout,
            allow_redirects=True,
        )
        raw = bytes(response.content[:max_bytes])
        html = raw.decode(response.encoding or "utf-8", errors="replace")
        return FetchedPage(
            url,
            str(response.url),
            int(response.status_code),
            html,
            None if response.status_code < 400 else f"HTTP {response.status_code}",
            0,
            "curl_cffi",
        )
    except Exception as e:
        return FetchedPage(url, url, 0, "", f"curl_cffi:{type(e).__name__}:{e}", 0, "curl_cffi")


def _fetch_with_requests(url: str, timeout: float, max_bytes: int) -> Optional[FetchedPage]:
    if requests is None:
        return None
    try:
        response = requests.get(
            url,
            timeout=timeout,
            allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        raw = response.content[:max_bytes]
        html = raw.decode(response.encoding or "utf-8", errors="replace")
        return FetchedPage(
            url,
            response.url,
            int(response.status_code),
            html,
            None if response.status_code < 400 else f"HTTP {response.status_code}",
            0,
            "requests",
        )
    except Exception as e:
        return FetchedPage(url, url, 0, "", f"requests:{type(e).__name__}:{e}", 0, "requests")


def _html_needs_browser(html: str) -> bool:
    if not html:
        return False
    parser = _StructureParser()
    try:
        parser.feed(html)
    except Exception:
        return False
    structural = len(parser.links) + len(parser.images) + len(parser.forms) + len(parser.inputs)
    return len(parser.scripts) >= 8 and structural <= 2


def _fetch_with_playwright(url: str, timeout: float, max_bytes: int) -> Optional[FetchedPage]:
    if os.getenv("GNN_USE_PLAYWRIGHT", "1") != "1":
        return None
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return None
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True,
                args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=int(timeout * 1000))
            page.wait_for_timeout(1500)
            html = page.content()[:max_bytes]
            final_url = page.url
            status = int(response.status) if response else 200
            browser.close()
            return FetchedPage(url, final_url, status, html, None, 0, "playwright")
    except Exception as e:
        return FetchedPage(url, url, 0, "", f"playwright:{type(e).__name__}:{e}", 0, "playwright")


def _maybe_browser_enhance(page: FetchedPage, timeout: float, max_bytes: int) -> FetchedPage:
    if page.error or page.status >= 400 or page.status == 0:
        return page
    if not _html_needs_browser(page.html):
        return page
    rendered = _fetch_with_playwright(page.final_url or page.requested_url, timeout, max_bytes)
    if rendered and not rendered.error and len(rendered.html) > len(page.html):
        return rendered
    return page


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
            return _maybe_browser_enhance(
                FetchedPage(normalized, final_url, status, html, None, redirects, "urlopen"),
                timeout,
                max_bytes,
            )
    except HTTPError as e:
        body = ""
        try:
            body = e.read(max_bytes).decode("utf-8", errors="replace")
        except Exception:
            body = ""
        page = FetchedPage(normalized, e.geturl() or normalized, int(e.code), body, str(e), 0, "urlopen")
    except (URLError, TimeoutError, socket.timeout, ssl.SSLError, OSError) as e:
        page = FetchedPage(normalized, normalized, 0, "", str(e), 0, "urlopen")

    for fallback in (
        _fetch_with_curl_cffi(normalized, timeout, max_bytes),
        _fetch_with_requests(normalized, timeout, max_bytes),
        _fetch_with_public_dns(normalized, timeout, max_bytes),
    ):
        if fallback and fallback.status and fallback.html and not fallback.error:
            return _maybe_browser_enhance(fallback, timeout, max_bytes)

    rendered = _fetch_with_playwright(normalized, timeout, max_bytes)
    if rendered and rendered.status and rendered.html and not rendered.error:
        return rendered
    return page


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
    fetch_method: str


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
    fetched = (
        fetch_page(normalized)
        if fetch
        else FetchedPage(normalized, normalized, 0, "", "fetch_disabled", 0, "disabled")
    )
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
    parsed_base = urlsplit(base_url)
    requested_base = urlsplit(normalized)
    visible_identity_text = " ".join(
        [
            requested_base.hostname or "",
            requested_base.path or "",
            parsed_base.hostname or "",
            parsed_base.path or "",
        ]
    )
    tokens = set(_tokenize(visible_identity_text))
    brands = {t for t in tokens if t in BRAND_WORDS}
    base_parts = set(_host_parts(base_domain))

    for brand in brands:
        risk = 0.75 if not _brand_matches_domain(brand, base_parts) else 0.05
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
        fetch_method=fetched.fetch_method,
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
    direct_edges = [(rel, dst) for src, rel, dst in graph.edges if src == "page:target"]
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
    brand_mismatch = (
        1.0
        if graph.brands and not any(_brand_matches_domain(brand, base_parts) for brand in graph.brands)
        else 0.0
    )
    risky_edges = sum(1 for _, _, dst in graph.edges if graph.node_risk.get(dst, 0.0) >= 0.5)
    input_count = max(1, counts.get("input", 0))
    password_input_ratio = _safe_ratio(counts.get("password_input", 0), input_count)
    suspicious_input_ratio = _safe_ratio(counts.get("suspicious_input", 0), input_count)
    form_count = _cap(counts.get("form", 0), 8.0)
    external_form_ratio = _safe_ratio(counts.get("submits_to_external", 0), total_forms)
    credential_surface = min(1.0, max(password_input_ratio, suspicious_input_ratio) + 0.10 * form_count)
    brand_capture_mismatch = brand_mismatch * credential_surface
    external_submission_risk = external_form_ratio * max(password_input_ratio, suspicious_input_ratio, form_count)

    relation_weights = {
        "submits_to": 1.25,
        "has_input": 1.10,
        "mentions_brand": 0.90,
        "embeds_iframe": 0.75,
        "loads_script": 0.50,
        "links_to": 0.30,
        "loads_image": 0.15,
    }

    def relation_mean(names: Set[str]) -> float:
        vals = [risks.get(dst, 0.0) for rel, dst in direct_edges if rel in names]
        return sum(vals) / len(vals) if vals else 0.0

    weighted_num = 0.0
    weighted_den = 0.0
    for rel, dst in direct_edges:
        weight = relation_weights.get(rel, 0.25)
        weighted_num += weight * risks.get(dst, 0.0)
        weighted_den += weight
    relation_weighted_risk = weighted_num / weighted_den if weighted_den else 0.0

    return {
        "html_fetched": 1.0 if graph.fetch_error is None and 200 <= graph.status < 400 and bool(graph.edges) else 0.0,
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
        "external_form_ratio": external_form_ratio,
        "password_input_ratio": password_input_ratio,
        "suspicious_input_ratio": suspicious_input_ratio,
        "credential_surface": credential_surface,
        "brand_capture_mismatch": brand_capture_mismatch,
        "external_submission_risk": external_submission_risk,
        "iframe_ratio": _safe_ratio(counts.get("iframe", 0), max(1, total_resources + total_links)),
        "script_ratio": _safe_ratio(counts.get("script", 0), max(1, total_resources + total_links)),
        "image_ratio": _safe_ratio(counts.get("image", 0), max(1, total_resources + total_links)),
        "brand_domain_mismatch": brand_mismatch,
        "empty_navigation_ratio": _safe_ratio(counts.get("empty_nav", 0), max(1, total_links + counts.get("empty_nav", 0))),
        "page_risk_after_mp": page_risk,
        "relation_weighted_risk": relation_weighted_risk,
        "form_neighbor_risk": relation_mean({"submits_to"}),
        "input_neighbor_risk": relation_mean({"has_input"}),
        "resource_neighbor_risk": relation_mean({"loads_script", "loads_image", "embeds_iframe"}),
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
    risk += 1.30 * features.get("credential_surface", 0.0)
    risk += 1.20 * features.get("external_submission_risk", 0.0)
    risk += 1.10 * features.get("brand_capture_mismatch", 0.0)
    risk += 0.90 * features.get("relation_weighted_risk", 0.0)
    risk += 1.15 * features.get("max_neighbor_risk", 0.0)
    risk += 0.95 * features.get("risky_edge_ratio", 0.0)
    risk += 0.90 * features.get("external_form_ratio", 0.0)
    risk += 0.70 * features.get("password_input_ratio", 0.0)
    risk += 0.45 * features.get("suspicious_input_ratio", 0.0)
    risk += 0.75 * features.get("brand_domain_mismatch", 0.0)
    risk += 0.35 * features.get("final_domain_changed", 0.0)
    risk += 0.30 * features.get("iframe_ratio", 0.0)
    risk += 0.20 * features.get("fetch_failed", 0.0)
    risk += 0.60 * features.get("suspicious_tld", 0.0)
    risk -= 0.45 * features.get("internal_link_ratio", 0.0)
    risk -= 0.20 * features.get("is_https", 0.0)
    return max(-1.25, min(1.25, risk - 0.65))


def _benign_structure_logit(features: Dict[str, float]) -> float:
    """Reduce false positives for normal first-party JS applications."""
    if (
        features.get("html_fetched", 0.0) <= 0.0
        and features.get("graph_edge_count", 0.0) <= 0.01
        and features.get("credential_surface", 0.0) == 0.0
    ):
        return -6.0
    if features.get("html_fetched", 0.0) <= 0.0:
        return 0.0
    has_capture_surface = (
        features.get("form_count", 0.0) > 0.0
        or features.get("password_input_ratio", 0.0) > 0.0
        or features.get("suspicious_input_ratio", 0.0) > 0.0
        or features.get("external_form_ratio", 0.0) > 0.0
        or features.get("credential_surface", 0.0) > 0.0
    )
    has_identity_mismatch = features.get("brand_domain_mismatch", 0.0) > 0.0
    has_external_risk = (
        features.get("external_resource_ratio", 0.0) > 0.15
        or features.get("risky_edge_ratio", 0.0) > 0.10
        or features.get("iframe_ratio", 0.0) > 0.05
    )
    first_party_spa = (
        features.get("script_ratio", 0.0) >= 0.80
        and features.get("external_resource_ratio", 0.0) <= 0.05
        and features.get("graph_edge_count", 0.0) >= 0.10
    )
    if first_party_spa and not has_capture_surface and not has_identity_mismatch and not has_external_risk:
        return -4.5
    rich_internal_page = (
        features.get("internal_link_ratio", 0.0) >= 0.65
        and features.get("external_form_ratio", 0.0) == 0.0
        and features.get("password_input_ratio", 0.0) == 0.0
        and not has_identity_mismatch
    )
    if rich_internal_page:
        return -1.25
    clean_first_party_login = (
        features.get("is_https", 0.0) > 0.0
        and features.get("form_count", 0.0) > 0.0
        and features.get("external_form_ratio", 0.0) == 0.0
        and features.get("brand_domain_mismatch", 0.0) == 0.0
        and features.get("brand_capture_mismatch", 0.0) == 0.0
        and features.get("risky_edge_ratio", 0.0) <= 0.08
        and features.get("page_risk_after_mp", 0.0) <= 0.18
        and features.get("relation_weighted_risk", 0.0) <= 0.22
        and features.get("external_resource_ratio", 0.0) <= 0.50
        and features.get("dot_count", 0.0) <= 0.30
        and features.get("hyphen_count", 0.0) <= 0.10
        and features.get("path_len", 0.0) <= 0.20
    )
    if clean_first_party_login:
        return -3.0
    no_capture_surface = (
        features.get("html_fetched", 0.0) > 0.0
        and features.get("form_count", 0.0) == 0.0
        and features.get("password_input_ratio", 0.0) == 0.0
        and features.get("suspicious_input_ratio", 0.0) == 0.0
        and features.get("external_form_ratio", 0.0) == 0.0
        and features.get("credential_surface", 0.0) == 0.0
        and features.get("brand_capture_mismatch", 0.0) == 0.0
    )
    if no_capture_surface:
        low_relation_risk = (
            features.get("brand_domain_mismatch", 0.0) == 0.0
            and features.get("external_resource_ratio", 0.0) <= 0.10
            and features.get("risky_edge_ratio", 0.0) <= 0.05
            and features.get("page_risk_after_mp", 0.0) <= 0.25
            and features.get("relation_weighted_risk", 0.0) <= 0.25
        )
        return -3.5 if low_relation_risk else -0.75
    return 0.0


GRAPH_NODE_TYPES = [
    "page",
    "url",
    "fetch",
    "link",
    "resource",
    "form",
    "input",
    "brand",
    "domain",
    "risk",
]
NODE_FEATURE_DIM = len(GRAPH_NODE_TYPES) + 8


@dataclass
class GraphSample:
    x: List[List[float]]
    edges: List[Tuple[int, int]]


def _node_features(node_type: str, attrs: Sequence[float]) -> List[float]:
    one_hot = [1.0 if node_type == t else 0.0 for t in GRAPH_NODE_TYPES]
    vals = [float(v) for v in attrs[:8]]
    vals.extend([0.0] * (8 - len(vals)))
    return one_hot + vals


def _edge_pair(edges: List[Tuple[int, int]], a: int, b: int) -> None:
    if a == b:
        return
    edges.append((a, b))
    edges.append((b, a))


def graph_sample_from_feature_map(fmap: Dict[str, float]) -> GraphSample:
    """Build a semantic phishing graph from URL/page-structure feature signals."""
    url_risk = min(
        1.0,
        0.25 * fmap.get("url_len", 0.0)
        + 0.20 * fmap.get("dot_count", 0.0)
        + 0.20 * fmap.get("hyphen_count", 0.0)
        + 0.25 * fmap.get("phish_word_ratio", 0.0)
        + 0.10 * fmap.get("digit_ratio", 0.0),
    )
    fetch_risk = max(
        fmap.get("fetch_failed", 0.0),
        fmap.get("status_bad", 0.0),
        0.65 * fmap.get("final_domain_changed", 0.0),
    )
    link_risk = min(
        1.0,
        0.55 * fmap.get("external_link_ratio", 0.0)
        + 0.45 * fmap.get("domain_diversity", 0.0),
    )
    resource_risk = min(
        1.0,
        0.65 * fmap.get("external_resource_ratio", 0.0)
        + 0.25 * fmap.get("iframe_ratio", 0.0)
        + 0.10 * fmap.get("script_ratio", 0.0),
    )
    form_risk = min(
        1.0,
        0.45 * fmap.get("form_count", 0.0)
        + 0.55 * fmap.get("external_form_ratio", 0.0),
    )
    input_risk = min(
        1.0,
        0.55 * fmap.get("password_input_ratio", 0.0)
        + 0.45 * fmap.get("suspicious_input_ratio", 0.0),
    )
    brand_risk = max(
        fmap.get("brand_domain_mismatch", 0.0),
        fmap.get("brand_capture_mismatch", 0.0),
    )
    domain_risk = max(
        fmap.get("final_domain_changed", 0.0),
        fmap.get("external_submission_risk", 0.0),
        fmap.get("domain_diversity", 0.0),
    )
    relation_risk = max(
        fmap.get("page_risk_after_mp", 0.0),
        fmap.get("relation_weighted_risk", 0.0),
        fmap.get("max_neighbor_risk", 0.0),
        fmap.get("risky_edge_ratio", 0.0),
    )

    nodes = [
        _node_features(
            "page",
            [
                relation_risk,
                fmap.get("graph_node_count", 0.0),
                fmap.get("graph_edge_count", 0.0),
                fmap.get("html_fetched", 0.0),
                fmap.get("is_https", 0.0),
                fmap.get("credential_surface", 0.0),
                fmap.get("brand_capture_mismatch", 0.0),
                fmap.get("external_submission_risk", 0.0),
            ],
        ),
        _node_features(
            "url",
            [
                url_risk,
                fmap.get("url_len", 0.0),
                fmap.get("host_len", 0.0),
                fmap.get("path_len", 0.0),
                fmap.get("entropy", 0.0),
                fmap.get("token_count", 0.0),
                fmap.get("phish_word_ratio", 0.0),
                fmap.get("brand_word_ratio", 0.0),
            ],
        ),
        _node_features(
            "fetch",
            [
                fetch_risk,
                fmap.get("html_fetched", 0.0),
                fmap.get("fetch_failed", 0.0),
                fmap.get("status_bad", 0.0),
                fmap.get("final_domain_changed", 0.0),
                fmap.get("redirect_count", 0.0),
                0.0,
                0.0,
            ],
        ),
        _node_features(
            "link",
            [
                link_risk,
                fmap.get("internal_link_ratio", 0.0),
                fmap.get("external_link_ratio", 0.0),
                fmap.get("empty_navigation_ratio", 0.0),
                fmap.get("domain_diversity", 0.0),
                0.0,
                0.0,
                0.0,
            ],
        ),
        _node_features(
            "resource",
            [
                resource_risk,
                fmap.get("external_resource_ratio", 0.0),
                fmap.get("iframe_ratio", 0.0),
                fmap.get("script_ratio", 0.0),
                fmap.get("image_ratio", 0.0),
                fmap.get("resource_neighbor_risk", 0.0),
                0.0,
                0.0,
            ],
        ),
        _node_features(
            "form",
            [
                form_risk,
                fmap.get("form_count", 0.0),
                fmap.get("external_form_ratio", 0.0),
                fmap.get("form_neighbor_risk", 0.0),
                fmap.get("external_submission_risk", 0.0),
                0.0,
                0.0,
                0.0,
            ],
        ),
        _node_features(
            "input",
            [
                input_risk,
                fmap.get("password_input_ratio", 0.0),
                fmap.get("suspicious_input_ratio", 0.0),
                fmap.get("credential_surface", 0.0),
                fmap.get("input_neighbor_risk", 0.0),
                0.0,
                0.0,
                0.0,
            ],
        ),
        _node_features(
            "brand",
            [
                brand_risk,
                fmap.get("brand_word_ratio", 0.0),
                fmap.get("brand_domain_mismatch", 0.0),
                fmap.get("brand_capture_mismatch", 0.0),
                fmap.get("credential_surface", 0.0),
                0.0,
                0.0,
                0.0,
            ],
        ),
        _node_features(
            "domain",
            [
                domain_risk,
                fmap.get("domain_diversity", 0.0),
                fmap.get("final_domain_changed", 0.0),
                fmap.get("external_submission_risk", 0.0),
                fmap.get("external_link_ratio", 0.0),
                fmap.get("external_resource_ratio", 0.0),
                0.0,
                0.0,
            ],
        ),
        _node_features(
            "risk",
            [
                relation_risk,
                fmap.get("page_risk_after_mp", 0.0),
                fmap.get("relation_weighted_risk", 0.0),
                fmap.get("max_neighbor_risk", 0.0),
                fmap.get("mean_neighbor_risk", 0.0),
                fmap.get("risk_spread", 0.0),
                fmap.get("risky_edge_ratio", 0.0),
                fmap.get("credential_surface", 0.0),
            ],
        ),
    ]
    edges: List[Tuple[int, int]] = []
    for idx in range(1, len(nodes)):
        _edge_pair(edges, 0, idx)
    _edge_pair(edges, 5, 6)  # form <-> input
    _edge_pair(edges, 5, 8)  # form <-> domain
    _edge_pair(edges, 7, 8)  # brand <-> domain
    _edge_pair(edges, 3, 8)  # links <-> domain
    _edge_pair(edges, 4, 8)  # resources <-> domain
    _edge_pair(edges, 9, 5)  # risk <-> form
    _edge_pair(edges, 9, 6)  # risk <-> input
    _edge_pair(edges, 9, 7)  # risk <-> brand
    return GraphSample(nodes, edges)


def graph_sample_for_url(url: str, fetch: bool = True) -> Tuple[GraphSample, WebGraph]:
    graph = build_web_graph(url, fetch=fetch)
    return graph_sample_from_feature_map(feature_map_from_graph(graph)), graph


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError(f"torch is required for the real GNN model: {_TORCH_IMPORT_ERROR}")


class GraphSAGEPhishingNet(_NN_MODULE):  # type: ignore[misc]
    def __init__(self, in_dim: int, hidden_dim: int = 48, dropout: float = 0.12):
        super().__init__()
        self.self1 = nn.Linear(in_dim, hidden_dim)
        self.neigh1 = nn.Linear(in_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.neigh2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _aggregate(self, x, edge_index):
        if edge_index.numel() == 0:
            return torch.zeros_like(x)
        src, dst = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, dst, x[src])
        deg = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        return out / deg.clamp_min(1.0).unsqueeze(1)

    def forward(self, x, edge_index):
        h = F.relu(self.self1(x) + self.neigh1(self._aggregate(x, edge_index)))
        h = self.dropout(h)
        h = F.relu(self.self2(h) + self.neigh2(self._aggregate(h, edge_index)))
        graph_emb = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=0)
        return self.classifier(graph_emb).squeeze(0)


def _sample_to_tensors(sample: GraphSample, device: str = "cpu"):
    _require_torch()
    x = torch.tensor(sample.x, dtype=torch.float32, device=device)
    if sample.edges:
        edge_index = torch.tensor(sample.edges, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    return x, edge_index


@dataclass
class WebStructureGNNModel:
    state_dict: Dict[str, Any]
    threshold: float
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        _require_torch()
        self.device = "cpu"
        self.net = GraphSAGEPhishingNet(
            int(self.metadata.get("node_feature_dim", NODE_FEATURE_DIM)),
            int(self.metadata.get("hidden_dim", 48)),
            float(self.metadata.get("dropout", 0.12)),
        )
        self.net.load_state_dict(self.state_dict)
        self.net.to(self.device)
        self.net.eval()

    @property
    def feature_names(self) -> List[str]:
        return list(FEATURE_NAMES)

    def predict_proba_one(self, url: str, fetch: bool = True) -> float:
        sample, _ = graph_sample_for_url(url, fetch=fetch)
        return self.predict_proba_from_sample(sample)

    def predict_proba_from_features(self, fmap: Dict[str, float]) -> float:
        return self.predict_proba_from_sample(graph_sample_from_feature_map(fmap))

    def predict_proba_from_sample(self, sample: GraphSample) -> float:
        x, edge_index = _sample_to_tensors(sample, self.device)
        with torch.no_grad():
            logit = self.net(x, edge_index)
            return float(torch.sigmoid(logit).item())

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
            "fetch_method": graph.fetch_method,
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
    epochs: int = 120,
    learning_rate: float = 0.003,
    l2: float = 0.001,
    threshold: float = 0.5,
) -> WebStructureGNNModel:
    if len(urls) != len(labels):
        raise ValueError("urls and labels length mismatch")
    if not urls:
        raise ValueError("empty training data")

    samples = [graph_sample_from_feature_map(feature_map_for_url(url, fetch=fetch_pages)) for url in urls]
    return train_web_structure_gnn_model_from_samples(
        samples,
        labels,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        metadata_extra={"fetch_pages_during_training": bool(fetch_pages)},
    )


def train_web_structure_gnn_model_from_vectors(
    vectors: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    epochs: int = 120,
    learning_rate: float = 0.003,
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
    samples = [
        graph_sample_from_feature_map({name: value for name, value in zip(FEATURE_NAMES, row)})
        for row in raw_vectors
    ]
    metadata = {"trained_from_stored_features": True}
    if metadata_extra:
        metadata.update(metadata_extra)
    return train_web_structure_gnn_model_from_samples(
        samples,
        labels,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        threshold=threshold,
        metadata_extra=metadata,
    )


def train_web_structure_gnn_model_from_samples(
    samples: Sequence[GraphSample],
    labels: Sequence[int],
    *,
    epochs: int = 120,
    learning_rate: float = 0.003,
    l2: float = 0.001,
    threshold: float = 0.65,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> WebStructureGNNModel:
    _require_torch()
    if len(samples) != len(labels):
        raise ValueError("samples and labels length mismatch")
    if not samples:
        raise ValueError("empty training data")
    torch.manual_seed(42)
    net = GraphSAGEPhishingNet(NODE_FEATURE_DIM)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=l2)
    criterion = nn.BCEWithLogitsLoss()
    order = list(range(len(samples)))
    for epoch in range(max(1, epochs)):
        random.Random(42 + epoch).shuffle(order)
        net.train()
        for idx in order:
            x, edge_index = _sample_to_tensors(samples[idx])
            y = torch.tensor(float(labels[idx]), dtype=torch.float32)
            optimizer.zero_grad()
            loss = criterion(net(x, edge_index), y)
            loss.backward()
            optimizer.step()
    metadata = {
        "kind": MODEL_KIND,
        "artifact_version": ARTIFACT_VERSION,
        "feature_names": FEATURE_NAMES,
        "graph_node_types": GRAPH_NODE_TYPES,
        "node_feature_dim": NODE_FEATURE_DIM,
        "hidden_dim": 48,
        "dropout": 0.12,
        "training_rows": len(samples),
        "model_family": "GraphSAGE",
        "description": "Real Torch GraphSAGE GNN over URL/page-structure phishing graph nodes.",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    return WebStructureGNNModel(
        {k: v.detach().cpu() for k, v in net.state_dict().items()},
        threshold,
        metadata,
    )


def save_gnn_artifact(model: WebStructureGNNModel, model_path: str, features_path: Optional[str] = None) -> None:
    artifact = {
        "kind": MODEL_KIND,
        "version": ARTIFACT_VERSION,
        "state_dict": model.state_dict,
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
        state_dict=dict(artifact["state_dict"]),
        threshold=float(artifact.get("threshold", 0.5)),
        metadata=dict(artifact.get("metadata", {})),
    )
def load_gnn_model(model_path: str, features_path: Optional[str] = None):
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)

    model = _artifact_to_model(artifact)

    if features_path and os.path.isfile(features_path):
        with open(features_path, "rb") as f:
            columns = pickle.load(f)
    else:
        columns = FEATURE_NAMES

    return model, columns

def _explain_gnn_load_error(exc: Exception) -> str:
    return (
        "Failed to load web-structure GNN artifact. Regenerate it with "
        "regenerate_gnn_model.py so gnn_model.pkl contains a "
        f"{MODEL_KIND} bundle. Detail: {type(exc).__name__}: {exc}"
    )

def build_explanation(
    evidence: Dict[str, Any],
    features: Optional[Dict[str, float]] = None,
    prob: float = 0.0,
) -> str:

    if not evidence:
        return "보안 분석 정보를 불러오지 못했습니다."

    lines = []
    counts = evidence.get("counts", {})
    password_count = counts.get("password_input", 0)
    form_count = counts.get("form", 0)
    iframe_count = counts.get("iframe", 0)
    external_links = counts.get("links_to_external", 0)
    internal_links = counts.get("links_to_internal", 0)

    # =========================================================
    # 1. 최종 판정
    # =========================================================
    
    if prob >= 0.85:
        lines.append("🚨 매우 위험한 사이트입니다.")
    elif prob >= 0.60:
        lines.append("⚠️ 위험한 사이트입니다.")
    elif prob >= 0.35:
        lines.append("🟡 수상한 사이트입니다.")
    else:
        return "✅ 안전한 사이트입니다. 위험한 요소가 발견되지 않았습니다."

    if features:
        if features.get("brand_domain_mismatch", 0) > 0:
            lines.append("AI 구조 분석 결과, 유명 브랜드 이름을 도용하고 있지만 실제 공식 주소와 일치하지 않는 가짜 사이트로 확인되었습니다.")
        elif features.get("external_form_ratio", 0) > 0.4:
            lines.append("AI 구조 분석 결과, 이 사이트에 입력한 정보가 전혀 관계없는 외부 서버로 전송되는 구조가 확인되었습니다.")
        elif features.get("final_domain_changed", 0) > 0:
            lines.append("AI 구조 분석 결과, 접속 과정에서 전혀 다른 주소로 자동 이동되는 피싱 수법이 감지되었습니다.")
        elif iframe_count > 0:
            lines.append("AI 구조 분석 결과, 사용자 눈에 보이지 않는 숨겨진 화면이 포함되어 있어 악성 사이트와 유사한 구조로 판단되었습니다.")
        elif counts.get("script", 0) > 5:
            lines.append("AI 구조 분석 결과, 외부에서 불러온 수상한 프로그램 코드가 다수 실행되고 있어 피싱 사이트와 유사한 패턴으로 판단되었습니다.")
        elif external_links > 2:
            lines.append("AI 구조 분석 결과, 정상 사이트에 비해 수상한 외부 주소와의 연결이 과도하게 많아 위험한 사이트로 판단되었습니다.")
        else:
            lines.append("AI 구조 분석 결과, 사이트 전체 연결 구조와 동작 패턴이 알려진 피싱 사이트와 유사하여 위험한 사이트로 판단되었습니다.")
        return " ".join(lines)


def predict_gnn(
    model: WebStructureGNNModel,
    column_order: List[str],
    raw_url: str,
) -> Dict[str, Any]:
    url = _normalize_url(raw_url)
    fetch = os.getenv("GNN_FETCH_PAGE", "1") != "0"
    sample, graph = graph_sample_for_url(url, fetch=fetch)
    prob_mal = float(model.predict_proba_from_sample(sample))
    fmap_tmp = feature_map_from_graph(graph) if fetch else {}
    
    if fmap_tmp.get("suspicious_tld", 0.0) > 0:
        prob_mal = min(1.0, prob_mal + 0.40)
        label = 1
    else:
        label = 1 if prob_mal >= model.threshold else 0
    out = {
        "url": url,
        "probability": round(prob_mal, 6),
        "label": label,
        "verdict": "malicious" if label == 1 else "benign",
        "model_type": MODEL_KIND,
    }
    if fetch:
        evidence = model.evidence_from_graph(graph)
        out["graph_evidence"] = evidence

                # 시각화용 그래프 데이터 생성
        out["graph_visual"] = {
            "nodes": list(graph.nodes),
            "edges": [
                {
                    "source": src,
                    "relation": rel,
                    "target": dst
                }
                for src, rel, dst in graph.edges
            ]
        }

        fmap = feature_map_from_graph(graph)
        out["explanation"] = build_explanation(evidence, fmap, prob_mal)
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
