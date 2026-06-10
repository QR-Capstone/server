from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import functools
import html
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = {
    "gnn": os.path.join(BASE_DIR, "gnn"),
    "xgboost": os.path.join(BASE_DIR, "xgboost"),
    "KoBERT": os.path.join(BASE_DIR, "KoBERT"),
    "url_ml": os.path.join(BASE_DIR, "url_ml"),
}
for _model_dir in MODEL_DIRS.values():
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)

# Tuning (same env vars as engine module): MAX_SEQ_LEN, USE_PLAYWRIGHT_IN_ANALYZE, etc.
# os.environ.setdefault("MAX_SEQ_LEN", "128")
# os.environ.setdefault("USE_PLAYWRIGHT_IN_ANALYZE", "0")
# os.environ.setdefault("TORCH_NUM_THREADS", "4")

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from XG_core import (
    build_all_explanations,
    load_bundle,
    predict_url,
    predict_url_dom,
    runtime_config as xg_runtime_config,
    xgboost_weighted_ensemble_verdict,
)
from gnn_engine import GNN_Engine, predict_gnn
from gnn_engine import runtime_config as gnn_runtime_config
from url_ml_engine import load_url_ml_model, predict_url_ml, predict_url_ml_batch

try:
    from trusted_domains import (
        is_trusted_official_url,
        is_low_risk_hosted_platform_url,
        strong_url_phishing_score,
        url_heuristic_phishing_score,
    )
except Exception:  # pragma: no cover
    def is_trusted_official_url(raw_url: str) -> bool:
        return False

    def is_low_risk_hosted_platform_url(raw_url: str) -> bool:
        return False

    def strong_url_phishing_score(raw_url: str) -> float:
        return 0.0

    def url_heuristic_phishing_score(raw_url: str) -> float:
        return 0.0

app = FastAPI(title="Phishing Detection API")


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Wall time for the full request (logged next to HTTP OK)."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dur = time.perf_counter() - t0
    response.headers["X-Process-Time"] = f"{dur:.3f}"
    if LOG_REQUEST_TIMING:
        print(
            f"--- [request done] {request.method} {request.url.path} "
            f"{response.status_code} OK ({dur:.3f}s)"
        )
    return response

# Warm up Playwright on startup (default off = faster boot)
STARTUP_WARMUP_PLAYWRIGHT = os.getenv("STARTUP_WARMUP_PLAYWRIGHT", "0") == "1"
STARTUP_SMOKE_GNN = os.getenv("STARTUP_SMOKE_GNN", "0") == "1"
LOG_REQUEST_TIMING = os.getenv("LOG_REQUEST_TIMING", "0") == "1"
ANALYZE_VERBOSE_LOGS = os.getenv("ANALYZE_VERBOSE_LOGS", "0") == "1"
URL_ML_CACHE_SIZE = int(os.getenv("URL_ML_CACHE_SIZE", "4096"))
KOBERT_CACHE_SIZE = int(os.getenv("KOBERT_CACHE_SIZE", "512"))
URL_RULE_CACHE_SIZE = int(os.getenv("URL_RULE_CACHE_SIZE", "8192"))
XGBOOST_CACHE_SIZE = int(os.getenv("XGBOOST_CACHE_SIZE", "1024"))
GNN_CACHE_SIZE = int(os.getenv("GNN_CACHE_SIZE", "512"))
XGBOOST_WORKERS = max(1, int(os.getenv("XGBOOST_WORKERS", "2")))
GNN_WORKERS = max(1, int(os.getenv("GNN_WORKERS", "2")))
SUPPRESS_KOBERT_DEBUG_LOGS = os.getenv("SUPPRESS_KOBERT_DEBUG_LOGS", "1") == "1"
SUPPRESS_XGBOOST_DEBUG_LOGS = os.getenv("SUPPRESS_XGBOOST_DEBUG_LOGS", "1") == "1"
URL_ML_FAST_PATH = os.getenv("URL_ML_FAST_PATH", "1") == "1"
URL_ML_FAST_SAFE_MAX = float(os.getenv("URL_ML_FAST_SAFE_MAX", "0.20"))
URL_ML_FAST_HEURISTIC_MAX = float(os.getenv("URL_ML_FAST_HEURISTIC_MAX", "0.05"))
URL_ML_FAST_DANGER_MIN = float(os.getenv("URL_ML_FAST_DANGER_MIN", "0.68"))
URL_ML_FINAL_SOLO_THRESHOLD = float(os.getenv("URL_ML_FINAL_SOLO_THRESHOLD", "0.90"))
URL_ML_FINAL_SUPPORT_THRESHOLD = float(os.getenv("URL_ML_FINAL_SUPPORT_THRESHOLD", "0.66"))
ANALYZE_BATCH_MAX_URLS = max(1, int(os.getenv("ANALYZE_BATCH_MAX_URLS", "256")))
ANALYZE_BATCH_WORKERS = max(1, int(os.getenv("ANALYZE_BATCH_WORKERS", "16")))
ANALYZE_TEXT_MAX_CHARS = max(1, int(os.getenv("ANALYZE_TEXT_MAX_CHARS", "20000")))
URL_ML_BATCH_VECTOR_MIN = max(1, int(os.getenv("URL_ML_BATCH_VECTOR_MIN", "17")))
KOBERT_URL_ML_PREFLIGHT = os.getenv("KOBERT_URL_ML_PREFLIGHT", "1") == "1"
URL_IN_TEXT_RE = re.compile(
    r"(?i)\b((?:hxxps?://|https?://|www\.)[^\s<>'\"`]+|[a-z0-9][a-z0-9.-]+\.[a-z]{2,}(?:/[^\s<>'\"`]*)?)"
)
BASE64_TEXT_TOKEN_RE = re.compile(r"(?<![a-zA-Z0-9+/_=-])([a-zA-Z0-9+/_-]{16,}={0,2})(?![a-zA-Z0-9+/_=-])")
HEX_TEXT_TOKEN_RE = re.compile(r"(?<![a-zA-Z0-9])([a-fA-F0-9]{32,4096})(?![a-zA-Z0-9])")
JS_FROM_CHAR_CODE_RE = re.compile(r"(?i)\b(?:String\.)?fromCharCode\s*\(([^)]{16,4096})\)")
JS_EMPTY_JOIN_ARRAY_RE = re.compile(
    r"""(?is)\[((?:\s*["'][^"']{1,256}["']\s*,?){2,40})\]\s*\.join\s*\(\s*(["'])\2\s*\)"""
)
JS_REVERSED_EMPTY_JOIN_ARRAY_RE = re.compile(
    r"""(?is)\[((?:\s*["'][^"']{1,256}["']\s*,?){2,40})\]\s*\.join\s*\(\s*(["'])\2\s*\)\s*\.split\s*\(\s*(["'])\3\s*\)\s*\.reverse\s*\(\s*\)\s*\.join\s*\(\s*(["'])\4\s*\)"""
)
JS_SIMPLE_STRING_LITERAL_RE = re.compile(r"""(?s)(["'])([^"']{1,256})\1""")
JS_REVERSED_STRING_RE = re.compile(
    r"""(?is)(["'])([^"']{8,4096})\1\s*\.split\s*\(\s*(["'])\3\s*\)\s*\.reverse\s*\(\s*\)\s*\.join\s*\(\s*(["'])\4\s*\)"""
)
JS_REVERSE_SUFFIX_RE = re.compile(r"""(?is)^\s*\.split\s*\(\s*(["'])\1\s*\)\s*\.reverse\s*\(\s*\)\s*\.join\s*\(\s*(["'])\2\s*\)""")
JS_FUNCTION_URL_FALSE_POSITIVE_RE = re.compile(r"(?i)^(?:String\.)?fromCharCode$")
REDIRECT_QUERY_PARAM_NAMES = frozenset(
    {
        "continue",
        "dest",
        "destination",
        "go",
        "link",
        "next",
        "r",
        "redirect",
        "redirect_to",
        "redirect_uri",
        "return",
        "return_to",
        "returnurl",
        "target",
        "targeturl",
        "to",
        "u",
        "uri",
        "url",
    }
)
DEFANGED_DOT_RE = re.compile(r"(?i)\s*(?:\[\.\]|\(\.\)|\{\.\}|<\.>|\[dot\]|\(dot\)|\{dot\}| dot )\s*")
DEFANGED_COLON_RE = re.compile(r"(?i)\s*(?:\[:\]|\(:\)|\{:\}|<:>|\[colon\]|\(colon\)|\{colon\}|colon|콜론|쌍점)\s*")
DEFANGED_SLASH_RE = re.compile(r"(?i)\s*(?:\[/\]|\(/\)|\{/}|</>|\[slash\]|\(slash\)|\{slash\}|slash|슬래시)\s*")
DEFANGED_AT_RE = re.compile(r"(?i)\s*(?:\[@\]|\(@\)|\{@\}|<@>|\[at\]|\(at\)|\{at\}| at )\s*")
HOST_SPACED_DOT_RE = re.compile(r"(?i)(?<=[a-z0-9])\s+\.\s+(?=[a-z0-9])")
KOREAN_DEFANGED_DOT_RE = re.compile(r"(?i)(?<=[a-z0-9])\s*(?:점|닷|쩜)\s*(?=[a-z0-9])")
KOREAN_SPACED_DEFANGED_DOT_RE = re.compile(r"(?i)(?<=[a-z0-9])\s+(?:점|닷|쩜)\s+(?=[a-z0-9])")
ESCAPED_URL_PUNCT_RE = re.compile(r"\\+([./:])")
ASCII_HEX_ESCAPE_RE = re.compile(r"\\x([0-7][0-9a-fA-F])")
ASCII_UNICODE_ESCAPE_RE = re.compile(r"(?:\\u|%u)00([0-7][0-9a-fA-F])")
ASCII_CODEPOINT_ESCAPE_RE = re.compile(r"\\u\{0*([2-7][0-9a-fA-F])\}")
ASCII_JS_OCTAL_ESCAPE_RE = re.compile(r"\\([1-7][0-7]{2}|0[1-7][0-7])")
ASCII_CSS_ZERO_PADDED_ESCAPE_RE = re.compile(r"\\0{1,4}([2-7][0-9a-fA-F])\s?")
ASCII_CSS_HEX_ESCAPE_RE = re.compile(r"\\([2-7][0-9a-fA-F])\s?")
REMAINING_ESCAPE_TOKEN_RE = re.compile(r"(?:\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|%u[0-9a-fA-F]{4})+")
ASCII_QUOTED_PRINTABLE_RE = re.compile(r"=([0-7][0-9a-fA-F])")
QUOTED_PRINTABLE_SOFT_BREAK_RE = re.compile(r"=\r?\n[ \t]*")
REMAINING_QUOTED_PRINTABLE_RE = re.compile(r"(?:=[0-9a-fA-F]{2})+")
SPACED_HTTPS_SCHEME_RE = re.compile(r"(?i)\bh\s*t\s*t\s*p\s*s\s*:\s*/\s*/")
SPACED_HTTP_SCHEME_RE = re.compile(r"(?i)\bh\s*t\s*t\s*p\s*:\s*/\s*/")
SPACED_HXXPS_SCHEME_RE = re.compile(r"(?i)\bh\s*x\s*x\s*p\s*s\s*:\s*/\s*/")
SPACED_HXXP_SCHEME_RE = re.compile(r"(?i)\bh\s*x\s*x\s*p\s*:\s*/\s*/")
SPACED_WWW_RE = re.compile(r"(?i)\bw\s*w\s*w\s*(?=\.)")
SPACED_URL_PREFIX_RE = re.compile(r"(?i)(?:https?://|www\.)")
SPACED_URL_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_/~%?=&+#")
SPACED_URL_BOUNDARY_CHARS = set("./-_~%?=&+#")
WRAPPED_WWW_AFTER_SCHEME_RE = re.compile(r"(?i)\b(https?://)\s*[\[(<{]\s*www\s*[\])>}]\s*(?=\.)")
STRING_CONCAT_BREAK_RE = re.compile(r"""(?<=[a-z0-9./:_\]\)-])["']\s*\+\s*["'](?=[a-z0-9\[\(.-])""", re.IGNORECASE)
STRING_LITERAL_ADJACENT_RE = re.compile(r"""(?<=[a-z0-9./:_-])["']\s+["'](?=[a-z0-9.-])""", re.IGNORECASE)
BACKSLASH_LINE_CONTINUATION_RE = re.compile(r"\\(?:r\\n|n|r|[ \t]*\r?\n)[ \t]*")
ZERO_WIDTH_TEXT_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
TEXT_URL_TRANSLATION = str.maketrans(
    {
        "\uff0e": ".",
        "\u3002": ".",
        "\uff61": ".",
        "\uff1a": ":",
        "\ufe55": ":",
        "\uff0f": "/",
        "\u2215": "/",
        "\u2044": "/",
    }
)

# Playwright sync API is bound to one thread; engine runs on a single worker.
_engine_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phish_engine")
# XGBoost on its own thread pool (runs in parallel with KoBERT).
_xgboost_executor = ThreadPoolExecutor(max_workers=XGBOOST_WORKERS, thread_name_prefix="xgboost_infer")
# Web-structure GNN (gnn_model.pkl)
_gnn_executor = ThreadPoolExecutor(max_workers=GNN_WORKERS, thread_name_prefix="gnn_webgraph")


class _NullWriter:
    def write(self, _: str) -> int:
        return 0

    def flush(self) -> None:
        return None


_NULL_WRITER = _NullWriter()
_NO_ADJUSTMENT = object()
_INFLIGHT_INFERENCE: dict[tuple[str, str], asyncio.Task] = {}


def _cache_stats(fn: Any) -> dict[str, int]:
    info = fn.cache_info()
    return {
        "hits": int(info.hits),
        "misses": int(info.misses),
        "maxsize": int(info.maxsize or 0),
        "currsize": int(info.currsize),
    }


def _runtime_cache_stats() -> dict[str, dict[str, int]]:
    return {
        "url_key": _cache_stats(_url_cache_key),
        "url_ml": _cache_stats(_predict_url_ml_cached),
        "kobert": _cache_stats(_predict_kobert_cached),
        "xgboost": _cache_stats(_predict_xgboost_cached),
        "gnn": _cache_stats(_predict_gnn_cached),
        "url_rule": _cache_stats(_url_rule_adjustment),
        "url_heuristic": _cache_stats(_url_heuristic_result),
    }


def _runtime_inflight_stats() -> dict[str, int]:
    counts = {"kobert": 0, "xgboost": 0, "gnn": 0}
    for model, _ in _INFLIGHT_INFERENCE:
        counts[model] = counts.get(model, 0) + 1
    counts["total"] = sum(counts.values())
    return counts


def _clear_runtime_caches() -> None:
    for fn in (
        _url_cache_key,
        _predict_url_ml_cached,
        _predict_kobert_cached,
        _predict_xgboost_cached,
        _predict_gnn_cached,
        _url_rule_adjustment,
        _url_heuristic_result,
        is_trusted_official_url,
        is_low_risk_hosted_platform_url,
        strong_url_phishing_score,
        url_heuristic_phishing_score,
    ):
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()
    _INFLIGHT_INFERENCE.clear()


async def _singleflight_inference(model: str, key: str, factory) -> Any:
    token = (model, key)
    task = _INFLIGHT_INFERENCE.get(token)
    if task is None:
        task = asyncio.create_task(factory())
        _INFLIGHT_INFERENCE[token] = task
    try:
        return await task
    finally:
        if _INFLIGHT_INFERENCE.get(token) is task:
            _INFLIGHT_INFERENCE.pop(token, None)


@functools.lru_cache(maxsize=URL_RULE_CACHE_SIZE)
def _url_cache_key(raw_url: str, *, default_scheme: str | None = None) -> str:
    raw = (raw_url or "").strip()
    if not raw:
        return raw
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
    except Exception:
        return raw
    host = parsed.hostname
    if not host:
        return raw
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    netloc = host.lower()
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        netloc = f"{auth}@{netloc}"
    scheme = (parsed.scheme or default_scheme or "").lower()
    if not scheme:
        return urlunsplit(("", netloc, parsed.path, parsed.query, parsed.fragment))[2:]
    return urlunsplit((scheme, netloc, parsed.path, parsed.query, parsed.fragment))


async def _run_engine(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_engine_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_engine_executor, functools.partial(fn, *args))


async def _run_xgboost(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_xgboost_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_xgboost_executor, functools.partial(fn, *args))


async def _run_gnn(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_gnn_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_gnn_executor, functools.partial(fn, *args))


async def _run_url_ml(fn, *args, **kwargs):
    if kwargs:
        return fn(*args, **kwargs)
    return fn(*args)


def _run_kobert_inference(raw_url: str):
    result = _predict_kobert_cached(_url_cache_key(raw_url, default_scheme="https"))
    return dict(result) if isinstance(result, dict) else result


async def _run_kobert_request(raw_url: str):
    key = _url_cache_key(raw_url, default_scheme="https")
    result = await _singleflight_inference(
        "kobert",
        key,
        lambda: _run_engine(_predict_kobert_cached, key),
    )
    return dict(result) if isinstance(result, dict) else result


@functools.lru_cache(maxsize=KOBERT_CACHE_SIZE)
def _predict_kobert_cached(raw_url: str):
    eng = getattr(app.state, "eng", None)
    if eng is None:
        return {
            "judgment": "unknown",
            "riskLevel": "UNKNOWN",
            "risklevel": "UNKNOWN",
            "engine_disabled": True,
            "engine_reason": getattr(app.state, "eng_status", {}).get("reason", "not_loaded"),
        }
    if SUPPRESS_KOBERT_DEBUG_LOGS:
        with contextlib.redirect_stdout(_NULL_WRITER), contextlib.redirect_stderr(_NULL_WRITER):
            return eng.predict_phishing_result(raw_url)
    return eng.predict_phishing_result(raw_url)


def _normalize_url_for_xgboost(url: str) -> str:
    """Normalize URL for XGBoost; prepend https if no scheme."""
    candidate = (url or "").strip()
    if not candidate:
        return candidate
    parsed = urlsplit(candidate)
    if parsed.scheme and parsed.netloc:
        return candidate
    return f"https://{candidate}"


def _run_xgboost_inference(raw_url: str):
    """Run XGBoost bundles (typo, domain-age, DOM); returns dict or None if no models."""
    result = _predict_xgboost_cached(_url_cache_key(raw_url, default_scheme="https"))
    return dict(result) if isinstance(result, dict) else result


async def _run_xgboost_request(raw_url: str):
    key = _url_cache_key(raw_url, default_scheme="https")
    result = await _singleflight_inference(
        "xgboost",
        key,
        lambda: _run_xgboost(_predict_xgboost_cached, key),
    )
    return dict(result) if isinstance(result, dict) else result


@functools.lru_cache(maxsize=XGBOOST_CACHE_SIZE)
def _predict_xgboost_cached(raw_url: str):
    if SUPPRESS_XGBOOST_DEBUG_LOGS:
        with contextlib.redirect_stdout(_NULL_WRITER), contextlib.redirect_stderr(_NULL_WRITER):
            return _predict_xgboost_uncached(raw_url)
    return _predict_xgboost_uncached(raw_url)


def _predict_xgboost_uncached(raw_url: str):
    typo_bundle = getattr(app.state, "xg_typo_bundle", None)
    domain_bundle = getattr(app.state, "xg_domain_bundle", None)
    dom_bundle = getattr(app.state, "xg_dom_bundle", None)
    if typo_bundle is None and domain_bundle is None and dom_bundle is None:
        return None

    url = _normalize_url_for_xgboost(raw_url)
    output = {"url": url}

    if typo_bundle is not None:
        typo_label, typo_prob, typo_feature_map = predict_url(
            typo_bundle,
            url,
            enable_domain_age=False,
            enable_ssl=bool(typo_bundle.meta.get("enable_ssl", False)),
            domain_only=False,
        )
        output["typo_probability"] = round(float(typo_prob), 6)
        output["typo_label"] = int(typo_label)
    else:
        typo_feature_map = {}

    if domain_bundle is not None:
        domain_label, domain_prob, domain_feature_map = predict_url(
            domain_bundle,
            url,
            enable_domain_age=True,
            enable_ssl=bool(domain_bundle.meta.get("enable_ssl", False)),
            domain_only=True,
        )
        output["domain_probability"] = round(float(domain_prob), 6)
        output["domain_label"] = int(domain_label)
    else:
        domain_feature_map = {}

    if dom_bundle is not None:
        dom_label, dom_prob, dom_feature_map = predict_url_dom(
            dom_bundle,
            url,
            print_dom_feature_debug=False,
        )
        output["dom_probability"] = round(float(dom_prob), 6)
        output["dom_label"] = int(dom_label)
        output["dom_features"] = {
            k: round(float(v), 6) for k, v in dom_feature_map.items()
        }
    else:
        dom_feature_map = {}

    typo_prob = float(output.get("typo_probability", 0.0))
    domain_prob = float(output.get("domain_probability", 0.0))
    dom_prob = float(output.get("dom_probability", 0.0))
    # CLI(XG_infer)와 동일한 가중치·게이트 (XG_core.xgboost_weighted_ensemble_verdict)
    final_prob, verdict_label = xgboost_weighted_ensemble_verdict(
        typo_prob, domain_prob, dom_prob
    )
    output["final_probability"] = round(float(final_prob), 6)
    output["label"] = int(verdict_label)
    output["verdict"] = "malicious" if verdict_label == 1 else "benign"
    explain_threshold = float(os.getenv("XG_EXPLAIN_THRESHOLD", "0.5"))
    output["explanations"] = build_all_explanations(
        url=url,
        typo_feat_map=typo_feature_map,
        typo_probability=typo_prob if typo_bundle is not None else 0.0,
        domain_feat_map=domain_feature_map,
        domain_probability=domain_prob if domain_bundle is not None else 0.0,
        dom_feature_map=dom_feature_map,
        dom_probability=dom_prob if dom_bundle is not None else 0.0,
        verdict_label=int(verdict_label),
        threshold=explain_threshold,
    )
    return output


def _run_gnn_inference(raw_url: str):
    """Web-structure GNN (gnn_model.pkl); None if model not loaded."""
    result = _predict_gnn_cached(_url_cache_key(raw_url, default_scheme="https"))
    return dict(result) if isinstance(result, dict) else result


async def _run_gnn_request(raw_url: str):
    key = _url_cache_key(raw_url, default_scheme="https")
    result = await _singleflight_inference(
        "gnn",
        key,
        lambda: _run_gnn(_predict_gnn_cached, key),
    )
    return dict(result) if isinstance(result, dict) else result


@functools.lru_cache(maxsize=GNN_CACHE_SIZE)
def _predict_gnn_cached(raw_url: str):
    model = getattr(app.state, "gnn_model", None)
    cols = getattr(app.state, "gnn_columns", None)
    if model is None or not cols:
        return None
    try:
        return predict_gnn(model, cols, raw_url)
    except Exception as e:
        return {"error": str(e), "verdict": "unknown", "enabled": True}


def _run_url_ml_inference(raw_url: str):
    model = getattr(app.state, "url_ml_model", None)
    if model is None:
        return None
    return _predict_url_ml_cached(_url_cache_key(raw_url))


@functools.lru_cache(maxsize=URL_ML_CACHE_SIZE)
def _predict_url_ml_cached(raw_url: str):
    model = getattr(app.state, "url_ml_model", None)
    if model is None:
        return None
    return predict_url_ml(model, raw_url)


def _log_line_kobert(result: dict) -> str:
    j = result.get("judgment", "?")
    rl = result.get("riskLevel", result.get("risklevel", "?"))
    if result.get("engine_disabled"):
        return f"KoBERT: disabled ({result.get('engine_reason', '')})"
    return f"KoBERT: judgment={j} riskLevel={rl}"


def _log_line_xgboost(xg: object) -> str:
    if xg is None:
        return "XGBoost: skipped (no model)"
    parts = [f"verdict={xg.get('verdict')}", f"final_p={xg.get('final_probability')}"]
    if xg.get("dom_probability") is not None:
        parts.append(f"dom_p={xg.get('dom_probability')}")
    return "XGBoost: " + " ".join(parts)


def _log_line_gnn(gnn: object) -> str:
    if gnn is None:
        st = getattr(app.state, "gnn_status", {}) or {}
        reason = st.get("reason") or "no_model_loaded"
        return f"GNN(web graph): skipped — {reason}"
    if isinstance(gnn, dict) and gnn.get("error"):
        return f"GNN(web graph): error {gnn.get('error', '')[:80]}"
    return f"GNN(web graph): verdict={gnn.get('verdict')} p={gnn.get('probability')}"


def _log_line_url_ml(url_ml: object) -> str:
    if url_ml is None:
        st = getattr(app.state, "url_ml_status", {}) or {}
        reason = st.get("reason") or "no_model_loaded"
        return f"URLML: skipped — {reason}"
    if isinstance(url_ml, dict) and url_ml.get("error"):
        return f"URLML: error {url_ml.get('error', '')[:80]}"
    return f"URLML: verdict={url_ml.get('verdict')} p={url_ml.get('probability')}"


def _risk_from_model_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"normal", "safe", "low", "benign", "clean", "allow", "allowed", "0"}:
        return "SAFE"
    if text in {
        "unnormal",
        "abnormal",
        "high",
        "danger",
        "dangerous",
        "malicious",
        "phishing",
        "scam",
        "block",
        "blocked",
        "1",
    }:
        return "DANGEROUS"
    return "UNKNOWN"


def _verdict_from_risk(risk_level: str) -> str:
    if risk_level == "SAFE":
        return "benign"
    if risk_level == "DANGEROUS":
        return "malicious"
    return "unknown"


def _judgment_from_risk(risk_level: str) -> str:
    if risk_level == "SAFE":
        return "normal"
    if risk_level == "DANGEROUS":
        return "unnormal"
    return "unknown"


def _korean_risk_text(risk_level: str) -> str:
    if risk_level == "SAFE":
        return "정상"
    if risk_level == "DANGEROUS":
        return "악성"
    return "의심"


def _extract_model_risk(result: Any) -> str:
    if not isinstance(result, dict):
        return "UNKNOWN"
    for key in ("judgment", "verdict", "riskLevel", "risklevel", "risk_level", "label"):
        if key in result:
            risk = _risk_from_model_text(result.get(key))
            if risk != "UNKNOWN":
                return risk
    return "UNKNOWN"


@functools.lru_cache(maxsize=URL_RULE_CACHE_SIZE)
def _url_rule_adjustment(url: str) -> dict[str, Any] | None:
    """Conservative URL-only override shared by all API lanes."""
    for embedded_url in _embedded_redirect_urls(url):
        embedded_score = float(strong_url_phishing_score(embedded_url))
        if embedded_score >= 0.66:
            return {
                "riskLevel": "DANGEROUS",
                "judgment": "unnormal",
                "verdict": "malicious",
                "probability": embedded_score,
                "reason": "redirect 파라미터 내부의 강한 URL 피싱 패턴 감지",
            }
    if is_trusted_official_url(url):
        return {
            "riskLevel": "SAFE",
            "judgment": "normal",
            "verdict": "benign",
            "probability": 0.0,
            "reason": "공식/신뢰 도메인 사전 통과",
        }
    score = float(strong_url_phishing_score(url))
    if score >= 0.66:
        return {
            "riskLevel": "DANGEROUS",
            "judgment": "unnormal",
            "verdict": "malicious",
            "probability": score,
            "reason": "강한 URL 피싱 패턴 사전 감지",
        }
    return None


def _model_probability(model: str, result: Any, risk_level: str | None = None) -> float | None:
    if not isinstance(result, dict):
        return None
    if result.get("probability") is not None and model in {"URLML", "URLHeuristic"}:
        try:
            return max(0.0, min(1.0, float(result.get("probability"))))
        except (TypeError, ValueError):
            return None
    if model == "XGBoost":
        for key in ("final_probability", "probability", "typo_probability", "domain_probability", "dom_probability"):
            if result.get(key) is not None:
                try:
                    return max(0.0, min(1.0, float(result.get(key))))
                except (TypeError, ValueError):
                    continue
    elif model == "GNN":
        if result.get("probability") is not None:
            try:
                return max(0.0, min(1.0, float(result.get("probability"))))
            except (TypeError, ValueError):
                return None
    elif model == "KoBERT":
        if result.get("threat_score") is not None:
            try:
                score = float(result.get("threat_score"))
                return max(0.0, min(1.0, score / 100.0 if score > 1.0 else score))
            except (TypeError, ValueError):
                pass

    risk = risk_level or _extract_model_risk(result)
    if risk == "DANGEROUS":
        return 0.75
    if risk == "SAFE":
        return 0.10
    return None


def _apply_url_rule_adjustment(
    model: str,
    url: str,
    result: Any,
    adjustment: dict[str, Any] | None | object = _NO_ADJUSTMENT,
) -> Any:
    if adjustment is _NO_ADJUSTMENT:
        adjustment = _url_rule_adjustment(url)
    if adjustment is None:
        return result

    out = dict(result) if isinstance(result, dict) else {}
    out.update(
        {
            "riskLevel": adjustment["riskLevel"],
            "risklevel": adjustment["riskLevel"],
            "judgment": adjustment["judgment"],
            "verdict": adjustment["verdict"],
            "adjusted_by_rule": True,
            "adjustment_reason": adjustment["reason"],
        }
    )
    prob = float(adjustment["probability"])
    if model == "XGBoost":
        out["final_probability"] = round(prob, 6)
        out["label"] = 1 if adjustment["riskLevel"] == "DANGEROUS" else 0
        out.setdefault("explanations", [adjustment["reason"]])
    elif model == "GNN":
        out["probability"] = round(prob, 6)
        out["label"] = 1 if adjustment["riskLevel"] == "DANGEROUS" else 0
        out.setdefault("explanation", [adjustment["reason"]])
    elif model == "KoBERT":
        out["threat_score"] = round(prob * 100.0, 1)
        out.setdefault("threat_type", "URL 구조 기반 판정" if prob else "안전(공식/신뢰 도메인)")
        out.setdefault(
            "evidence",
            {
                "heuristic_evidence": {"detected_actions": [], "rule_trigger": adjustment["reason"]},
                "ai_semantic_evidence": {
                    "suspect_sentence": url,
                    "ai_inference_logic": adjustment["reason"],
                },
            },
        )
    return out


def _apply_url_ml_hint_to_model(model: str, result: Any, url_ml_result: Any) -> Any:
    if not isinstance(result, dict) or not isinstance(url_ml_result, dict):
        return result
    url_ml_risk = _extract_model_risk(url_ml_result)
    if url_ml_risk not in {"SAFE", "DANGEROUS"}:
        return result
    model_risk = _extract_model_risk(result)
    if model_risk == url_ml_risk:
        return result

    out = dict(result)
    prob = _model_probability("URLML", url_ml_result, url_ml_risk)
    if prob is None:
        prob = 0.95 if url_ml_risk == "DANGEROUS" else 0.05
    label = 1 if url_ml_risk == "DANGEROUS" else 0
    out.update(
        {
            "riskLevel": url_ml_risk,
            "risklevel": url_ml_risk,
            "judgment": _judgment_from_risk(url_ml_risk),
            "verdict": _verdict_from_risk(url_ml_risk),
            "label": label,
            "adjusted_by_url_ml": True,
            "url_ml_hint_probability": round(float(prob), 6),
        }
    )
    if model == "XGBoost":
        out["final_probability"] = round(float(prob), 6)
        explanations = out.get("explanations")
        reason = f"URL ML 보조 신호로 {model} 단독 판정을 보정했습니다."
        if isinstance(explanations, list):
            out["explanations"] = [reason, *explanations]
        else:
            out["explanations"] = [reason]
    else:
        out["probability"] = round(float(prob), 6)
        reason = f"URL ML 보조 신호로 {model} 단독 판정을 보정했습니다."
        explanations = out.get("explanation")
        if isinstance(explanations, list):
            out["explanation"] = [reason, *explanations]
        else:
            out["explanation"] = [reason]
    return out


@functools.lru_cache(maxsize=URL_RULE_CACHE_SIZE)
def _url_heuristic_result(url: str) -> dict[str, Any]:
    score = float(url_heuristic_phishing_score(url))
    ensemble_probability: float | None = score
    if is_trusted_official_url(url):
        risk_level = "SAFE"
        reason = "공식/신뢰 도메인 URL 휴리스틱 통과"
        ensemble_probability = 0.0
    elif score >= 0.66:
        risk_level = "DANGEROUS"
        reason = "강한 URL 휴리스틱 악성 패턴"
    elif score >= 0.35:
        risk_level = "UNKNOWN"
        reason = "중간 강도 URL 휴리스틱 의심 패턴"
    else:
        risk_level = "SAFE"
        reason = "URL 휴리스틱 특이사항 낮음"
        ensemble_probability = None
    return {
        "model": "URLHeuristic",
        "available": True,
        "riskLevel": risk_level,
        "judgment": _judgment_from_risk(risk_level),
        "verdict": _verdict_from_risk(risk_level),
        "summary": f"URLHeuristic: {_korean_risk_text(risk_level)}",
        "probability": round(ensemble_probability, 6) if ensemble_probability is not None else None,
        "raw_probability": round(score, 6),
        "evidence_reasons": [reason],
    }


def _model_detail(
    model: str,
    result: Any,
    status: dict | None = None,
) -> dict[str, Any]:
    if result is None:
        reason = (status or {}).get("reason") or "model_not_loaded"
        return {
            "model": model,
            "available": False,
            "riskLevel": "UNKNOWN",
            "judgment": "unknown",
            "verdict": "unknown",
            "summary": f"{model} 모델은 실행되지 않았습니다: {reason}",
        }

    if not isinstance(result, dict):
        return {
            "model": model,
            "available": True,
            "riskLevel": "UNKNOWN",
            "judgment": "unknown",
            "verdict": "unknown",
            "summary": f"{model} 모델 결과 형식이 예상과 다릅니다.",
            "raw": result,
        }

    risk_level = _extract_model_risk(result)
    judgment = str(result.get("judgment") or _judgment_from_risk(risk_level))
    verdict = str(result.get("verdict") or _verdict_from_risk(risk_level))
    probability = _model_probability(model, result, risk_level)
    summary = f"{model}: {_korean_risk_text(risk_level)}"
    if result.get("error"):
        summary = f"{model}: 의심"
    elif result.get("engine_disabled"):
        summary = f"{model}: 의심"

    detail = {
        "model": model,
        "available": not (result.get("engine_disabled") or result.get("status") == "unavailable"),
        "riskLevel": risk_level,
        "judgment": judgment,
        "verdict": verdict,
        "summary": summary,
        "probability": round(probability, 6) if probability is not None else None,
        "adjusted_by_rule": bool(result.get("adjusted_by_rule")),
        "adjustment_reason": result.get("adjustment_reason"),
    }

    if model == "KoBERT":
        evidence = result.get("evidence") or {}
        semantic = evidence.get("ai_semantic_evidence") or {}
        evidence_reasons = [
            f"문맥 분석 결과 - {semantic.get('ai_inference_logic')}"
            if semantic.get("ai_inference_logic")
            else None,
        ]
        detail.update(
            {
                "threat_type": result.get("threat_type"),
                "site_category": result.get("site_category"),
                "evidence_reasons": [r for r in evidence_reasons if r],
                "ai_inference_logic": semantic.get("ai_inference_logic"),
            },
        )
    elif model == "XGBoost":
        explanations = _clean_xgboost_explanations(result.get("explanations") or [])
        detail.update(
            {
                "evidence_reasons": explanations,
                "typo_label": result.get("typo_label"),
                "domain_label": result.get("domain_label"),
                "dom_label": result.get("dom_label"),
            },
        )
    elif model == "GNN":
        probability = result.get("probability")
        evidence_reasons = _clean_gnn_explanations(result.get("explanation"))
        detail.update(
            {
                "evidence_reasons": evidence_reasons,
                "model_type": result.get("model_type"),
                "probability": probability,
            },
        )
    elif model == "URLML":
        evidence_reasons = []
        if result.get("adjustment_reason"):
            evidence_reasons.append(str(result.get("adjustment_reason")))
        elif result.get("ml_probability") is not None:
            evidence_reasons.append(
                f"URL 문자열 ML 확률 {float(result.get('ml_probability')):.3f}, "
                f"휴리스틱 확률 {float(result.get('heuristic_probability', 0.0)):.3f}"
            )
        detail.update({"evidence_reasons": evidence_reasons})

    return {k: v for k, v in detail.items() if v not in (None, "", [])}


def _clean_xgboost_explanations(explanations: list[Any]) -> list[str]:
    cleaned = []
    for item in explanations:
        text = str(item).strip()
        if not text or text.startswith("["):
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        if text:
            cleaned.append(text)
    return cleaned


def _clean_gnn_explanations(explanation: Any) -> list[str]:
    if not explanation:
        return []
    # build_explanation 이 List[str] 을 반환하므로 그대로 정리
    if isinstance(explanation, list):
        items = [str(x) for x in explanation]
    else:
        # 과거 호환: 단일 문자열이면 줄 단위 분할
        items = str(explanation).splitlines()

    cleaned = []
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        # 과거 emoji 접두 제거 (안전망)
        for emoji in ("🚨 ", "⚠️ ", "🔍 ", "✅ ", "🟡 ", "🔥 ", "🎯 "):
            if text.startswith(emoji):
                text = text[len(emoji):]
        if text:
            cleaned.append(text)
    return cleaned


def _xgboost_dominant_signal(result: dict[str, Any]) -> str:
    signals = [
        ("타이포스쿼팅", result.get("typo_probability"), result.get("typo_label")),
        ("도메인 평판", result.get("domain_probability"), result.get("domain_label")),
        ("DOM 구조", result.get("dom_probability"), result.get("dom_label")),
    ]
    present = [
        (name, float(prob), label)
        for name, prob, label in signals
        if prob is not None
    ]
    if not present:
        return "사용 가능한 세부 신호 없음"
    name, prob, label = max(present, key=lambda item: item[1])
    return f"{name} 신호가 가장 강함(label={label}, probability={prob:.3f})"


def _decide_final_risk(details: list[dict[str, Any]]) -> str:
    official_override = any(
        detail.get("adjusted_by_rule")
        and detail.get("riskLevel") == "SAFE"
        and "공식/신뢰" in str(detail.get("adjustment_reason") or "")
        for detail in details
    )
    if official_override:
        return "SAFE"

    strong_url_override = any(
        detail.get("adjusted_by_rule")
        and detail.get("riskLevel") == "DANGEROUS"
        and "강한 URL" in str(detail.get("adjustment_reason") or "")
        for detail in details
    )
    if strong_url_override:
        return "DANGEROUS"

    by_model = {str(detail.get("model")): detail for detail in details}
    url_ml = by_model.get("URLML") or {}
    url_heuristic = by_model.get("URLHeuristic") or {}
    kobert = by_model.get("KoBERT") or {}
    if (
        url_ml.get("available")
        and url_ml.get("riskLevel") == "SAFE"
        and url_heuristic.get("available")
        and url_heuristic.get("riskLevel") == "SAFE"
        and kobert.get("riskLevel") != "DANGEROUS"
    ):
        return "SAFE"

    xgboost = by_model.get("XGBoost") or {}
    gnn = by_model.get("GNN") or {}
    if (
        url_ml.get("available")
        and url_ml.get("riskLevel") == "SAFE"
        and url_heuristic.get("available")
        and url_heuristic.get("riskLevel") == "SAFE"
        and xgboost.get("riskLevel") == "SAFE"
        and gnn.get("riskLevel") != "DANGEROUS"
        and kobert.get("riskLevel") == "DANGEROUS"
    ):
        return "UNKNOWN"

    if url_ml.get("available") and url_ml.get("riskLevel") == "DANGEROUS":
        try:
            url_ml_prob = float(url_ml.get("probability") or 0.0)
        except Exception:
            url_ml_prob = 0.0
        strong_support = 0
        for name in ("KoBERT", "XGBoost", "GNN", "URLHeuristic"):
            detail = by_model.get(name) or {}
            if detail.get("riskLevel") != "DANGEROUS":
                continue
            if detail.get("adjusted_by_rule"):
                strong_support += 1
                continue
            try:
                support_prob = float(detail.get("probability") or 0.0)
            except Exception:
                support_prob = 0.0
            if support_prob >= URL_ML_FINAL_SUPPORT_THRESHOLD:
                strong_support += 1
        if url_ml_prob >= URL_ML_FINAL_SOLO_THRESHOLD or strong_support >= 1:
            return "DANGEROUS"

    usable_probs = [
        float(detail["probability"])
        for detail in details
        if detail.get("available") and detail.get("probability") is not None
    ]
    if usable_probs:
        avg_prob = sum(usable_probs) / len(usable_probs)
        danger_threshold = float(os.getenv("FINAL_DANGER_THRESHOLD", "0.40"))
        unknown_threshold = float(os.getenv("FINAL_UNKNOWN_THRESHOLD", "0.30"))
        if avg_prob >= danger_threshold:
            return "DANGEROUS"
        if avg_prob >= unknown_threshold:
            return "UNKNOWN"
        return "SAFE"

    malicious_count = sum(
        1 for detail in details
        if detail.get("available") and detail.get("riskLevel") == "DANGEROUS"
    )
    if malicious_count >= 2:
        return "DANGEROUS"
    if malicious_count == 1:
        return "UNKNOWN"
    return "SAFE"


def _detail_reason_lines(details: list[dict[str, Any]]) -> list[str]:
    lines = []
    for detail in details:
        evidence_reasons = detail.get("evidence_reasons") or []
        for reason in evidence_reasons:
            lines.append(f"{detail['model']} : {reason}")
    return lines


def _build_final_response(
    target_url: str,
    kobert_result: dict,
    xg_result: object,
    gnn_result: object,
    dur_wall: float,
    t_kobert: float,
    t_xg: float,
    t_gnn: float,
    url_ml_result: object = None,
    t_url_ml: float = 0.0,
) -> dict:
    rule_url = _url_cache_key(target_url)
    url_adjustment = _url_rule_adjustment(rule_url)
    kobert_result = _apply_url_rule_adjustment("KoBERT", target_url, kobert_result, url_adjustment)
    xg_result = _apply_url_rule_adjustment("XGBoost", target_url, xg_result, url_adjustment)
    gnn_result = _apply_url_rule_adjustment("GNN", target_url, gnn_result, url_adjustment)
    url_ml_result = _apply_url_rule_adjustment("URLML", target_url, url_ml_result, url_adjustment)
    eng_status = getattr(app.state, "eng_status", {"enabled": False})
    xg_status = getattr(app.state, "xg_status", {"enabled": False})
    gnn_status = getattr(app.state, "gnn_status", {"enabled": False})
    url_ml_status = getattr(app.state, "url_ml_status", {"enabled": False})
    details = [
        _model_detail("KoBERT", kobert_result, eng_status),
        _model_detail("XGBoost", xg_result, xg_status),
        _model_detail("GNN", gnn_result, gnn_status),
        _model_detail("URLML", url_ml_result, url_ml_status),
        _url_heuristic_result(rule_url),
    ]
    risk_level = _decide_final_risk(details)
    judgment = _judgment_from_risk(risk_level)
    malicious_count = sum(
        1 for detail in details
        if detail.get("available") and detail.get("riskLevel") == "DANGEROUS"
    )
    reasons = [
        f"최종 판단 - {_korean_risk_text(risk_level)} (악성 판정 모델 {malicious_count}개)",
        *_detail_reason_lines(details),
    ]
    return {
        "url": target_url,
        "judgment": judgment,
        "riskLevel": risk_level,
        "conclusion": _korean_risk_text(risk_level),
        "decision_method": "score_weighted_ensemble",
        "reasons": reasons,
        "model_details": details,
        "koBERT": kobert_result,
        "xgboost": xg_result,
        "gnn": gnn_result,
        "url_ml": url_ml_result,
        "engine_status": eng_status,
        "xgboost_status": xg_status,
        "gnn_status": gnn_status,
        "url_ml_status": url_ml_status,
        "duration_sec": round(dur_wall, 3),
        "timing": {
            "koBERT_sec": round(t_kobert, 6),
            "xgboost_sec": round(t_xg, 6),
            "gnn_sec": round(t_gnn, 6),
            "url_ml_sec": round(t_url_ml, 6),
            "total_wall_sec": round(dur_wall, 6),
        },
    }


def _skipped_after_urlml_result(model: str) -> dict[str, Any]:
    return {
        "judgment": "unknown",
        "riskLevel": "UNKNOWN",
        "risklevel": "UNKNOWN",
        "verdict": "unknown",
        "engine_disabled": True,
        "status": "unavailable",
        "engine_reason": f"skipped_after_decisive_url_ml:{model}",
    }


def _should_fast_path_after_urlml(result: Any) -> bool:
    if not URL_ML_FAST_PATH or not isinstance(result, dict):
        return False
    risk = _extract_model_risk(result)
    return risk in {"DANGEROUS", "SAFE"}


async def _url_ml_hint_for_request(target_url: str) -> Any:
    url_ml_hint = await _run_url_ml(_run_url_ml_inference, target_url)
    return _apply_url_rule_adjustment(
        "URLML",
        target_url,
        url_ml_hint,
        _url_rule_adjustment(_url_cache_key(target_url)),
    )


def _url_ml_preflight_model_result(model: str, url_ml_hint: Any) -> Any:
    result = _skipped_after_urlml_result(model)
    result["preflight_skipped"] = True
    return _apply_url_ml_hint_to_model(model, result, url_ml_hint)


class URLRequest(BaseModel):
    url: str


class URLBatchRequest(BaseModel):
    urls: list[str]


class TextAnalyzeRequest(BaseModel):
    text: str


def _decode_ascii_url_escapes(text: str) -> str:
    def from_hex(match: re.Match[str]) -> str:
        return chr(int(match.group(1), 16))

    def from_octal(match: re.Match[str]) -> str:
        codepoint = int(match.group(1), 8)
        if codepoint < 32 or codepoint > 126:
            return " "
        return chr(codepoint)

    text = QUOTED_PRINTABLE_SOFT_BREAK_RE.sub("", text)
    decoded = ASCII_HEX_ESCAPE_RE.sub(from_hex, text)
    decoded = ASCII_UNICODE_ESCAPE_RE.sub(from_hex, decoded)
    decoded = ASCII_CODEPOINT_ESCAPE_RE.sub(from_hex, decoded)
    decoded = ASCII_JS_OCTAL_ESCAPE_RE.sub(from_octal, decoded)
    decoded = ASCII_CSS_ZERO_PADDED_ESCAPE_RE.sub(from_hex, decoded)
    decoded = ASCII_CSS_HEX_ESCAPE_RE.sub(from_hex, decoded)
    decoded = ASCII_QUOTED_PRINTABLE_RE.sub(from_hex, decoded)
    decoded = REMAINING_ESCAPE_TOKEN_RE.sub(" ", decoded)
    return REMAINING_QUOTED_PRINTABLE_RE.sub(" ", decoded)


def _decode_repeated_percent_url_text(text: str, *, max_rounds: int = 3) -> str:
    decoded = text
    for _ in range(max(1, max_rounds)):
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    return decoded


def _compact_spaced_url_segments(text: str) -> str:
    output: list[str] = []
    cursor = 0
    for match in SPACED_URL_PREFIX_RE.finditer(text):
        start = match.start()
        if start < cursor:
            continue
        output.append(text[cursor:start])
        segment = [match.group(0)]
        index = match.end()
        slash_from_spaced_separator = False
        while index < len(text):
            ch = text[index]
            if ch in SPACED_URL_CHARS:
                segment.append(ch)
                if ch != "/":
                    slash_from_spaced_separator = False
                index += 1
                continue
            if not ch.isspace():
                break

            next_index = index
            while next_index < len(text) and text[next_index].isspace():
                next_index += 1
            if next_index >= len(text) or text[next_index] not in SPACED_URL_CHARS:
                break

            token_end = next_index
            while token_end < len(text) and text[token_end] in SPACED_URL_CHARS and text[token_end] not in SPACED_URL_BOUNDARY_CHARS:
                token_end += 1
            token_len = token_end - next_index
            prev = segment[-1] if segment else ""
            next_ch = text[next_index]
            if prev == "/" and token_len > 1:
                lookahead = token_end
                while lookahead < len(text) and text[lookahead].isspace():
                    lookahead += 1
                if not slash_from_spaced_separator and (lookahead >= len(text) or text[lookahead] not in SPACED_URL_BOUNDARY_CHARS):
                    break
            if prev in SPACED_URL_BOUNDARY_CHARS or next_ch in SPACED_URL_BOUNDARY_CHARS or token_len == 1:
                if next_ch == "/":
                    slash_from_spaced_separator = True
                index = next_index
                continue
            break

        compacted = "".join(segment)
        output.append(compacted if "." in compacted else "".join(segment))
        cursor = index
    output.append(text[cursor:])
    return "".join(output)


def _normalize_text_for_url_extraction(text: str) -> str:
    normalized_text = html.unescape(ZERO_WIDTH_TEXT_RE.sub("", text))
    normalized_text = _decode_repeated_percent_url_text(normalized_text)
    normalized_text = _decode_ascii_url_escapes(normalized_text)
    normalized_text = unicodedata.normalize("NFKC", normalized_text).translate(TEXT_URL_TRANSLATION)
    normalized_text = STRING_CONCAT_BREAK_RE.sub("", normalized_text)
    normalized_text = BACKSLASH_LINE_CONTINUATION_RE.sub("", normalized_text)
    normalized_text = ESCAPED_URL_PUNCT_RE.sub(r"\1", normalized_text)
    normalized_text = STRING_CONCAT_BREAK_RE.sub("", normalized_text)
    normalized_text = STRING_LITERAL_ADJACENT_RE.sub("", normalized_text)
    normalized_text = DEFANGED_DOT_RE.sub(".", normalized_text)
    normalized_text = DEFANGED_COLON_RE.sub(":", normalized_text)
    normalized_text = DEFANGED_SLASH_RE.sub("/", normalized_text)
    normalized_text = DEFANGED_AT_RE.sub("@", normalized_text)
    normalized_text = KOREAN_SPACED_DEFANGED_DOT_RE.sub(".", normalized_text)
    normalized_text = KOREAN_DEFANGED_DOT_RE.sub(".", normalized_text)
    normalized_text = HOST_SPACED_DOT_RE.sub(".", normalized_text)
    normalized_text = SPACED_HTTPS_SCHEME_RE.sub("https://", normalized_text)
    normalized_text = SPACED_HTTP_SCHEME_RE.sub("http://", normalized_text)
    normalized_text = SPACED_HXXPS_SCHEME_RE.sub("https://", normalized_text)
    normalized_text = SPACED_HXXP_SCHEME_RE.sub("http://", normalized_text)
    normalized_text = SPACED_WWW_RE.sub("www", normalized_text)
    normalized_text = re.sub(r"\s*:\s*/\s*/\s*", "://", normalized_text)
    normalized_text = re.sub(r"(?i)\bhxxps://", "https://", normalized_text)
    normalized_text = re.sub(r"(?i)\bhxxp://", "http://", normalized_text)
    normalized_text = WRAPPED_WWW_AFTER_SCHEME_RE.sub(r"\1www", normalized_text)
    normalized_text = _compact_spaced_url_segments(normalized_text)
    return normalized_text.replace(":// ", "://")


def _strip_url_trailing_punctuation(candidate: str) -> str:
    candidate = (candidate or "").strip().rstrip(".,;:!?\"'")
    while candidate and candidate[-1] in ")]}":
        closer = candidate[-1]
        opener = {"}": "{", "]": "[", ")": "("}[closer]
        if candidate.count(closer) > candidate.count(opener):
            candidate = candidate[:-1]
            continue
        break
    return candidate


def _embedded_redirect_urls(candidate: str) -> list[str]:
    try:
        parsed = urlsplit(candidate if "://" in candidate else f"https://{candidate}")
    except Exception:
        return []
    param_sources = [parsed.query or ""]
    fragment = parsed.fragment or ""
    if fragment:
        param_sources.append(fragment[1:] if fragment.startswith("?") else fragment)
        if "?" in fragment:
            param_sources.append(fragment.split("?", 1)[1])
    embedded: list[str] = []
    for param_source in param_sources:
        if not param_source:
            continue
        for name, value in parse_qsl(param_source, keep_blank_values=False):
            key = name.strip().lower().replace("-", "_")
            if key not in REDIRECT_QUERY_PARAM_NAMES:
                continue
            decoded_value = _decode_repeated_percent_url_text(value)
            normalized_value = _normalize_text_for_url_extraction(decoded_value)
            values_to_scan = [normalized_value]
            for base64_candidate in {value, decoded_value}:
                base64_decoded_value = _decode_base64_url_text(base64_candidate)
                if base64_decoded_value:
                    values_to_scan.append(_normalize_text_for_url_extraction(base64_decoded_value))
                hex_decoded_value = _decode_hex_url_text(base64_candidate)
                if hex_decoded_value:
                    values_to_scan.append(_normalize_text_for_url_extraction(hex_decoded_value))
            for value_to_scan in values_to_scan:
                match = URL_IN_TEXT_RE.search(value_to_scan)
                if not match:
                    continue
                redirect_url = _strip_url_trailing_punctuation(match.group(1))
                if redirect_url and "." in redirect_url:
                    embedded.append(redirect_url)
                    break
    return embedded


def _append_urls_from_normalized_text(normalized_text: str, urls: list[str]) -> None:
    seen_spans: set[tuple[int, int]] = set()
    for match in URL_IN_TEXT_RE.finditer(normalized_text):
        span = match.span(1)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        candidate = _strip_url_trailing_punctuation(match.group(1))
        if JS_FUNCTION_URL_FALSE_POSITIVE_RE.fullmatch(candidate):
            continue
        if not candidate or "." not in candidate:
            continue
        urls.append(candidate)
        for embedded_url in _embedded_redirect_urls(candidate):
            urls.append(embedded_url)
            if len(urls) > ANALYZE_BATCH_MAX_URLS:
                raise HTTPException(
                    status_code=413,
                    detail=f"Too many URLs in text: {len(urls)} > {ANALYZE_BATCH_MAX_URLS}",
                )
        if len(urls) > ANALYZE_BATCH_MAX_URLS:
            raise HTTPException(
                status_code=413,
                detail=f"Too many URLs in text: {len(urls)} > {ANALYZE_BATCH_MAX_URLS}",
            )


def _decode_base64_url_text(token: str) -> str:
    if len(token) > 4096:
        return ""
    padded = token + ("=" * ((4 - len(token) % 4) % 4))
    try:
        decoded = base64.b64decode(padded, altchars=b"-_", validate=False)
    except (binascii.Error, ValueError):
        return ""
    try:
        decoded_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    normalized_text = _normalize_text_for_url_extraction(decoded_text)
    if not URL_IN_TEXT_RE.search(normalized_text):
        return ""
    return decoded_text


def _decode_hex_url_text(token: str) -> str:
    if len(token) > 4096 or len(token) % 2:
        return ""
    try:
        decoded = bytes.fromhex(token)
    except ValueError:
        return ""
    try:
        decoded_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    if any((ord(char) < 32 and char not in "\t\r\n") for char in decoded_text):
        return ""
    normalized_text = _normalize_text_for_url_extraction(decoded_text)
    if not URL_IN_TEXT_RE.search(normalized_text):
        return ""
    return decoded_text


def _decode_js_charcode_url_text(args_text: str) -> str:
    if len(args_text) > 4096:
        return ""
    values = re.findall(r"(?i)(?:0x[0-9a-f]{2,4}|\d{2,5})", args_text)
    if len(values) < 8:
        return ""
    chars: list[str] = []
    for value in values[:512]:
        try:
            codepoint = int(value, 16) if value.lower().startswith("0x") else int(value, 10)
        except ValueError:
            return ""
        if codepoint in (9, 10, 13):
            chars.append(" ")
            continue
        if codepoint < 32 or codepoint > 126:
            return ""
        chars.append(chr(codepoint))
    decoded_text = "".join(chars)
    normalized_text = _normalize_text_for_url_extraction(decoded_text)
    if not URL_IN_TEXT_RE.search(normalized_text):
        return ""
    return decoded_text


def _decode_js_empty_join_url_text(array_text: str) -> str:
    parts = [match.group(2) for match in JS_SIMPLE_STRING_LITERAL_RE.finditer(array_text)]
    if len(parts) < 2:
        return ""
    decoded_text = _decode_ascii_url_escapes("".join(parts))
    normalized_text = _normalize_text_for_url_extraction(decoded_text)
    if not URL_IN_TEXT_RE.search(normalized_text):
        return ""
    return decoded_text


def _decode_js_reversed_empty_join_url_text(array_text: str) -> str:
    parts = [match.group(2) for match in JS_SIMPLE_STRING_LITERAL_RE.finditer(array_text)]
    if len(parts) < 2:
        return ""
    joined_text = _decode_ascii_url_escapes("".join(parts))
    return _decode_js_reversed_url_text(joined_text)


def _has_js_reverse_suffix(text: str, offset: int) -> bool:
    return bool(JS_REVERSE_SUFFIX_RE.match(text[offset : offset + 80]))


def _decode_js_reversed_url_text(reversed_text: str) -> str:
    if len(reversed_text) > 4096:
        return ""
    decoded_text = _decode_ascii_url_escapes(reversed_text[::-1])
    normalized_text = _normalize_text_for_url_extraction(decoded_text)
    if not URL_IN_TEXT_RE.search(normalized_text):
        return ""
    return decoded_text


def _extract_urls_from_text(text: str) -> list[str]:
    if not text:
        return []
    if len(text) > ANALYZE_TEXT_MAX_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"Text too long: {len(text)} > {ANALYZE_TEXT_MAX_CHARS}",
        )

    urls: list[str] = []
    normalized_text = _normalize_text_for_url_extraction(text)
    direct_scan_text = JS_REVERSED_STRING_RE.sub(" ", normalized_text)
    direct_scan_text = JS_REVERSED_EMPTY_JOIN_ARRAY_RE.sub(" ", direct_scan_text)
    _append_urls_from_normalized_text(direct_scan_text, urls)
    for token_match in BASE64_TEXT_TOKEN_RE.finditer(normalized_text):
        decoded_text = _decode_base64_url_text(token_match.group(1))
        if decoded_text:
            _append_urls_from_normalized_text(_normalize_text_for_url_extraction(decoded_text), urls)
    for token_match in HEX_TEXT_TOKEN_RE.finditer(normalized_text):
        decoded_text = _decode_hex_url_text(token_match.group(1))
        if decoded_text:
            _append_urls_from_normalized_text(_normalize_text_for_url_extraction(decoded_text), urls)
    for charcode_match in JS_FROM_CHAR_CODE_RE.finditer(normalized_text):
        decoded_text = _decode_js_charcode_url_text(charcode_match.group(1))
        if decoded_text:
            _append_urls_from_normalized_text(_normalize_text_for_url_extraction(decoded_text), urls)
    for join_match in JS_EMPTY_JOIN_ARRAY_RE.finditer(normalized_text):
        if _has_js_reverse_suffix(normalized_text, join_match.end()):
            continue
        decoded_text = _decode_js_empty_join_url_text(join_match.group(1))
        if decoded_text:
            _append_urls_from_normalized_text(_normalize_text_for_url_extraction(decoded_text), urls)
    for reverse_match in JS_REVERSED_STRING_RE.finditer(normalized_text):
        decoded_text = _decode_js_reversed_url_text(reverse_match.group(2))
        if decoded_text:
            _append_urls_from_normalized_text(_normalize_text_for_url_extraction(decoded_text), urls)
    for reverse_join_match in JS_REVERSED_EMPTY_JOIN_ARRAY_RE.finditer(normalized_text):
        decoded_text = _decode_js_reversed_empty_join_url_text(reverse_join_match.group(1))
        if decoded_text:
            _append_urls_from_normalized_text(_normalize_text_for_url_extraction(decoded_text), urls)
    return urls


def _prepare_batch_urls(raw_urls: list[str]) -> tuple[list[str], list[str], dict[str, str], dict[str, int]]:
    if not raw_urls:
        raise HTTPException(status_code=400, detail="URLs are empty.")
    if len(raw_urls) > ANALYZE_BATCH_MAX_URLS:
        raise HTTPException(
            status_code=413,
            detail=f"Too many URLs: {len(raw_urls)} > {ANALYZE_BATCH_MAX_URLS}",
        )

    normalized_urls: list[str] = []
    keys: list[str] = []
    unique_urls: dict[str, str] = {}
    first_index_by_key: dict[str, int] = {}
    for index, raw_url in enumerate(raw_urls):
        url = (raw_url or "").strip()
        if not url:
            raise HTTPException(status_code=400, detail=f"URL at index {index} is empty.")
        key = _url_cache_key(url, default_scheme="https")
        normalized_urls.append(url)
        keys.append(key)
        if key not in unique_urls:
            unique_urls[key] = url
            first_index_by_key[key] = index
    return normalized_urls, keys, unique_urls, first_index_by_key


@app.on_event("startup")
async def startup_event():
    print("--- [1/2] Loading models (import only) ---")
    # Same contract as koBERT.warmup_engine / predict_phishing_result
    app.state.eng = None
    app.state.eng_status = {"enabled": False, "reason": "not_loaded"}
    try:
        import koBERT as eng
        app.state.eng = eng
        app.state.eng_status = {"enabled": True}
    except Exception as e:
        app.state.eng_status = {"enabled": False, "reason": str(e)}
        print(f"[warn] KoBERT engine load failed: {e}")
    _predict_kobert_cached.cache_clear()

    app.state.warmup_done = False
    app.state.warmup_info = None
    app.state.xg_typo_bundle = None
    app.state.xg_domain_bundle = None
    app.state.xg_dom_bundle = None
    app.state.xg_status = {"enabled": False, "reason": "not_loaded"}
    app.state.gnn_model = None
    app.state.gnn_columns = None
    app.state.gnn_status = {"enabled": False, "reason": "not_loaded"}
    app.state.gnn_engine = None
    app.state.url_ml_model = None
    app.state.url_ml_status = {"enabled": False, "reason": "not_loaded"}
    _clear_runtime_caches()

    try:
        url_ml_model, url_ml_status = load_url_ml_model()
        app.state.url_ml_model = url_ml_model
        app.state.url_ml_status = url_ml_status.__dict__
        _predict_url_ml_cached.cache_clear()
        _url_rule_adjustment.cache_clear()
        _url_heuristic_result.cache_clear()
        if url_ml_model is not None:
            sm = predict_url_ml(url_ml_model, "https://example.com")
            print(f"  [URLML] smoke OK — verdict={sm.get('verdict')} p={sm.get('probability')}")
        else:
            print(f"  [URLML] skipped — {url_ml_status.reason}")
    except Exception as e:
        app.state.url_ml_status = {"enabled": False, "reason": str(e)}

    # Optional: web-structure GNN (gnn_engine: gnn_model.pkl + gnn_model_features.pkl)
    try:
        eng_gnn = GNN_Engine()
        app.state.gnn_engine = eng_gnn
        if eng_gnn.ok:
            app.state.gnn_model = eng_gnn.model
            app.state.gnn_columns = eng_gnn.columns
            app.state.gnn_status = {
                "enabled": True,
                "model_path": eng_gnn.model_path,
                "feature_columns_path": eng_gnn.feature_columns_path
                if os.path.isfile(eng_gnn.feature_columns_path)
                else None,
            }
        else:
            reason = eng_gnn._load_error or "model_not_loaded"
            if reason.startswith("missing_model:"):
                app.state.gnn_status = {"enabled": False, "reason": reason}
            else:
                app.state.gnn_status = {"enabled": False, "reason": reason}
    except Exception as e:
        app.state.gnn_status = {"enabled": False, "reason": str(e)}
    _predict_gnn_cached.cache_clear()

    # Optional smoke predict; disabled by default because it performs network fetches.
    if (
        STARTUP_SMOKE_GNN
        and app.state.gnn_model is not None
        and app.state.gnn_columns is not None
    ):
        try:
            sm = predict_gnn(
                app.state.gnn_model,
                app.state.gnn_columns,
                "https://example.com",
            )
            print(
                f"  [GNN] smoke OK — verdict={sm.get('verdict')} "
                f"p={sm.get('probability')}"
            )
        except Exception as e:
            print(f"  [GNN] smoke failed (requests may fail too): {e}")

    # Optional XGBoost bundles (typo, domain-age, DOM — same as XG_infer.py)
    xg_typo_path = os.getenv(
        "XG_MODEL_TYPO",
        os.path.join(MODEL_DIRS["xgboost"], "url_xgb_paired_first.joblib"),
    )
    xg_domain_path = os.getenv(
        "XG_MODEL_DOMAIN",
        os.path.join(MODEL_DIRS["xgboost"], "url_xgb_domain_age.joblib"),
    )
    xg_dom_path = os.getenv(
        "XG_MODEL_DOM",
        os.path.join(MODEL_DIRS["xgboost"], "url_xgb_dom.joblib"),
    )
    xg_errors = []
    try:
        if os.path.isfile(xg_typo_path):
            app.state.xg_typo_bundle = load_bundle(xg_typo_path)
        else:
            xg_errors.append(f"missing_typo_model:{xg_typo_path}")
    except Exception as e:
        xg_errors.append(f"typo_load_error:{e}")

    try:
        if os.path.isfile(xg_domain_path):
            app.state.xg_domain_bundle = load_bundle(xg_domain_path)
        else:
            xg_errors.append(f"missing_domain_model:{xg_domain_path}")
    except Exception as e:
        xg_errors.append(f"domain_load_error:{e}")

    try:
        if os.path.isfile(xg_dom_path):
            app.state.xg_dom_bundle = load_bundle(xg_dom_path)
        else:
            xg_errors.append(f"missing_dom_model:{xg_dom_path}")
    except Exception as e:
        xg_errors.append(f"dom_load_error:{e}")

    if (
        app.state.xg_typo_bundle is not None
        or app.state.xg_domain_bundle is not None
        or app.state.xg_dom_bundle is not None
    ):
        app.state.xg_status = {
            "enabled": True,
            "typo_loaded": app.state.xg_typo_bundle is not None,
            "domain_loaded": app.state.xg_domain_bundle is not None,
            "dom_loaded": app.state.xg_dom_bundle is not None,
            "warnings": xg_errors,
        }
    else:
        app.state.xg_status = {
            "enabled": False,
            "warnings": xg_errors,
        }
    _predict_xgboost_cached.cache_clear()

    print("--- [2/2] Warmup (warmup_engine) ---")
    if app.state.eng is not None:
        try:
            info = await _run_engine(app.state.eng.warmup_engine, STARTUP_WARMUP_PLAYWRIGHT)
            app.state.warmup_info = info
            app.state.warmup_done = True
        except Exception as e:
            print(f"[warmup failed] {e}")
            app.state.warmup_done = False
            # Server still starts; /analyze may retry Playwright as needed.
            app.state.warmup_info = {"error": str(e)}
    else:
        app.state.warmup_done = False
        app.state.warmup_info = {"skipped": "engine_not_loaded"}

    print("--- [system] Ready. /analyze runs full checks. ---")
    eng = getattr(app.state, "eng", None)
    if eng is None:
        r = getattr(app.state, "eng_status", {}).get("reason", "")
        print(
            "  [hint] KoBERT engine did not load (import failed). "
            "/ready will show ready=false until fixed. "
            "Install deps with the SAME Python that runs uvicorn, e.g.: "
            "python -m pip install -r requirements.txt"
        )
        if r:
            print(f"  [hint] Import error was: {r}")


@app.get("/health")
async def health():
    """Liveness: process is up (use for K8s livenessProbe). Does not check ML deps."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: KoBERT warmup finished. False if import failed or warmup errored."""
    eng_ok = getattr(app.state, "eng", None) is not None
    warmup_done = bool(getattr(app.state, "warmup_done", False))
    eng_status = getattr(app.state, "eng_status", {"enabled": False})
    issues = []
    if not eng_ok:
        issues.append(
            f"koBERT import failed: {eng_status.get('reason', 'unknown')}. "
            "Use the same interpreter for pip and uvicorn (python -m pip install -r requirements.txt)."
        )
    elif not warmup_done:
        wi = getattr(app.state, "warmup_info", None)
        issues.append(f"warmup incomplete: {wi}")

    return {
        "http_ok": True,
        "ready": warmup_done and eng_ok,
        "kobert_import_ok": eng_ok,
        "warmup_done": warmup_done,
        "warmup": getattr(app.state, "warmup_info", None),
        "engine": eng_status,
        "xgboost": getattr(app.state, "xg_status", {"enabled": False}),
        "gnn": getattr(app.state, "gnn_status", {"enabled": False}),
        "url_ml": getattr(app.state, "url_ml_status", {"enabled": False}),
        "runtime": {
            "xgboost_workers": XGBOOST_WORKERS,
            "gnn_workers": GNN_WORKERS,
            "kobert_cache_size": KOBERT_CACHE_SIZE,
            "xgboost_cache_size": XGBOOST_CACHE_SIZE,
            "gnn_cache_size": GNN_CACHE_SIZE,
            "url_ml_cache_size": URL_ML_CACHE_SIZE,
            "analyze_batch_max_urls": ANALYZE_BATCH_MAX_URLS,
            "analyze_batch_workers": ANALYZE_BATCH_WORKERS,
            "analyze_text_max_chars": ANALYZE_TEXT_MAX_CHARS,
            "url_ml_batch_vector_min": URL_ML_BATCH_VECTOR_MIN,
            "kobert_url_ml_preflight": KOBERT_URL_ML_PREFLIGHT,
            "gnn": gnn_runtime_config(),
            "xgboost": xg_runtime_config(),
        },
        "cache": _runtime_cache_stats(),
        "inflight": _runtime_inflight_stats(),
        "issues": issues,
    }


@app.post("/warmup")
async def warmup_manual():
    """Manual warmup if Playwright was skipped at boot."""
    eng = getattr(app.state, "eng", None)
    if eng is None:
        raise HTTPException(
            status_code=503,
            detail=f"Engine not loaded: {getattr(app.state, 'eng_status', {}).get('reason', 'unknown')}",
        )
    include_pw = os.getenv("WARMUP_PLAYWRIGHT", "1") == "1"
    t0 = time.perf_counter()
    try:
        info = await _run_engine(eng.warmup_engine, include_pw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    app.state.warmup_info = info
    app.state.warmup_done = True
    dur = time.perf_counter() - t0
    return {"ok": True, "warmup": info, "duration_sec": round(dur, 3)}


async def _url_ml_results_for_unique_items(unique_items: list[tuple[str, str]]) -> list[Any]:
    model = getattr(app.state, "url_ml_model", None)
    if model is not None and len(unique_items) >= URL_ML_BATCH_VECTOR_MIN:
        return await _run_url_ml(
            predict_url_ml_batch,
            model,
            [url for _key, url in unique_items],
        )
    return [await _run_url_ml(_run_url_ml_inference, url) for _key, url in unique_items]


async def _analyze_target_url(
    target_url: str,
    url_ml_result: Any = None,
    t_url_ml: float | None = None,
) -> dict[str, Any]:
    """URLML fast path, then parallel koBERT, XGBoost, GNN when needed."""
    if ANALYZE_VERBOSE_LOGS:
        print(f"--- [analyze] URL: {target_url} ---")

    t_wall0 = time.perf_counter()
    try:
        if url_ml_result is None:
            t_url_ml0 = time.perf_counter()
            url_ml_result = await _run_url_ml(_run_url_ml_inference, target_url)
            t_url_ml = time.perf_counter() - t_url_ml0
        elif t_url_ml is None:
            t_url_ml = 0.0
        if _should_fast_path_after_urlml(url_ml_result):
            dur_wall = time.perf_counter() - t_wall0
            if ANALYZE_VERBOSE_LOGS:
                print(
                    f"--- [analyze fast-path] URL: {target_url}\n"
                    f"    {_log_line_url_ml(url_ml_result)}  ({t_url_ml:.3f}s)\n"
                    f"    skipped koBERT/XGBoost/GNN after decisive URLML\n"
                    f"    wall time: {dur_wall:.3f}s"
                )
            return _build_final_response(
                target_url=target_url,
                kobert_result=_skipped_after_urlml_result("KoBERT"),
                xg_result=_skipped_after_urlml_result("XGBoost"),
                gnn_result=_skipped_after_urlml_result("GNN"),
                dur_wall=dur_wall,
                t_kobert=0.0,
                t_xg=0.0,
                t_gnn=0.0,
                url_ml_result=url_ml_result,
                t_url_ml=t_url_ml,
            )

        # Three branches in parallel (not sequential)
        async def _kobert_timed():
            t0 = time.perf_counter()
            eng = getattr(app.state, "eng", None)
            if eng is not None:
                r = await _run_kobert_request(target_url)
            else:
                r = {
                    "judgment": "unknown",
                    "riskLevel": "UNKNOWN",
                    "risklevel": "UNKNOWN",
                    "engine_disabled": True,
                    "engine_reason": getattr(app.state, "eng_status", {}).get("reason", "not_loaded"),
                }
            return r, time.perf_counter() - t0

        async def _xg_timed():
            t0 = time.perf_counter()
            r = await _run_xgboost_request(target_url)
            return r, time.perf_counter() - t0

        async def _gnn_timed():
            t0 = time.perf_counter()
            r = await _run_gnn_request(target_url)
            return r, time.perf_counter() - t0

        (
            (kobert_result, t_kobert),
            (xg_result, t_xg),
            (gnn_result, t_gnn),
        ) = await asyncio.gather(
            _kobert_timed(),
            _xg_timed(),
            _gnn_timed(),
        )
    except Exception as e:
        print(f"[error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

    dur_wall = time.perf_counter() - t_wall0
    if ANALYZE_VERBOSE_LOGS:
        print(
            f"--- [analyze summary] URL: {target_url}  (koBERT | xgboost | gnn parallel)\n"
            f"    {_log_line_kobert(kobert_result)}  ({t_kobert:.3f}s)\n"
            f"    {_log_line_xgboost(xg_result)}  ({t_xg:.3f}s)\n"
            f"    {_log_line_gnn(gnn_result)}  ({t_gnn:.3f}s)\n"
            f"    {_log_line_url_ml(url_ml_result)}  ({t_url_ml:.3f}s)\n"
            f"    wall time (parallel): {dur_wall:.3f}s"
        )
    return _build_final_response(
        target_url=target_url,
        kobert_result=kobert_result,
        xg_result=xg_result,
        gnn_result=gnn_result,
        dur_wall=dur_wall,
        t_kobert=t_kobert,
        t_xg=t_xg,
        t_gnn=t_gnn,
        url_ml_result=url_ml_result,
        t_url_ml=t_url_ml,
    )


@app.post("/analyze")
async def analyze_url(request: URLRequest):
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    return await _analyze_target_url(target_url)


@app.post("/analyze/batch")
async def analyze_url_batch(request: URLBatchRequest):
    raw_urls = list(request.urls or [])
    normalized_urls, keys, unique_urls, first_index_by_key = _prepare_batch_urls(raw_urls)

    semaphore = asyncio.Semaphore(ANALYZE_BATCH_WORKERS)
    results_by_key: dict[str, dict[str, Any]] = {}
    unique_items = list(unique_urls.items())

    t0 = time.perf_counter()
    t_url_ml0 = time.perf_counter()
    url_ml_results = await _url_ml_results_for_unique_items(unique_items)
    batch_t_url_ml = time.perf_counter() - t_url_ml0
    url_ml_by_key: dict[str, Any] = {}
    for (key, url), url_ml_result in zip(unique_items, url_ml_results):
        url_ml_by_key[key] = _apply_url_rule_adjustment(
            "URLML",
            url,
            url_ml_result,
            _url_rule_adjustment(key),
        )

    async def analyze_one(key: str, url: str) -> None:
        async with semaphore:
            per_url_t_url_ml = batch_t_url_ml / max(1, len(unique_items))
            results_by_key[key] = await _analyze_target_url(url, url_ml_by_key.get(key), per_url_t_url_ml)

    await asyncio.gather(*(analyze_one(key, url) for key, url in unique_items))
    results = [
        {
            "index": index,
            "url": normalized_urls[index],
            "duplicate_of": first_index_by_key[key] if first_index_by_key[key] != index else None,
            "result": results_by_key[key],
        }
        for index, key in enumerate(keys)
    ]
    return {
        "count": len(raw_urls),
        "unique_count": len(unique_urls),
        "deduplicated": len(unique_urls) < len(raw_urls),
        "duration_sec": round(time.perf_counter() - t0, 3),
        "results": results,
    }


@app.post("/analyze/text")
async def analyze_text(request: TextAnalyzeRequest):
    urls = _extract_urls_from_text(request.text or "")
    if not urls:
        return {
            "count": 0,
            "unique_count": 0,
            "deduplicated": False,
            "duration_sec": 0.0,
            "results": [],
        }
    return await analyze_url_batch(URLBatchRequest(urls=urls))


@app.post("/analyze/url-ml/batch")
async def analyze_url_ml_batch(request: URLBatchRequest):
    raw_urls = list(request.urls or [])
    normalized_urls, keys, unique_urls, first_index_by_key = _prepare_batch_urls(raw_urls)

    t0 = time.perf_counter()
    results_by_key: dict[str, dict[str, Any]] = {}
    unique_items = list(unique_urls.items())
    batch_results = await _url_ml_results_for_unique_items(unique_items)

    for (key, url), url_ml_result in zip(unique_items, batch_results):
        url_ml_result = _apply_url_rule_adjustment(
            "URLML",
            url,
            url_ml_result,
            _url_rule_adjustment(key),
        )
        results_by_key[key] = {
            "url_ml": url_ml_result,
            "url_ml_status": getattr(app.state, "url_ml_status", {"enabled": False}),
        }

    results = [
        {
            "index": index,
            "url": normalized_urls[index],
            "duplicate_of": first_index_by_key[key] if first_index_by_key[key] != index else None,
            "result": results_by_key[key],
        }
        for index, key in enumerate(keys)
    ]
    return {
        "count": len(raw_urls),
        "unique_count": len(unique_urls),
        "deduplicated": len(unique_urls) < len(raw_urls),
        "duration_sec": round(time.perf_counter() - t0, 3),
        "results": results,
    }


@app.post("/analyze/url-ml/text")
async def analyze_url_ml_text(request: TextAnalyzeRequest):
    urls = _extract_urls_from_text(request.text or "")
    if not urls:
        return {
            "count": 0,
            "unique_count": 0,
            "deduplicated": False,
            "duration_sec": 0.0,
            "results": [],
        }
    return await analyze_url_ml_batch(URLBatchRequest(urls=urls))


@app.post("/analyze/engine")
async def analyze_engine_only(request: URLRequest):
    """KoBERT lane with URLML preflight for decisive URL-only cases."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    url_ml_hint = None
    if KOBERT_URL_ML_PREFLIGHT:
        url_ml_hint = await _url_ml_hint_for_request(target_url)
        if _should_fast_path_after_urlml(url_ml_hint):
            return {
                **_url_ml_preflight_model_result("KoBERT", url_ml_hint),
                "engine_status": getattr(app.state, "eng_status", {"enabled": False}),
            }
    eng = getattr(app.state, "eng", None)
    if eng is not None:
        result = await _run_kobert_request(target_url)
    else:
        result = {
            "judgment": "unknown",
            "riskLevel": "UNKNOWN",
            "risklevel": "UNKNOWN",
            "engine_disabled": True,
            "engine_reason": getattr(app.state, "eng_status", {}).get("reason", "not_loaded"),
        }
    result = _apply_url_rule_adjustment(
        "KoBERT",
        target_url,
        result,
        _url_rule_adjustment(_url_cache_key(target_url)),
    )
    if url_ml_hint is not None:
        result = _apply_url_ml_hint_to_model("KoBERT", result, url_ml_hint)
    return {
        **result,
        "engine_status": getattr(app.state, "eng_status", {"enabled": False}),
    }


@app.post("/analyze/xgboost")
async def analyze_xgboost_only(request: URLRequest):
    """XGBoost lane with URLML preflight; use /analyze/gnn for GNN separately."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    url_ml_hint = await _url_ml_hint_for_request(target_url)
    if _should_fast_path_after_urlml(url_ml_hint):
        return {
            "xgboost": _url_ml_preflight_model_result("XGBoost", url_ml_hint),
            "xgboost_status": getattr(app.state, "xg_status", {"enabled": False}),
        }
    xg_result = await _run_xgboost_request(target_url)
    xg_result = _apply_url_rule_adjustment(
        "XGBoost",
        target_url,
        xg_result,
        _url_rule_adjustment(_url_cache_key(target_url)),
    )
    xg_result = _apply_url_ml_hint_to_model("XGBoost", xg_result, url_ml_hint)
    return {
        "xgboost": xg_result,
        "xgboost_status": getattr(app.state, "xg_status", {"enabled": False}),
    }


@app.post("/analyze/gnn")
async def analyze_gnn_only(request: URLRequest):
    """Web-structure GNN lane with URLML preflight."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    url_ml_hint = await _url_ml_hint_for_request(target_url)
    if _should_fast_path_after_urlml(url_ml_hint):
        return {
            "gnn": _url_ml_preflight_model_result("GNN", url_ml_hint),
            "gnn_status": getattr(app.state, "gnn_status", {"enabled": False}),
        }
    gnn_result = await _run_gnn_request(target_url)
    gnn_result = _apply_url_rule_adjustment(
        "GNN",
        target_url,
        gnn_result,
        _url_rule_adjustment(_url_cache_key(target_url)),
    )
    gnn_result = _apply_url_ml_hint_to_model("GNN", gnn_result, url_ml_hint)
    return {
        "gnn": gnn_result,
        "gnn_status": getattr(app.state, "gnn_status", {"enabled": False}),
    }


@app.post("/analyze/url-ml")
async def analyze_url_ml_only(request: URLRequest):
    """Fast URL lexical ML only."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    url_ml_result = await _run_url_ml(_run_url_ml_inference, target_url)
    url_ml_result = _apply_url_rule_adjustment(
        "URLML",
        target_url,
        url_ml_result,
        _url_rule_adjustment(_url_cache_key(target_url)),
    )
    return {
        "url_ml": url_ml_result,
        "url_ml_status": getattr(app.state, "url_ml_status", {"enabled": False}),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
