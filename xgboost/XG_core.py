"""Shared XGBoost URL feature/model utilities."""


from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import socket
import ssl
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

try:
    from xgboost import XGBClassifier
except Exception as e:  # pragma: no cover
    XGBClassifier = None  # type: ignore
    _XGBOOST_IMPORT_ERROR = e

try:
    import joblib
except Exception as e:  # pragma: no cover
    joblib = None  # type: ignore
    _JOBLIB_IMPORT_ERROR = e

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

# ============================================================
# 1. Common helpers (공용 보조 함수)
# ============================================================

_RE_IP = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[0-1]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[0-1]?\d?\d)$"
)

def _safe_lower(s: str) -> str:
    try:
        return s.lower()
    except Exception:
        return str(s).lower()

def _count_digits(s: str) -> int:
    return sum(ch.isdigit() for ch in s)

def _count_letters(s: str) -> int:
    return sum(ch.isalpha() for ch in s)

def _count_substring(s: str, sub: str) -> int:
    if not sub:
        return 0
    return s.count(sub)


def xgboost_ensemble_verdict_label(
    prob_typo: float, prob_domain: float, prob_dom: float
) -> int:
    """Malicious/ben label from typo, domain-age, and DOM head probabilities (CLI + API)."""
    if (
        prob_typo >= 0.90
        or prob_domain >= 0.85
        or prob_dom >= 0.85
    ):
        return 1
    if (
        (prob_typo >= 0.50 and prob_domain >= 0.45)
        or (prob_typo >= 0.45 and prob_domain >= 0.50)
        or (prob_typo >= 0.50 and prob_dom >= 0.45)
        or (prob_typo >= 0.45 and prob_dom >= 0.50)
        or (prob_domain >= 0.50 and prob_dom >= 0.45)
        or (prob_domain >= 0.45 and prob_dom >= 0.50)
    ):
        return 1
    return 0


COMMON_MULTI_TLDS = {
    "co.uk", "gov.uk", "ac.uk",
    "co.kr", "go.kr", "or.kr",
    "co.jp", "go.jp",
    "gov.cn"
}

def has_multi_level_tld(host: str) -> int:
    host = host.lower()
    return 1 if any(host.endswith("." + tld) or host == tld for tld in COMMON_MULTI_TLDS) else 0

COMMON_CCTLDS = {
    "uk", "cn", "kr", "jp", "de", "fr", "au", "ca"
}

SAFE_SECOND_LEVEL_HINTS = {
    "gov", "co", "ac", "edu", "or", "go"
}

def has_country_code_tld(host: str) -> int:
    host = host.lower().strip(".")
    parts = host.split(".")
    if not parts:
        return 0
    return 1 if parts[-1] in COMMON_CCTLDS else 0

def has_safe_second_level_hint(host: str) -> int:
    host = host.lower().strip(".")
    parts = host.split(".")
    if len(parts) < 2:
        return 0
    return 1 if any(part in SAFE_SECOND_LEVEL_HINTS for part in parts[:-1]) else 0

def _split_host_labels(host: str) -> List[str]:
    return [part for part in (host or "").lower().split(".") if part]

def _get_tld(host: str) -> str:
    parts = _split_host_labels(host)
    return parts[-1] if parts else ""

def _get_subdomain_labels(host: str) -> List[str]:
    parts = _split_host_labels(host)
    if len(parts) <= 2:
        return []
    return parts[:-2]

def _get_sld(host: str) -> str:
    """Extract second-level domain from host (e.g. google.com -> google, www.google.co.uk -> google)."""
    host = host.strip(".").lower()
    if not host or _RE_IP.match(host):
        return ""
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return parts[0] if parts else ""
    # Use second-to-last part as SLD (e.g. google.com -> google; www.google.co.uk -> google).
    return parts[-2] if len(parts) >= 2 else parts[0]

# ============================================================
# 2. Typosquatting features (타이포스쿼팅 특징 추출)
# ============================================================

_DEFAULT_BRAND_DICTIONARY: List[str] = [
    "google", "facebook", "amazon", "paypal", "apple",
    "microsoft", "netflix", "instagram", "twitter",
    "linkedin", "github", "yahoo",
]

_DEFAULT_BRAND_DICTIONARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "brands_top300.txt",
)

def _load_brand_dictionary(path: str = _DEFAULT_BRAND_DICTIONARY_PATH) -> List[str]:
    """Load brands from a newline-delimited file, falling back to built-ins if unavailable."""
    brands: List[str] = []
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#", 1)[0].strip().lower()
                if not line or line in seen:
                    continue
                brands.append(line)
                seen.add(line)
    except OSError:
        return list(_DEFAULT_BRAND_DICTIONARY)
    return brands or list(_DEFAULT_BRAND_DICTIONARY)

BRAND_DICTIONARY: List[str] = _load_brand_dictionary()

def _levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein (insert, delete, substitute)."""
    a, b = a.lower(), b.lower()
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]

def _damerau_levenshtein(a: str, b: str) -> int:
    """Damerau-Levenshtein (adds adjacent transposition)."""
    a, b = a.lower(), b.lower()
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    # Inf sentinel
    inf = n + m
    da: Dict[str, int] = {}
    for c in (a + b):
        da[c] = 0
    h = [[0] * (m + 2) for _ in range(n + 2)]
    for i in range(n + 2):
        h[i][0] = inf
        h[i][1] = i - 1
    for j in range(m + 2):
        h[0][j] = inf
        h[1][j] = j - 1
    h[1][1] = 0
    for i in range(2, n + 2):
        db = 0
        for j in range(2, m + 2):
            i1 = da.get(b[j - 2], 0)
            j1 = db
            cost = 1 if a[i - 2] != b[j - 2] else 0
            if cost == 0:
                db = j
            h[i][j] = min(
                h[i - 1][j] + 1,
                h[i][j - 1] + 1,
                h[i - 1][j - 1] + cost,
                h[i1][j1] + (i - 2 - i1) + 1 + (j - 2 - j1),
            )
        da[a[i - 2]] = i - 1
    return int(h[n + 1][m + 1])

def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity in [0, 1]. Empty -> 0."""
    a, b = a.lower(), b.lower()
    if not a or not b:
        return 0.0

    def ngrams(s: str) -> set:
        return set(s[i : i + n] for i in range(len(s) - n + 1)) if len(s) >= n else set()

    ga, gb = ngrams(a), ngrams(b)
    if not ga and not gb:
        return 1.0
    inter = len(ga & gb)
    union = len(ga | gb)
    return inter / union if union else 0.0

def _repeated_char_count(s: str) -> int:
    """Count of repeated consecutive character pairs (e.g. gooogle -> 2 for oo, oo)."""
    if len(s) < 2:
        return 0
    count = 0
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            count += 1
    return count

def _missing_char_score(sld: str, brand: str) -> float:
    """Rough score: chars in brand not in sld (normalized by brand length)."""
    if not brand:
        return 0.0
    set_sld = set(sld)
    missing = sum(1 for c in brand if c not in set_sld)
    return float(missing) / len(brand)

def _adjacent_transposition_indicator(sld: str, brands: List[str]) -> float:
    """1.0 if sld is one adjacent swap away from any brand, else 0."""
    sld = sld.lower()
    for b in brands:
        b = b.lower()
        if len(sld) != len(b):
            continue
        diffs = [i for i in range(len(sld)) if sld[i] != b[i]]
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            if sld[diffs[0]] == b[diffs[1]] and sld[diffs[1]] == b[diffs[0]]:
                return 1.0
    return 0.0

_KEYBOARD_NEIGHBORS: Dict[str, List[str]] = {
    "q": ["w", "a"], "w": ["q", "e", "s", "a"], "e": ["w", "r", "d", "s"], "r": ["e", "t", "f", "d"],
    "t": ["r", "y", "g", "f"], "y": ["t", "u", "h", "g"], "u": ["y", "i", "j", "h"], "i": ["u", "o", "k", "j"],
    "o": ["i", "p", "l", "k"], "p": ["o", "l"], "a": ["q", "w", "s", "z"], "s": ["a", "w", "e", "d", "z", "x"],
    "d": ["s", "e", "r", "f", "x", "c"], "f": ["d", "r", "t", "g", "c", "v"], "g": ["f", "t", "y", "h", "v", "b"],
    "h": ["g", "y", "u", "j", "b", "n"], "j": ["h", "u", "i", "k", "n", "m"], "k": ["j", "i", "o", "l", "m"],
    "l": ["k", "o", "p"], "z": ["a", "s", "x"], "x": ["z", "s", "d", "c"], "c": ["x", "d", "f", "v"],
    "v": ["c", "f", "g", "b"], "b": ["v", "g", "h", "n"], "n": ["b", "h", "j", "m"], "m": ["n", "j", "k"],
}

def _keyboard_neighbor_substitution_count(sld: str, brand: str) -> int:
    """Count positions where sld has a keyboard neighbor of the brand character."""
    sld, brand = sld.lower(), brand.lower()
    if len(sld) != len(brand):
        return 0
    count = 0
    for i in range(len(sld)):
        bc = brand[i]
        sc = sld[i]
        if sc != bc and bc in _KEYBOARD_NEIGHBORS and sc in _KEYBOARD_NEIGHBORS[bc]:
            count += 1
    return count

def _homoglyph_reverse_score(host: str) -> float:
    """Score for digits/symbols that look like letters (0->o, 1->l, etc). High = likely typosquatting."""
    # Reverse mapping: character -> could it be a homoglyph substitute?
    rev = {"0": 1, "1": 1, "3": 1, "4": 1, "5": 1, "8": 1, "9": 1, "!": 1, "@": 1, "$": 1}
    return float(sum(rev.get(c, 0) for c in host.lower()))

def _sld_vs_brand_features(sld: str) -> Tuple[
    float, float, float, float, float, float, float, float
]:
    """SLD-vs-brand features. Returns (damerau_lev, norm_edit, jaccard3, repeated_count,
    missing_score, adj_trans_ind, kbd_neighbor_count, homoglyph_reverse)."""
    sld = (sld or "").lower()
    if not sld or not BRAND_DICTIONARY:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    best_lev = min(_levenshtein(sld, b) for b in BRAND_DICTIONARY)
    best_dl = min(_damerau_levenshtein(sld, b) for b in BRAND_DICTIONARY)
    closest_brand = min(BRAND_DICTIONARY, key=lambda b: _levenshtein(sld, b))
    max_len = max(len(sld), len(closest_brand), 1)
    norm_edit = float(best_lev) / max_len
    jaccard3 = max(_ngram_jaccard(sld, b, 3) for b in BRAND_DICTIONARY)
    repeated = float(_repeated_char_count(sld))
    missing = _missing_char_score(sld, closest_brand)
    adj_trans = _adjacent_transposition_indicator(sld, BRAND_DICTIONARY)
    kbd_count = float(_keyboard_neighbor_substitution_count(sld, closest_brand))
    homoglyph_rev = _homoglyph_reverse_score(sld)
    return (float(best_dl), norm_edit, jaccard3, repeated, missing, adj_trans, kbd_count, homoglyph_rev)

def _closest_brand(sld: str) -> Optional[str]:
    """Return the brand with minimum Levenshtein distance to sld, or None if no brands."""
    sld = (sld or "").lower()
    if not sld or not BRAND_DICTIONARY:
        return None
    return min(BRAND_DICTIONARY, key=lambda b: _levenshtein(sld, b))

_HOMOGLYPH_CHARS = set("01!3$45689@")  # 0→o, 1→l/i, !→i, 3→e, $→s, 4→a, 5→s, 6→g, 8→b, 9→g, @→a

_VOWEL_LIKE_DIGITS = set("0134")        # 0→o, 1→i/l, 3→e, 4→a

def _domain_digit_letter_ratio(host: str) -> float:
    """Digit-to-letter ratio in domain (host) only. Typosquatting often mixes digits with letters."""
    if not host:
        return 0.0
    d = _count_digits(host)
    L = _count_letters(host)
    total = d + L
    if total == 0:
        return 0.0
    return float(d) / float(total)

def _domain_homoglyph_count(host: str) -> int:
    """Count of visually confusing characters in domain (0, 1, !, 3, 4, etc.)."""
    return sum(1 for c in host if c in _HOMOGLYPH_CHARS)

def _domain_vowel_like_digit_count(host: str) -> int:
    """Count of digits that commonly replace vowels in domain: 0(o), 1(i/l), 3(e), 4(a)."""
    return sum(1 for c in host if c in _VOWEL_LIKE_DIGITS)

def _domain_letter_digit_alternations(host: str) -> int:
    """Number of transitions between letter and digit in domain. High = suspicious mix (e.g. g0o0gle)."""
    if len(host) < 2:
        return 0
    count = 0
    for i in range(len(host) - 1):
        a, b = host[i], host[i + 1]
        a_letter, a_digit = a.isalpha(), a.isdigit()
        b_letter, b_digit = b.isalpha(), b.isdigit()
        if (a_letter and b_digit) or (a_digit and b_letter):
            count += 1
    return count

_SUSPICIOUS_BRAND_KEYWORDS: Tuple[str, ...] = (
    "login",
    "secure",
    "auth",
    "verify",
    "verification",
    "update",
    "account",
    "support",
    "check",
    "center",
    "warning",
    "confirm",
)

BRAND_TARGET_TOKENS: List[str] = [
    "account", "user", "member", "customer", "client",
]

BRAND_ACTION_TOKENS: List[str] = [
    "login", "verify", "auth", "secure", "update",
    "support", "help", "center", "confirm", "service",
]

def _get_brand_tokens_in_host(host: str) -> List[str]:
    host = (host or "").lower()
    if not host:
        return []
    matched_brands: List[str] = []
    seen = set()
    for brand in BRAND_DICTIONARY:
        if not brand or brand in seen:
            continue
        if brand in host:
            matched_brands.append(brand)
            seen.add(brand)
    return matched_brands

def _brand_token_in_subdomain(host: str) -> float:
    subdomains = _get_subdomain_labels(host)
    if not subdomains:
        return 0.0
    for label in subdomains:
        for brand in _get_brand_tokens_in_host(label):
            if brand and brand in label:
                return 1.0
    return 0.0

def _brand_plus_keyword_pattern(host: str) -> float:
    host = (host or "").lower()
    brand_tokens = _get_brand_tokens_in_host(host)
    if not host or not brand_tokens:
        return 0.0

    compact_host = re.sub(r"[-._]", "", host)
    for brand in brand_tokens:
        compact_brand = re.sub(r"[-._]", "", brand)
        if not compact_brand:
            continue
        for keyword in _SUSPICIOUS_BRAND_KEYWORDS:
            compact_keyword = re.sub(r"[-._]", "", keyword)
            if not compact_keyword:
                continue
            if compact_brand + compact_keyword in compact_host:
                return 1.0
            if compact_keyword + compact_brand in compact_host:
                return 1.0
            if brand in host and keyword in host:
                return 1.0
    return 0.0

def _brand_hyphen_compound(host: str) -> float:
    host = (host or "").lower()
    if "-" not in host:
        return 0.0
    for label in _split_host_labels(host):
        if "-" not in label:
            continue
        if _get_brand_tokens_in_host(label):
            return 1.0
    return 0.0

def _brand_target_action_pattern(host: str) -> float:
    host = (host or "").lower()
    if not host:
        return 0.0

    compact_host = re.sub(r"[-._]", "", host)
    brand_tokens = _get_brand_tokens_in_host(compact_host)
    if not compact_host or not brand_tokens:
        return 0.0

    compact_targets = [
        re.sub(r"[-._]", "", token)
        for token in BRAND_TARGET_TOKENS
        if token
    ]
    compact_actions = [
        re.sub(r"[-._]", "", token)
        for token in BRAND_ACTION_TOKENS
        if token
    ]

    for brand in brand_tokens:
        compact_brand = re.sub(r"[-._]", "", brand)
        if not compact_brand or compact_brand not in compact_host:
            continue
        has_target = any(token and token in compact_host for token in compact_targets)
        has_action = any(token and token in compact_host for token in compact_actions)
        if has_target or has_action:
            return 1.0
    return 0.0

# ============================================================
# 3. URL length / lexical features (URL 길이 및 구조 특징)
# ============================================================

def extract_length_features(url: str) -> Dict[str, float]:
    """Scaled length helpers only."""
    u = (url or "").strip()
    parsed = urlsplit(u if "://" in u else "http://" + u)
    host = _safe_lower(parsed.hostname or "")
    # www 정규화 추가 (extract_features와 host_length 기준 통일)
    if host.startswith("www."):
        host = host[4:]
    return {
        "url_length": float(len(u)) / 100.0,
        "host_length": float(len(host)) / 50.0,
    }

# ============================================================
# 4. Domain age / RDAP features (도메인 나이)
# ============================================================

_RDAP_LOOKUP_TIMEOUT_SECONDS = 3.0
_DOMAIN_AGE_MAX_DAYS = 36500.0
_DOMAIN_AGE_LOOKUP_FAILED = {
    "domain_age_days": 0.0,
    "domain_age_log_days": 0.0,
    "domain_age_missing": 1.0,
    "rdap_status_ok": 0.0,
    "rdap_status_not_registered": 0.0,
    "rdap_status_lookup_failed": 1.0,
    "rdap_status_parse_failed": 0.0,
}
_DOMAIN_AGE_NOT_REGISTERED = {
    "domain_age_days": 0.0,
    "domain_age_log_days": 0.0,
    "domain_age_missing": 1.0,
    "rdap_status_ok": 0.0,
    "rdap_status_not_registered": 1.0,
    "rdap_status_lookup_failed": 0.0,
    "rdap_status_parse_failed": 0.0,
}
_DOMAIN_AGE_PARSE_FAILED = {
    "domain_age_days": 0.0,
    "domain_age_log_days": 0.0,
    "domain_age_missing": 1.0,
    "rdap_status_ok": 0.0,
    "rdap_status_not_registered": 0.0,
    "rdap_status_lookup_failed": 0.0,
    "rdap_status_parse_failed": 1.0,
}
_DOMAIN_AGE_DISABLED = {
    "domain_age_days": 0.0,
    "domain_age_log_days": 0.0,
    "domain_age_missing": 1.0,
    "rdap_status_ok": 0.0,
    "rdap_status_not_registered": 0.0,
    "rdap_status_lookup_failed": 0.0,
    "rdap_status_parse_failed": 0.0,
}
_DOMAIN_AGE_CACHE: Dict[str, Dict[str, float]] = {}
_NETWORK_CACHE: Dict[str, Dict[str, Any]] = {}

def normalize_host_for_network(url: str) -> Dict[str, Any]:
    """
    Returns:
    {
        "raw_host": str,
        "ascii_host": str,
        "registered_domain": str,
        "ascii_registered_domain": str,
        "dns_resolved": bool
    }
    """
    raw_host = _extract_host_for_domain_age(url)
    cache_key = raw_host or (url or "").strip()
    cached = _NETWORK_CACHE.get(cache_key)
    if cached is not None:
        print(f"[NETWORK DEBUG] raw_host={cached['raw_host']}")
        print(f"[NETWORK DEBUG] ascii_host={cached['ascii_host']}")
        print(f"[NETWORK DEBUG] registered_domain={cached['registered_domain']}")
        print(f"[NETWORK DEBUG] ascii_registered_domain={cached['ascii_registered_domain']}")
        print(f"[NETWORK DEBUG] dns_resolved={cached['dns_resolved']}")
        return dict(cached)

    ascii_host = raw_host
    if raw_host:
        try:
            ascii_host = raw_host.encode("idna").decode("ascii")
        except Exception:
            ascii_host = raw_host

    registered_domain = _get_registered_domain_for_rdap(raw_host)
    ascii_registered_domain = registered_domain
    if registered_domain:
        try:
            ascii_registered_domain = registered_domain.encode("idna").decode("ascii")
        except Exception:
            ascii_registered_domain = registered_domain

    dns_resolved = False
    if ascii_host:
        try:
            socket.gethostbyname(ascii_host)
            dns_resolved = True
        except Exception:
            dns_resolved = False

    info = {
        "raw_host": raw_host,
        "ascii_host": ascii_host,
        "registered_domain": registered_domain,
        "ascii_registered_domain": ascii_registered_domain,
        "dns_resolved": dns_resolved,
    }
    _NETWORK_CACHE[cache_key] = dict(info)

    print(f"[NETWORK DEBUG] raw_host={raw_host}")
    print(f"[NETWORK DEBUG] ascii_host={ascii_host}")
    print(f"[NETWORK DEBUG] registered_domain={registered_domain}")
    print(f"[NETWORK DEBUG] ascii_registered_domain={ascii_registered_domain}")
    print(f"[NETWORK DEBUG] dns_resolved={dns_resolved}")
    return dict(info)

def _extract_host_for_domain_age(url_or_host: str) -> str:
    raw = (url_or_host or "").strip()
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
        host = _safe_lower((parsed.hostname or "").strip("."))
    except Exception:
        host = ""
    if not host:
        fallback = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        host = _safe_lower(fallback.split(":", 1)[0].strip("."))
    return host

def _get_registered_domain_for_rdap(url_or_host: str) -> str:
    host = _extract_host_for_domain_age(url_or_host)
    if not host or _RE_IP.match(host):
        return ""
    parts = [part for part in host.split(".") if part]
    if len(parts) < 2:
        return host
    suffix = ".".join(parts[-2:])
    if suffix in COMMON_MULTI_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])

def _fetch_rdap_payload(registered_domain: str) -> Tuple[Optional[Dict[str, Any]], str]:
    if not registered_domain:
        return None, "lookup_failed"
    request = Request(
        f"https://rdap.org/domain/{registered_domain}",
        headers={
            "Accept": "application/rdap+json, application/json",
            "User-Agent": "Mozilla/5.0",
        },
    )
    try:
        with urlopen(request, timeout=_RDAP_LOOKUP_TIMEOUT_SECONDS) as response:
            payload = response.read()
            charset = response.headers.get_content_charset() or "utf-8"
    except HTTPError as e:
        print(f"[RDAP DEBUG] http_status={e.code}")
        if e.code == 404:
            return None, "not_registered"
        return None, "lookup_failed"
    except TimeoutError:
        print("[RDAP DEBUG] timeout")
        return None, "lookup_failed"
    except URLError as e:
        print(f"[RDAP DEBUG] url_error={e}")
        return None, "lookup_failed"
    except (ValueError, OSError):
        return None, "lookup_failed"
    try:
        parsed = json.loads(payload.decode(charset, errors="replace"))
    except Exception:
        print("[RDAP DEBUG] json_parse_failed")
        return None, "lookup_failed"
    if not isinstance(parsed, dict):
        return None, "lookup_failed"
    return parsed, "ok"

def _parse_rdap_datetime(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    candidates = [text]
    if text.endswith("Z"):
        candidates.append(text[:-1] + "+00:00")
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    return None

def _extract_rdap_creation_date(payload: Optional[Dict[str, Any]]) -> Optional[datetime]:
    if not isinstance(payload, dict):
        return None

    events = payload.get("events")
    if isinstance(events, list):
        for event in events:
            if not isinstance(event, dict):
                continue
            action = str(event.get("eventAction", "")).strip().lower()
            event_date = event.get("eventDate")
            if action in {"registration", "registered", "creation", "created"} and isinstance(event_date, str):
                parsed = _parse_rdap_datetime(event_date)
                if parsed is not None:
                    return parsed
        for event in events:
            if not isinstance(event, dict):
                continue
            event_date = event.get("eventDate")
            if isinstance(event_date, str):
                parsed = _parse_rdap_datetime(event_date)
                if parsed is not None:
                    return parsed

    for key in ("created", "creationDate", "registrationDate", "createdDate"):
        value = payload.get(key)
        if isinstance(value, str):
            parsed = _parse_rdap_datetime(value)
            if parsed is not None:
                return parsed
    return None

def _compute_domain_age_days(created_at: datetime, now: Optional[datetime] = None) -> float:
    reference_time = now or datetime.now(timezone.utc)
    age_days = max(0.0, (reference_time - created_at).total_seconds() / 86400.0)
    return float(min(age_days, _DOMAIN_AGE_MAX_DAYS))

def extract_domain_age_features(url: str) -> Dict[str, float]:
    network_info = normalize_host_for_network(url)
    registered_domain = str(network_info.get("ascii_registered_domain", "") or "")
    print(f"[DOMAIN DEBUG] url={url}")
    print(f"[DOMAIN DEBUG] registered_domain={registered_domain}")
    print(f"[DOMAIN DEBUG] rdap_cache_hit={registered_domain in _DOMAIN_AGE_CACHE}")

    if not registered_domain:
        print("[RDAP DEBUG] status=lookup_failed")
        return dict(_DOMAIN_AGE_LOOKUP_FAILED)
    if not bool(network_info.get("dns_resolved", False)):
        print("[RDAP DEBUG] dns_failed_but_rdap_lookup_attempted")

    cached = _DOMAIN_AGE_CACHE.get(registered_domain)
    if cached is not None:
        return dict(cached)

    payload, rdap_status = _fetch_rdap_payload(registered_domain)
    print(f"[RDAP DEBUG] status={rdap_status}")
    print(f"[DOMAIN DEBUG] rdap_payload_exists={payload is not None}")

    if rdap_status == "not_registered":
        features = dict(_DOMAIN_AGE_NOT_REGISTERED)
        _DOMAIN_AGE_CACHE[registered_domain] = dict(features)
        return dict(features)
    if rdap_status == "lookup_failed":
        features = dict(_DOMAIN_AGE_LOOKUP_FAILED)
        # lookup_failed is cached for the current run only to avoid repeated network delays.
        _DOMAIN_AGE_CACHE[registered_domain] = dict(features)
        return dict(features)

    created_at = _extract_rdap_creation_date(payload)
    if created_at is None:
        print("[RDAP DEBUG] creation_date_parse_failed")
        features = dict(_DOMAIN_AGE_PARSE_FAILED)
        _DOMAIN_AGE_CACHE[registered_domain] = dict(features)
        return dict(features)

    domain_age_days = _compute_domain_age_days(created_at)
    print(f"[DOMAIN DEBUG] domain_age_days={domain_age_days}")
    features = {
        "domain_age_days": float(domain_age_days),
        "domain_age_log_days": float(math.log1p(domain_age_days)),
        "domain_age_missing": 0.0,
        "rdap_status_ok": 1.0,
        "rdap_status_not_registered": 0.0,
        "rdap_status_lookup_failed": 0.0,
        "rdap_status_parse_failed": 0.0,
    }
    _DOMAIN_AGE_CACHE[registered_domain] = dict(features)
    return dict(features)

def get_domain_age_features_for_mode(url: str, enable_domain_age: bool) -> Dict[str, float]:
    if not enable_domain_age:
        return dict(_DOMAIN_AGE_DISABLED)
    return extract_domain_age_features(url)

# ============================================================
# 5. SSL certificate features (SSL 유효기간)
# ============================================================

_SSL_LOOKUP_TIMEOUT_SECONDS = 3.0
_SSL_CERT_MAX_DAYS = 36500.0
_SSL_FALLBACK = {
    "ssl_valid_days": 0.0,
    "ssl_remaining_days": 0.0,
    "ssl_age_days": 0.0,
    "ssl_missing": 1.0,
    "ssl_status_no_cert": 0.0,
    "ssl_status_lookup_failed": 1.0,
}
_SSL_FALLBACK_NO_CERT = {
    "ssl_valid_days": 0.0,
    "ssl_remaining_days": 0.0,
    "ssl_age_days": 0.0,
    "ssl_missing": 1.0,
    "ssl_status_no_cert": 1.0,
    "ssl_status_lookup_failed": 0.0,
}
_SSL_FALLBACK_LOOKUP_FAILED = {
    "ssl_valid_days": 0.0,
    "ssl_remaining_days": 0.0,
    "ssl_age_days": 0.0,
    "ssl_missing": 1.0,
    "ssl_status_no_cert": 0.0,
    "ssl_status_lookup_failed": 1.0,
}
_SSL_CACHE: Dict[str, Dict[str, float]] = {}

def _parse_ssl_cert_datetime(value: str) -> Optional[datetime]:
    text = (value or "").strip()
    if not text:
        return None
    for fmt in ("%b %d %H:%M:%S %Y %Z", "%Y%m%d%H%M%SZ"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def _clamp_ssl_days(value: float) -> float:
    return float(min(max(0.0, value), _SSL_CERT_MAX_DAYS))

def extract_ssl_features(url: str) -> Dict[str, float]:
    parsed = urlsplit(url if "://" in (url or "") else "http://" + (url or ""))
    network_info = normalize_host_for_network(url)
    if not bool(network_info.get("dns_resolved", False)):
        print("[SSL DEBUG] skipped due to DNS failure")
        return dict(_SSL_FALLBACK_LOOKUP_FAILED)
    host = str(network_info.get("ascii_host", "") or "")
    print(f"[SSL DEBUG] url={url}")
    print(f"[SSL DEBUG] scheme={parsed.scheme}")
    print(f"[SSL DEBUG] host={host}")
    print(f"[DOMAIN DEBUG] ssl_host={host}")
    print(f"[DOMAIN DEBUG] ssl_cache_hit={host in _SSL_CACHE}")

    if not host:
        return dict(_SSL_FALLBACK_LOOKUP_FAILED)

    cached = _SSL_CACHE.get(host)
    if cached is not None:
        return dict(cached)

    context = ssl.create_default_context()
    try:
        with socket.create_connection((host, 443), timeout=_SSL_LOOKUP_TIMEOUT_SECONDS) as sock:
            with context.wrap_socket(sock, server_hostname=host) as tls_sock:
                cert = tls_sock.getpeercert()
    except socket.gaierror:
        print("[SSL DEBUG] DNS RESOLUTION FAILED")
        print("[DOMAIN DEBUG] SSL FAILED -> fallback")
        _SSL_CACHE[host] = dict(_SSL_FALLBACK_LOOKUP_FAILED)
        return dict(_SSL_FALLBACK_LOOKUP_FAILED)
    except socket.timeout:
        print("[SSL DEBUG] TCP CONNECTION TIMEOUT")
        print("[DOMAIN DEBUG] SSL FAILED -> fallback")
        _SSL_CACHE[host] = dict(_SSL_FALLBACK_LOOKUP_FAILED)
        return dict(_SSL_FALLBACK_LOOKUP_FAILED)
    except ConnectionRefusedError:
        print("[SSL DEBUG] CONNECTION REFUSED (port 443 closed)")
        print("[DOMAIN DEBUG] SSL FAILED -> fallback")
        _SSL_CACHE[host] = dict(_SSL_FALLBACK_NO_CERT)
        return dict(_SSL_FALLBACK_NO_CERT)
    except ssl.SSLError as e:
        print(f"[SSL DEBUG] SSL HANDSHAKE FAILED: {e}")
        print("[DOMAIN DEBUG] SSL FAILED -> fallback")
        _SSL_CACHE[host] = dict(_SSL_FALLBACK_NO_CERT)
        return dict(_SSL_FALLBACK_NO_CERT)
    except (socket.error, ValueError, OSError):
        print("[SSL DEBUG] UNKNOWN ERROR: socket/value/os level exception")
        print("[DOMAIN DEBUG] SSL FAILED -> fallback")
        _SSL_CACHE[host] = dict(_SSL_FALLBACK_LOOKUP_FAILED)
        return dict(_SSL_FALLBACK_LOOKUP_FAILED)
    except Exception as e:
        print(f"[SSL DEBUG] UNKNOWN ERROR: {e}")
        print("[DOMAIN DEBUG] SSL FAILED -> fallback")
        _SSL_CACHE[host] = dict(_SSL_FALLBACK_LOOKUP_FAILED)
        return dict(_SSL_FALLBACK_LOOKUP_FAILED)

    print("[SSL DEBUG] TLS HANDSHAKE SUCCESS")
    print(f"[SSL DEBUG] notBefore={cert.get('notBefore')}")
    print(f"[SSL DEBUG] notAfter={cert.get('notAfter')}")

    not_before = _parse_ssl_cert_datetime(str(cert.get("notBefore", "")))
    not_after = _parse_ssl_cert_datetime(str(cert.get("notAfter", "")))
    if not_before is None or not_after is None:
        print("[DOMAIN DEBUG] SSL FAILED -> fallback")
        _SSL_CACHE[host] = dict(_SSL_FALLBACK_LOOKUP_FAILED)
        return dict(_SSL_FALLBACK_LOOKUP_FAILED)

    now = datetime.now(timezone.utc)
    ssl_valid_days = _clamp_ssl_days((not_after - not_before).total_seconds() / 86400.0)
    ssl_remaining_days = _clamp_ssl_days((not_after - now).total_seconds() / 86400.0)
    ssl_age_days = _clamp_ssl_days((now - not_before).total_seconds() / 86400.0)
    print(f"[DOMAIN DEBUG] ssl_valid_days={ssl_valid_days}")
    print(f"[DOMAIN DEBUG] ssl_remaining_days={ssl_remaining_days}")
    print(f"[DOMAIN DEBUG] ssl_age_days={ssl_age_days}")
    features = {
        "ssl_valid_days": float(ssl_valid_days),
        "ssl_remaining_days": float(ssl_remaining_days),
        "ssl_age_days": float(ssl_age_days),
        "ssl_missing": 0.0,
        "ssl_status_no_cert": 0.0,
        "ssl_status_lookup_failed": 0.0,
    }
    _SSL_CACHE[host] = dict(features)
    return dict(features)

def get_ssl_features_for_mode(url: str, enable_ssl: bool) -> Dict[str, float]:
    if not enable_ssl:
        return dict(_SSL_FALLBACK)
    return extract_ssl_features(url)

def _bucketize_domain_age_features(domain_age_days: float, domain_age_missing: float) -> Dict[str, float]:
    if domain_age_missing >= 1.0:
        return {
            "domain_is_very_new": 0.0,
            "domain_is_new": 0.0,
            "domain_is_established": 0.0,
            "domain_is_old": 0.0,
        }

    return {
        "domain_is_very_new": 1.0 if domain_age_days <= 30.0 else 0.0,
        "domain_is_new": 1.0 if 30.0 < domain_age_days <= 180.0 else 0.0,
        "domain_is_established": 1.0 if 180.0 < domain_age_days <= 365.0 else 0.0,
        "domain_is_old": 1.0 if domain_age_days > 365.0 else 0.0,
    }

def _bucketize_ssl_features(
    ssl_valid_days: float,
    ssl_remaining_days: float,
    ssl_age_days: float,
    ssl_missing: float,
) -> Dict[str, float]:
    if ssl_missing >= 1.0:
        return {
            "ssl_is_short_lived": 0.0,
            "ssl_is_normal_lived": 0.0,
            "ssl_is_long_lived": 0.0,
            "ssl_expires_very_soon": 0.0,
            "ssl_expires_soon": 0.0,
            "ssl_expires_far": 0.0,
            "ssl_is_very_new": 0.0,
            "ssl_is_recent": 0.0,
            "ssl_is_mature": 0.0,
        }

    return {
        "ssl_is_short_lived": 1.0 if ssl_valid_days <= 90.0 else 0.0,
        "ssl_is_normal_lived": 1.0 if 90.0 < ssl_valid_days <= 398.0 else 0.0,
        "ssl_is_long_lived": 1.0 if ssl_valid_days > 398.0 else 0.0,
        "ssl_expires_very_soon": 1.0 if ssl_remaining_days <= 7.0 else 0.0,
        "ssl_expires_soon": 1.0 if 7.0 < ssl_remaining_days <= 30.0 else 0.0,
        "ssl_expires_far": 1.0 if ssl_remaining_days > 30.0 else 0.0,
        "ssl_is_very_new": 1.0 if ssl_age_days <= 7.0 else 0.0,
        "ssl_is_recent": 1.0 if 7.0 < ssl_age_days <= 30.0 else 0.0,
        "ssl_is_mature": 1.0 if ssl_age_days > 30.0 else 0.0,
    }

def extract_domain_only_features(
    url: str,
    enable_domain_age: bool,
    enable_ssl: bool = False,
) -> Dict[str, float]:
    features = get_domain_age_features_for_mode(url, enable_domain_age)
    features.update(
        _bucketize_domain_age_features(
            float(features.get("domain_age_days", 0.0)),
            float(features.get("domain_age_missing", 1.0)),
        )
    )
    ssl_features = get_ssl_features_for_mode(url, enable_ssl)
    features.update(ssl_features)
    features.update(
        _bucketize_ssl_features(
            float(ssl_features.get("ssl_valid_days", 0.0)),
            float(ssl_features.get("ssl_remaining_days", 0.0)),
            float(ssl_features.get("ssl_age_days", 0.0)),
            float(ssl_features.get("ssl_missing", 1.0)),
        )
    )
    return features

# ============================================================
# 6. DOM features (DOM 구조 특징)
# ============================================================

_DOM_FETCH_TIMEOUT_SECONDS = 5.0
_DOM_FETCH_FALLBACK = {
    "dom_max_depth": 0.0,
    "dead_link_ratio": 0.0,
    "hidden_tags_count": 0.0,
    "suspicious_form_action": 0.0,
    "dom_fetch_failed": 1.0,
}
_DOM_MODEL_FEATURE_NAMES: List[str] = [
    "dom_max_depth",
    "dead_link_ratio",
    "hidden_tags_count",
    "suspicious_form_action",
    "dom_fetch_failed"
]
_DOM_FEATURE_CACHE: Dict[str, Dict[str, float]] = {}

def _normalize_url_for_dom_fetch(url: str) -> str:
    raw = (url or "").strip()
    print(f"[DOM DEBUG] normalize_raw_url={raw}")
    if not raw:
        print("[DOM DEBUG] normalize_failed: empty url")
        return ""
    candidate = raw if "://" in raw else f"http://{raw}"
    print(f"[DOM DEBUG] normalize_candidate={candidate}")
    try:
        parsed = urlsplit(candidate)
    except Exception:
        print("[DOM DEBUG] normalize_failed: urlsplit exception")
        return ""
    normalized = candidate if parsed.hostname else ""
    if not normalized:
        print("[DOM DEBUG] normalize_failed: hostname missing")
    return normalized

def _fetch_html_for_dom(url: str) -> Tuple[str, bool]:
    network_info = normalize_host_for_network(url)
    if not bool(network_info.get("dns_resolved", False)):
        print("[DOM DEBUG] skipped due to DNS failure")
        return "", True

    target_url = _normalize_url_for_dom_fetch(url)
    if not target_url:
        print("[DOM DEBUG] normalize produced empty target_url")
        return "", True
    ascii_host = str(network_info.get("ascii_host", "") or "")
    if ascii_host:
        try:
            parsed = urlsplit(target_url)
            auth_part = ""
            if parsed.username:
                auth_part = parsed.username
                if parsed.password:
                    auth_part += f":{parsed.password}"
                auth_part += "@"
            port_part = f":{parsed.port}" if parsed.port else ""
            netloc = f"{auth_part}{ascii_host}{port_part}"
            target_url = parsed._replace(netloc=netloc).geturl()
        except Exception:
            pass
    if requests is None:
        print("[DOM DEBUG] requests library is not available")
        return "", True
    print(f"[DOM DEBUG] fetching_url={target_url}")
    try:
        response = requests.get(  # type: ignore[union-attr]
            target_url,
            timeout=_DOM_FETCH_TIMEOUT_SECONDS,
            headers={"User-Agent": "Mozilla/5.0"},
        )
    except requests.exceptions.Timeout:
        print("[DOM DEBUG] FETCH TIMEOUT")
        return "", True
    except requests.exceptions.SSLError as e:
        print(f"[DOM DEBUG] SSL ERROR: {e}")
        return "", True
    except requests.exceptions.ConnectionError as e:
        print(f"[DOM DEBUG] CONNECTION ERROR: {e}")
        return "", True
    except requests.exceptions.TooManyRedirects as e:
        print(f"[DOM DEBUG] TOO MANY REDIRECTS: {e}")
        return "", True
    except Exception as e:
        print(f"[DOM DEBUG] UNKNOWN FETCH ERROR: {type(e).__name__}: {e}")
        return "", True
    print(f"[DOM DEBUG] status_code={response.status_code}")
    print(f"[DOM DEBUG] final_url={response.url}")
    print(f"[DOM DEBUG] content_type={response.headers.get('Content-Type')}")
    print(f"[DOM DEBUG] html_length={len(response.text or '')}")
    if not response.ok:
        print(f"[DOM DEBUG] RESPONSE NOT OK: status_code={response.status_code}")
        return "", True
    return response.text or "", False

def _parse_dom_soup(html: str) -> Optional[Any]:
    if BeautifulSoup is None:
        return None
    try:
        return BeautifulSoup(html or "", "html.parser")
    except Exception:
        return None

def _get_dom_tree_depth(tag: Any, current_depth: int = 1) -> int:
    children = [child for child in getattr(tag, "children", []) if getattr(child, "name", None)]
    if not children:
        return current_depth
    return max(_get_dom_tree_depth(child, current_depth + 1) for child in children)

def _is_hidden_dom_tag(tag: Any) -> bool:
    if getattr(tag, "has_attr", lambda _: False)("hidden"):
        return True
    style = _safe_lower(str(tag.get("style", ""))).replace(" ", "")
    if "display:none" in style or "visibility:hidden" in style:
        return True
    class_values = tag.get("class") or []
    if isinstance(class_values, str):
        class_tokens = [_safe_lower(class_values)]
    else:
        class_tokens = [_safe_lower(str(token)) for token in class_values]
    return any(token in {"blind", "hidden", "sr-only"} for token in class_tokens)

def _has_suspicious_form_action(current_url: str, action: str) -> bool:
    action_text = (action or "").strip()
    if not action_text:
        return False
    try:
        parsed = urlsplit(action_text)
    except Exception:
        return False
    if _safe_lower(parsed.scheme) not in {"http", "https"}:
        return False
    current_registered_domain = _get_registered_domain_for_rdap(current_url)
    action_registered_domain = _get_registered_domain_for_rdap(action_text)
    if not current_registered_domain or not action_registered_domain:
        return False
    return current_registered_domain != action_registered_domain

def _extract_dom_features_from_html(html: str, current_url: str) -> Dict[str, float]:
    soup = _parse_dom_soup(html)
    if soup is None:
        print("[DOM DEBUG] BeautifulSoup parsing failed")
        return dict(_DOM_FETCH_FALLBACK)

    root_tags = [child for child in soup.children if getattr(child, "name", None)]
    dom_max_depth = float(max((_get_dom_tree_depth(tag, 1) for tag in root_tags), default=0))

    links = soup.find_all("a")
    dead_links = 0
    for link in links:
        href = _safe_lower(str(link.get("href", "")).strip())
        if href in {"", "#", "javascript:void(0);"} or href.startswith("javascript:"):
            dead_links += 1
    dead_link_ratio = float(dead_links) / float(len(links)) * 100.0 if links else 0.0

    hidden_tags_count = float(sum(1 for tag in soup.find_all(True) if _is_hidden_dom_tag(tag)))

    suspicious_form_action = 0.0
    for form in soup.find_all("form"):
        if _has_suspicious_form_action(current_url, str(form.get("action", ""))):
            suspicious_form_action = 1.0
            break

    return {
        "dom_max_depth": dom_max_depth,
        "dead_link_ratio": dead_link_ratio,
        "hidden_tags_count": hidden_tags_count,
        "suspicious_form_action": suspicious_form_action,
        "dom_fetch_failed": 0.0,
    }

def extract_dom_features(url: str) -> Dict[str, float]:
    target_url = _normalize_url_for_dom_fetch(url)
    print(f"[DOM DEBUG] input_url={url}")
    print(f"[DOM DEBUG] normalized_url={target_url}")
    if not target_url:
        return dict(_DOM_FETCH_FALLBACK)

    cached = _DOM_FEATURE_CACHE.get(target_url)
    if cached is not None:
        return dict(cached)

    html, fetch_failed = _fetch_html_for_dom(target_url)
    if fetch_failed:
        _DOM_FEATURE_CACHE[target_url] = dict(_DOM_FETCH_FALLBACK)
        return dict(_DOM_FETCH_FALLBACK)

    features = _extract_dom_features_from_html(html, target_url)
    _DOM_FEATURE_CACHE[target_url] = dict(features)
    return dict(features)

def _dom_feature_dict_to_array(
    feature_values: Dict[str, float],
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    names = feature_names or _DOM_MODEL_FEATURE_NAMES
    return np.array(
        [float(feature_values.get(name, 0.0)) for name in names],
        dtype=np.float32,
    )

def extract_dom_feature_array(url: str, feature_names: Optional[List[str]] = None) -> np.ndarray:
    dom_features = extract_dom_features(url)
    return _dom_feature_dict_to_array(dom_features, feature_names=feature_names)

# ============================================================
# 7. Error analysis (오탐/미탐 분석)
# ============================================================

_ANALYSIS_INFRA_HINTS = (
    "cdn",
    "api",
    "static",
    "asset",
    "assets",
    "edge",
    "cache",
    "cloudfront",
    "akamai",
    "gateway",
    "gw",
)

_ANALYSIS_MEDIA_HINTS = (
    "media",
    "video",
    "stream",
    "movie",
    "music",
    "audio",
    "img",
    "image",
    "photo",
    "gif",
    "news",
    "press",
    "times",
    "journal",
    "blog",
    "content",
    "scribd",
    "giphy",
)

_ANALYSIS_PUBLIC_TLDS = {"org", "gov", "edu", "ac", "go"}

_ANALYSIS_PUBLIC_HINTS = (
    "public",
    "official",
    "civic",
    "government",
    "ngo",
    "foundation",
    "service",
    "un",
    "npr",
    "wiki",
)

_ANALYSIS_TECH_HINTS = (
    "tech",
    "net",
    "cloud",
    "data",
    "sys",
    "dev",
    "dns",
    "ip",
    "sdk",
    "host",
    "labs",
    "compute",
)

_HOMOGLYPH_TRANSLATION = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
})

def _split_url_for_analysis(url: str) -> Tuple[str, str, str]:
    raw = (url or "").strip()
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
        host = _safe_lower((parsed.hostname or "").strip("."))
        path = _safe_lower(parsed.path or "")
    except Exception:
        host = ""
        path = ""
    if not host:
        fallback = raw.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        host = _safe_lower(fallback.split(":", 1)[0].strip("."))
    return host, path, _get_sld(host)

def _normalize_homoglyph_text(text: str) -> str:
    return (text or "").lower().translate(_HOMOGLYPH_TRANSLATION)

def _contains_any_token(text: str, tokens: Tuple[str, ...]) -> bool:
    lower_text = (text or "").lower()
    return any(token in lower_text for token in tokens)

def _closest_brand_with_distance(text: str) -> Tuple[Optional[str], int, int]:
    candidate = (text or "").lower()
    brand = _closest_brand(candidate)
    if not candidate or not brand:
        return None, 999, 999
    return brand, _levenshtein(candidate, brand), _damerau_levenshtein(candidate, brand)

def _looks_like_brand_service_pattern(sld: str) -> bool:
    sld = (sld or "").lower()
    if not sld:
        return False
    for brand in BRAND_DICTIONARY:
        if not brand or sld == brand:
            continue
        if sld.startswith(brand):
            suffix = sld[len(brand):].strip("-_.")
            if len(suffix) >= 2:
                return True
    return False

def _is_infra_service_url(host: str, path: str) -> bool:
    combined = f"{host}/{path}".lower()
    return any(token in combined for token in _ANALYSIS_INFRA_HINTS)

def _is_media_content_url(host: str, path: str) -> bool:
    combined = f"{host}/{path}".lower()
    return _contains_any_token(combined, _ANALYSIS_MEDIA_HINTS)

def _is_org_public_url(host: str, sld: str) -> bool:
    tld = _get_tld(host)
    if tld in _ANALYSIS_PUBLIC_TLDS:
        return True
    combined = f"{host}.{sld}".lower()
    return _contains_any_token(combined, _ANALYSIS_PUBLIC_HINTS)

def _is_numeric_tech_url(host: str) -> bool:
    lower_host = (host or "").lower()
    if any(ch.isdigit() for ch in lower_host):
        return True
    return _contains_any_token(lower_host, _ANALYSIS_TECH_HINTS)

def _is_homoglyph_typo(sld: str) -> bool:
    sld = (sld or "").lower()
    if not sld:
        return False
    normalized = _normalize_homoglyph_text(sld)
    if normalized == sld:
        return False
    brand = _closest_brand(normalized)
    if not brand:
        return False
    return _levenshtein(normalized, brand) <= 1 or _damerau_levenshtein(normalized, brand) <= 1

def _is_natural_typo(sld: str) -> bool:
    sld = (sld or "").lower()
    brand = _closest_brand(sld)
    if not sld or not brand:
        return False
    dist = _damerau_levenshtein(sld, brand)
    max_len = max(len(sld), len(brand), 1)
    return dist <= 1 or (dist <= 2 and float(dist) / float(max_len) <= 0.25)

def _is_transposition_typo(sld: str) -> bool:
    return _adjacent_transposition_indicator((sld or "").lower(), BRAND_DICTIONARY) >= 1.0

def _is_hyphen_phishing_url(host: str, sld: str) -> bool:
    if "-" not in host:
        return False
    compact = (sld or "").replace("-", "")
    if _is_homoglyph_typo(compact) or _is_natural_typo(compact):
        return True
    pieces = [part for part in re.split(r"[-_.]+", sld or "") if part]
    for piece in pieces:
        brand, lev, dl = _closest_brand_with_distance(piece)
        if brand and (piece == brand or lev <= 1 or dl <= 1):
            return True
    return _looks_like_brand_service_pattern(compact)

def _is_subdomain_attack_url(host: str) -> bool:
    labels = _split_host_labels(host)
    subdomains = _get_subdomain_labels(host)
    if len(labels) >= 4:
        return True
    for label in subdomains:
        brand, lev, dl = _closest_brand_with_distance(label)
        if brand and (brand in label or lev <= 1 or dl <= 1):
            return True
    return False

def classify_false_positive(url: str) -> str:
    host, path, sld = _split_url_for_analysis(url)
    if 0 < len(sld) <= 4:
        return "short_domain"
    if _is_infra_service_url(host, path):
        return "infra_cdn_api"
    if _looks_like_brand_service_pattern(sld):
        return "brand_service"
    if _is_media_content_url(host, path):
        return "media_content"
    if _is_org_public_url(host, sld):
        return "org_public"
    if _is_numeric_tech_url(host):
        return "numeric_tech"
    return "benign_other"

def classify_false_negative(url: str) -> str:
    host, _, sld = _split_url_for_analysis(url)
    if _is_homoglyph_typo(sld):
        return "homoglyph"
    if _is_transposition_typo(sld):
        return "transposition"
    if _is_hyphen_phishing_url(host, sld):
        return "hyphen_phishing"
    if _is_subdomain_attack_url(host):
        return "subdomain_attack"
    if _is_natural_typo(sld):
        return "natural_typo"
    return "benign_like_fake"

def _classify_false_positive_url(url: str) -> str:
    return classify_false_positive(url)

def _classify_false_negative_url(url: str) -> str:
    return classify_false_negative(url)

def _build_error_analysis_summary(
    false_positives: List[str],
    false_negatives: List[str],
) -> Dict[str, List[Dict[str, object]]]:
    fp_specs = [
        {
            "key": "short_domain",
            "label_en": "short_domain",
            "label_ko": "짧은 도메인",
            "detail_en": "Benign domains with very short SLDs that can look suspicious by length alone.",
            "detail_ko": "길이가 짧은 정상 도메인이 의심 URL로 잘못 분류된 경우",
        },
        {
            "key": "infra_cdn_api",
            "label_en": "infra_cdn_api",
            "label_ko": "인프라/CDN/API형",
            "detail_en": "Benign infrastructure, CDN, static, or API-style service URLs flagged as suspicious.",
            "detail_ko": "CDN, API, static 서버 등 정상 인프라 URL이 오탐된 경우",
        },
        {
            "key": "brand_service",
            "label_en": "brand_service",
            "label_ko": "브랜드+서비스형",
            "detail_en": "Benign domains where a brand-like token is followed by additional service wording.",
            "detail_ko": "브랜드명 뒤에 추가 문자열이 붙은 정상 서비스 URL이 오탐된 경우",
        },
        {
            "key": "media_content",
            "label_en": "media_content",
            "label_ko": "미디어/콘텐츠형",
            "detail_en": "Benign media, publishing, image, or content delivery domains flagged as suspicious.",
            "detail_ko": "영상, 이미지, 뉴스, 콘텐츠 계열의 정상 도메인이 오탐된 경우",
        },
        {
            "key": "org_public",
            "label_en": "org_public",
            "label_ko": "기관/공공서비스형",
            "detail_en": "Public, organizational, educational, or civic service domains flagged as suspicious.",
            "detail_ko": "기관, 공공, 교육, 비영리 성격의 정상 도메인이 오탐된 경우",
        },
        {
            "key": "numeric_tech",
            "label_en": "numeric_tech",
            "label_ko": "숫자/기술형 정상 도메인",
            "detail_en": "Benign domains with digits or technical service wording that resemble machine-generated patterns.",
            "detail_ko": "숫자나 기술 서비스 표현 때문에 의심스럽게 보이는 정상 도메인",
        },
        {
            "key": "benign_other",
            "label_en": "benign_other",
            "label_ko": "기타 정상 패턴",
            "detail_en": "Other benign domains that do not match the more specific false-positive patterns.",
            "detail_ko": "위 규칙에 직접 걸리지 않지만 정상으로 보이는 기타 오탐 사례",
        },
    ]
    fn_specs = [
        {
            "key": "natural_typo",
            "label_en": "natural_typo",
            "label_ko": "자연스러운 타이포",
            "detail_en": "Fake URLs whose typo distance is so small that they still look natural.",
            "detail_ko": "edit distance가 매우 낮아 정상처럼 보이는 fake URL",
        },
        {
            "key": "homoglyph",
            "label_en": "homoglyph",
            "label_ko": "유사문자 치환",
            "detail_en": "Fake URLs created with look-alike substitutions such as o->0 or l->1.",
            "detail_ko": "o->0, l->1 등 문자 치환으로 생성된 fake URL",
        },
        {
            "key": "transposition",
            "label_en": "transposition",
            "label_ko": "문자 순서 변경",
            "detail_en": "Fake URLs formed by swapping adjacent characters while keeping the overall shape familiar.",
            "detail_ko": "인접 문자의 순서를 바꿔 정상처럼 보이게 만든 fake URL",
        },
        {
            "key": "hyphen_phishing",
            "label_en": "hyphen_phishing",
            "label_ko": "하이픈 피싱형",
            "detail_en": "Fake URLs that inject hyphens around brand-like or typo-like segments to appear legitimate.",
            "detail_ko": "브랜드 또는 유사 문자열 주변에 하이픈을 넣어 정상처럼 보이게 만든 fake URL",
        },
        {
            "key": "subdomain_attack",
            "label_en": "subdomain_attack",
            "label_ko": "서브도메인 악용형",
            "detail_en": "Fake URLs that hide suspicious brand-like text in deeper subdomain structures.",
            "detail_ko": "깊은 서브도메인 구조를 이용해 브랜드 유사 문자열을 숨긴 fake URL",
        },
        {
            "key": "benign_like_fake",
            "label_en": "benign_like_fake",
            "label_ko": "정상처럼 보이는 fake",
            "detail_en": "Other fake URLs that look close enough to benign service domains to evade simple rules.",
            "detail_ko": "위 규칙에 딱 맞지 않지만 전반적으로 정상 URL처럼 보이는 fake URL",
        },
    ]

    fp_counts = {spec["key"]: 0 for spec in fp_specs}
    fn_counts = {spec["key"]: 0 for spec in fn_specs}

    for url in false_positives:
        category = _classify_false_positive_url(url)
        if category in fp_counts:
            fp_counts[category] += 1

    for url in false_negatives:
        category = _classify_false_negative_url(url)
        if category in fn_counts:
            fn_counts[category] += 1

    return {
        "false_positive_analysis": [
            {
                "type": spec["key"],
                "label_en": spec["label_en"],
                "label_ko": spec["label_ko"],
                "detail_en": spec["detail_en"],
                "detail_ko": spec["detail_ko"],
                "count": fp_counts[spec["key"]],
            }
            for spec in fp_specs
        ],
        "false_negative_analysis": [
            {
                "type": spec["key"],
                "label_en": spec["label_en"],
                "label_ko": spec["label_ko"],
                "detail_en": spec["detail_en"],
                "detail_ko": spec["detail_ko"],
                "count": fn_counts[spec["key"]],
            }
            for spec in fn_specs
        ],
    }

def _print_error_analysis_summary(summary: Dict[str, List[Dict[str, object]]]) -> None:
    print("\n[False Positive Analysis]")
    for item in summary.get("false_positive_analysis", []):
        print(f"* {item['type']}({item['label_ko']}): {item['count']}")
        print(f"  -> {item['detail_en']} / {item['detail_ko']}")

    print("\n[False Negative Analysis]")
    for item in summary.get("false_negative_analysis", []):
        print(f"* {item['type']}({item['label_ko']}): {item['count']}")
        print(f"  -> {item['detail_en']} / {item['detail_ko']}")

# ============================================================
# 8. Synthetic typo generation (합성 타이포 URL 생성)
# ============================================================

_LETTER_TO_HOMOGLYPH: Dict[str, List[str]] = {
    "o": ["0"], "l": ["1"], "i": ["1", "!"], "e": ["3"], "a": ["4", "@"],
    "s": ["5", "$"], "g": ["9"], "b": ["8"],
}

def _mutate_sld_homoglyph(sld: str, rng: random.Random) -> Optional[str]:
    """One homoglyph substitution in SLD (avoid first/last char)."""
    if len(sld) < 3:
        return None
    candidates = [i for i in range(1, len(sld) - 1) if sld[i] in _LETTER_TO_HOMOGLYPH]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    repl = rng.choice(_LETTER_TO_HOMOGLYPH[sld[idx]])
    return sld[:idx] + repl + sld[idx + 1 :]

def _mutate_sld_deletion(sld: str, rng: random.Random) -> Optional[str]:
    """Delete one character (not first/last)."""
    if len(sld) < 4:
        return None
    idx = rng.randint(1, len(sld) - 2)
    return sld[:idx] + sld[idx + 1 :]

def _mutate_sld_insertion(sld: str, rng: random.Random) -> Optional[str]:
    """Insert a random letter (typo) next to a random position."""
    if len(sld) < 2:
        return None
    idx = rng.randint(1, len(sld) - 1)
    ch = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return sld[:idx] + ch + sld[idx:]

def _mutate_sld_transposition(sld: str, rng: random.Random) -> Optional[str]:
    """Swap two adjacent characters (not first/last)."""
    if len(sld) < 4:
        return None
    idx = rng.randint(1, len(sld) - 2)
    return sld[:idx] + sld[idx + 1] + sld[idx] + sld[idx + 2 :]

def _mutate_sld_keyboard_neighbor(sld: str, rng: random.Random) -> Optional[str]:
    """Replace one char with a keyboard neighbor (not first/last)."""
    if len(sld) < 3:
        return None
    candidates = [i for i in range(1, len(sld) - 1) if sld[i] in _KEYBOARD_NEIGHBORS]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    repl = rng.choice(_KEYBOARD_NEIGHBORS[sld[idx]])
    return sld[:idx] + repl + sld[idx + 1 :]

def _mutate_sld_repetition(sld: str, rng: random.Random) -> Optional[str]:
    """Duplicate one character (not first/last)."""
    if len(sld) < 3:
        return None
    idx = rng.randint(1, len(sld) - 1)
    return sld[: idx + 1] + sld[idx] + sld[idx + 1 :]

def _mutate_sld_omission(sld: str, rng: random.Random) -> Optional[str]:
    """Same as deletion: remove one char (not first/last)."""
    return _mutate_sld_deletion(sld, rng)

def _synthetic_typosquat_url(url: str, substitutions: int = 1, rng: Optional[random.Random] = None) -> str:
    """
    Realistic typosquatting generator. Mutates only host (prioritizes SLD).
    Applies 1 or rarely 2 mutations: homoglyph, deletion, insertion, transposition,
    keyboard neighbor, repetition, omission. Keeps scheme, path, query, fragment, TLD unchanged.
    """
    rng = rng or random.Random()
    u = (url or "").strip()
    if not u:
        return u
    parsed = urlsplit(u if "://" in u else "http://" + u)
    host = (parsed.hostname or "").lower()
    if not host or "." not in host:
        return u
    parts = [p for p in host.split(".") if p]
    if len(parts) < 2:
        return u
    tld = parts[-1]
    sld = parts[-2]
    prefix = parts[:-2]  # subdomains if any
    mutations = [
        _mutate_sld_homoglyph,
        _mutate_sld_deletion,
        _mutate_sld_insertion,
        _mutate_sld_transposition,
        _mutate_sld_keyboard_neighbor,
        _mutate_sld_repetition,
        _mutate_sld_omission,
    ]
    rng.shuffle(mutations)
    new_sld = None
    for mut in mutations:
        new_sld = mut(sld, rng)
        if new_sld and new_sld != sld and len(new_sld) >= 2:
            break
    if not new_sld or new_sld == sld:
        # Fallback: homoglyph on full host if SLD had no candidate
        host_list = list(host)
        candidates = [i for i, c in enumerate(host_list) if c in _LETTER_TO_HOMOGLYPH]
        if candidates:
            idx = rng.choice(candidates)
            host_list[idx] = rng.choice(_LETTER_TO_HOMOGLYPH[host_list[idx]])
            new_host = "".join(host_list)
        else:
            return u
    else:
        new_parts = prefix + [new_sld] + [tld]
        new_host = ".".join(new_parts)
    if not new_host or "." not in new_host:
        return u
    # If input was bare domain (no scheme), return only the mutated host so normal/fake have same format.
    if "://" not in u:
        return new_host
    scheme = parsed.scheme or "http"
    path = parsed.path or ""
    query = "?" + parsed.query if parsed.query else ""
    fragment = "#" + parsed.fragment if parsed.fragment else ""
    return f"{scheme}://{new_host}{path}{query}{fragment}"

# ============================================================
# 9. Feature schema / labels (feature 정의 및 라벨)
# ============================================================

FEATURE_NAMES: List[str] = [
    "host_len",
    "has_multi_level_tld",
    "has_cdn_keyword",
    "has_api_keyword",
    "is_short_domain",
    "has_hyphen",
    "num_hyphens",
    "num_underscores",
    "num_at",
    "has_ip_host",
    "num_subdomains",
    # Typosquatting-specific (domain-level)
    "domain_digit_ratio",
    "domain_homoglyph_ratio",
    "domain_vowel_like_digit_count",
    "domain_letter_digit_alternations",
    # SLD vs brand (typosquatting similarity)
    "sld_damerau_levenshtein_closest_brand",
    "sld_normalized_edit_distance",
    "sld_3gram_jaccard_closest_brand",
    "sld_repeated_char_count",
    "sld_missing_char_score",
    "sld_adjacent_transposition_indicator",
    "sld_keyboard_neighbor_substitution_count",
    "sld_homoglyph_reverse_score",
    "has_country_code_tld",
    "has_safe_second_level_hint",
    "path_depth",
    "has_do_or_html_endpoint",
    "has_redirect_pattern",
    "host_length",
    "host_contains_brand_token",
    "brand_token_in_subdomain",
    "brand_plus_keyword_pattern",
    "brand_hyphen_compound",
    "brand_target_action_pattern",
    "domain_age_days",
    "domain_age_log_days",
    "domain_age_missing",
    "domain_is_very_new",
    "domain_is_new",
    "domain_is_established",
    "domain_is_old",
    "ssl_valid_days",
    "ssl_remaining_days",
    "ssl_age_days",
    "ssl_missing",
    "ssl_is_short_lived",
    "ssl_is_normal_lived",
    "ssl_is_long_lived",
    "ssl_expires_very_soon",
    "ssl_expires_soon",
    "ssl_expires_far",
    "ssl_is_very_new",
    "ssl_is_recent",
    "ssl_is_mature",
]

_FEATURE_LABELS_KO: Dict[str, str] = {
    "sld_normalized_edit_distance": "정규화된 편집 거리",
    "sld_damerau_levenshtein_closest_brand": "Damerau-Levenshtein 거리, 브랜드 기준",
    "sld_adjacent_transposition_indicator": "인접 문자 위치 변경 여부",
    "domain_letter_digit_alternations": "문자-숫자 교차 패턴",
    "domain_digit_ratio": "숫자 비율",
    "sld_3gram_jaccard_closest_brand": "3-gram Jaccard 유사도",
    "num_subdomains": "서브도메인 개수",
    "num_hyphens": "하이픈 개수",
    "has_country_code_tld": "국가 코드 TLD 포함 여부",
    "sld_keyboard_neighbor_substitution_count": "키보드 인접 치환 횟수",
    "has_hyphen": "하이픈 포함 여부",
    "domain_homoglyph_ratio": "유사 문자 비율",
    "has_safe_second_level_hint": "안전한 SLD 힌트 포함 여부",
    "sld_repeated_char_count": "반복 문자 개수",
    "sld_missing_char_score": "문자 누락 점수",
    "is_short_domain": "짧은 도메인 여부",
    "host_len": "호스트 길이",
    "has_multi_level_tld": "다중 TLD 여부",
    "has_cdn_keyword": "CDN 키워드 포함 여부",
    "path_depth": "경로 깊이",
    "has_do_or_html_endpoint": "서비스 엔드포인트 경로 포함 여부",
    "has_redirect_pattern": "리디렉션 파라미터 패턴 포함 여부",
    "host_length": "호스트 길이(스케일)",
    "host_contains_brand_token": "호스트 내 브랜드 토큰 포함 여부",
    "brand_token_in_subdomain": "서브도메인 내 브랜드 토큰 포함 여부",
    "brand_plus_keyword_pattern": "브랜드+의심 키워드 조합 여부",
    "brand_hyphen_compound": "브랜드 하이픈 결합형 여부",
    "brand_target_action_pattern": "브랜드+대상+행위 위장 패턴",
    "domain_age_days": "도메인 등록 후 경과 일수",
    "domain_age_log_days": "도메인 나이 로그 변환값",
    "domain_age_missing": "도메인 나이 조회 실패 여부",
    "ssl_valid_days": "SSL 인증서 전체 유효 일수",
    "ssl_remaining_days": "SSL 인증서 남은 유효 일수",
    "ssl_age_days": "SSL 인증서 발급 후 경과 일수",
    "ssl_missing": "SSL 인증서 조회 실패 여부",
}

_METRIC_LABELS_KO: Dict[str, str] = {
    "accuracy": "정확도",
    "roc_auc": "ROC AUC, 분류 성능 지표",
    "precision": "정밀도, 오탐 억제",
    "recall": "재현율, 미탐 억제",
    "f1": "F1 점수, 균형 지표",
    "threshold": "판정 임계값",
    "confusion_matrix": "혼동 행렬",
}

_CLASSIFICATION_REPORT_HEADER = "precision    recall  f1-score   support"

_CLASSIFICATION_REPORT_HEADER_BILINGUAL = (
    "precision(정밀도)    recall(재현율)  f1-score(F1 점수)   support(샘플 수)"
)

def _annotate_feature_name(name: str) -> str:
    label_ko = _FEATURE_LABELS_KO.get(name)
    return f"{name}({label_ko})" if label_ko else name

def _annotate_metric_name(name: str) -> str:
    label_ko = _METRIC_LABELS_KO.get(name)
    return f"{name}({label_ko})" if label_ko else name

def _format_classification_report_for_print(report: str) -> str:
    if not report:
        return report
    return report.replace(_CLASSIFICATION_REPORT_HEADER, _CLASSIFICATION_REPORT_HEADER_BILINGUAL, 1)

def _feature_dict_to_array(feature_values: Dict[str, float]) -> np.ndarray:
    return np.array(
        [float(feature_values.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float32,
    )

def _domain_only_feature_dict_to_array(feature_values: Dict[str, float]) -> np.ndarray:
    padded_features = {name: 0.0 for name in FEATURE_NAMES}
    for name, value in feature_values.items():
        if name in padded_features:
            padded_features[name] = float(value)
    return _feature_dict_to_array(padded_features)

def _print_metric_summary(metrics: Dict[str, object]) -> None:
    print("[Metrics with Korean explanations]")
    for key in ("accuracy", "roc_auc", "precision", "recall", "f1", "threshold", "confusion_matrix"):
        if key in metrics:
            print(f"  {_annotate_metric_name(key)}: {metrics[key]}")

# ============================================================
# 10. Feature pipeline (최종 feature 벡터 조립)
# ============================================================

def extract_features(
    url: str,
    enable_domain_age: bool = False,
    enable_ssl: bool = False,
    domain_only: bool = False,
) -> np.ndarray:
    """
    Extract lexical features from a URL.
    Returns:
        np.ndarray shape (len(FEATURE_NAMES),)
    """
    u = (url or "").strip()
    if domain_only:
        features = extract_domain_only_features(u, enable_domain_age, enable_ssl=enable_ssl)
        return _domain_only_feature_dict_to_array(features)

    # urlsplit requires scheme to parse netloc well. If missing, prepend.
    parsed = urlsplit(u if "://" in u else "http://" + u)
    host = _safe_lower(parsed.hostname or "")
    if host.startswith("www."):
        host = host[4:]
    path = _safe_lower(parsed.path or "")
    url_lower = _safe_lower(u)
    multi_tld_flag = has_multi_level_tld(host)
    country_code_tld_flag = has_country_code_tld(host)
    safe_second_level_hint_flag = has_safe_second_level_hint(host)
    length_feats = extract_length_features(url)
    domain_age_feats = get_domain_age_features_for_mode(url, enable_domain_age)
    ssl_feats = get_ssl_features_for_mode(url, enable_ssl)
    domain_age_bucket_feats = _bucketize_domain_age_features(
        float(domain_age_feats["domain_age_days"]),
        float(domain_age_feats["domain_age_missing"]),
    )
    ssl_bucket_feats = _bucketize_ssl_features(
        float(ssl_feats["ssl_valid_days"]),
        float(ssl_feats["ssl_remaining_days"]),
        float(ssl_feats["ssl_age_days"]),
        float(ssl_feats["ssl_missing"]),
    )

    host_len = len(host)

    num_hyphens = _count_substring(u, "-")
    num_underscores = _count_substring(u, "_")
    num_at = _count_substring(u, "@")

    has_ip_host = 1.0 if (_RE_IP.match(host) is not None) else 0.0

    # subdomains: a.b.c.tld -> subdomains = len(parts)-2 (exclude domain + tld)
    if not host or _RE_IP.match(host):
        num_subdomains = 0.0
    else:
        parts = [p for p in host.split(".") if p]
        num_subdomains = float(max(0, len(parts) - 2))

    # Normal infrastructure hints: CDN/static asset related keywords in the URL.
    has_cdn_keyword = 1.0 if any(k in url_lower for k in ("cdn", "static", "img", "js", "asset")) else 0.0

    # Normal service hints: API/gateway style keywords often seen in benign service URLs.
    has_api_keyword = 1.0 if any(k in url_lower for k in ("api", "gateway", "gw")) else 0.0

    # Short brand-like domains can be normal; use SLD length when available.
    short_domain_target = sld if (sld := _get_sld(host)) else host
    is_short_domain = 1.0 if 0 < len(short_domain_target) < 5 else 0.0

    # Binary hyphen indicator, separate from raw hyphen count.
    has_hyphen = 1.0 if "-" in host else 0.0

    path_depth = float(len([segment for segment in path.split("/") if segment]))
    has_do_or_html_endpoint = 1.0 if (".do" in path or ".html" in path) else 0.0
    has_redirect_pattern = 1.0 if any(
        pattern in url_lower for pattern in ("redirect=", "returnurl=", "continue=", "url=")
    ) else 0.0

    brand_tokens_in_host = _get_brand_tokens_in_host(host)
    host_parts = _split_host_labels(host)
    simple_brand_host_label_count = 3 if multi_tld_flag else 2
    is_simple_brand_host = any(
        len(host_parts) == simple_brand_host_label_count and host_parts[0] == brand
        for brand in brand_tokens_in_host
    )
    host_contains_brand_token = (
        1.0 if brand_tokens_in_host and not is_simple_brand_host else 0.0
    )
    brand_token_in_subdomain = _brand_token_in_subdomain(host)
    brand_plus_keyword_pattern = _brand_plus_keyword_pattern(host)
    brand_hyphen_compound = _brand_hyphen_compound(host)
    brand_target_action_pattern = _brand_target_action_pattern(host)

    # Typosquatting-specific features (domain / host only)
    domain_digit_ratio = _domain_digit_letter_ratio(host)
    homoglyph_count = _domain_homoglyph_count(host)
    domain_homoglyph_ratio = float(homoglyph_count) / max(1, host_len)
    vowel_like_count = _domain_vowel_like_digit_count(host)
    domain_letter_digit_alternations = float(_domain_letter_digit_alternations(host))

    (sld_dl, sld_norm_edit, sld_jaccard3, sld_repeated, sld_missing,
     sld_adj_trans, sld_kbd_count, sld_homoglyph_rev) = _sld_vs_brand_features(sld)

    feats = np.array(
        [
            float(host_len),
            float(multi_tld_flag),
            float(has_cdn_keyword),
            float(has_api_keyword),
            float(is_short_domain),
            float(has_hyphen),
            float(num_hyphens),
            float(num_underscores),
            float(num_at),
            float(has_ip_host),
            float(num_subdomains),
            float(domain_digit_ratio),
            float(domain_homoglyph_ratio),
            float(vowel_like_count),
            float(domain_letter_digit_alternations),
            float(sld_dl),
            float(sld_norm_edit),
            float(sld_jaccard3),
            float(sld_repeated),
            float(sld_missing),
            float(sld_adj_trans),
            float(sld_kbd_count),
            float(sld_homoglyph_rev),
            float(country_code_tld_flag),
            float(safe_second_level_hint_flag),
            float(path_depth),
            float(has_do_or_html_endpoint),
            float(has_redirect_pattern),
            float(length_feats["host_length"]),
            float(host_contains_brand_token),
            float(brand_token_in_subdomain),
            float(brand_plus_keyword_pattern),
            float(brand_hyphen_compound),
            float(brand_target_action_pattern),
            float(domain_age_feats["domain_age_days"]),
            float(domain_age_feats["domain_age_log_days"]),
            float(domain_age_feats["domain_age_missing"]),
            float(domain_age_bucket_feats["domain_is_very_new"]),
            float(domain_age_bucket_feats["domain_is_new"]),
            float(domain_age_bucket_feats["domain_is_established"]),
            float(domain_age_bucket_feats["domain_is_old"]),
            float(ssl_feats["ssl_valid_days"]),
            float(ssl_feats["ssl_remaining_days"]),
            float(ssl_feats["ssl_age_days"]),
            float(ssl_feats["ssl_missing"]),
            float(ssl_bucket_feats["ssl_is_short_lived"]),
            float(ssl_bucket_feats["ssl_is_normal_lived"]),
            float(ssl_bucket_feats["ssl_is_long_lived"]),
            float(ssl_bucket_feats["ssl_expires_very_soon"]),
            float(ssl_bucket_feats["ssl_expires_soon"]),
            float(ssl_bucket_feats["ssl_expires_far"]),
            float(ssl_bucket_feats["ssl_is_very_new"]),
            float(ssl_bucket_feats["ssl_is_recent"]),
            float(ssl_bucket_feats["ssl_is_mature"]),
        ],
        dtype=np.float32,
    )
    return feats

def featurize_urls(
    urls: Iterable[str],
    enable_domain_age: bool = False,
    enable_ssl: bool = False,
    domain_only: bool = False,
) -> np.ndarray:
    urls_list = list(urls)
    X = np.vstack(
        [
            extract_features(
                u,
                enable_domain_age=enable_domain_age,
                enable_ssl=enable_ssl,
                domain_only=domain_only,
            )
            for u in urls_list
        ]
    ) if urls_list else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return X

# ============================================================
# 11. Data split logic (데이터 분할 로직)
# ============================================================

def _group_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float = 0.3,
    random_state: int = 42,
    sample_weight: Optional[np.ndarray] = None,
    return_indices: bool = False,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
]:
    """
    Split so that all samples in the same group stay in the same split (no leak).
    Returns (X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test).
    If return_indices is True, returns (..., train_idx, val_idx, test_idx) as extra tuple.
    w_* are None if sample_weight is None.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, rest_idx = next(gss.split(X, y, groups))
    X_rest = X[rest_idx]
    y_rest = y[rest_idx]
    groups_rest = groups[rest_idx]
    w_rest = sample_weight[rest_idx] if sample_weight is not None else None
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_idx_local, test_idx_local = next(gss2.split(X_rest, y_rest, groups_rest))
    val_idx = rest_idx[val_idx_local]
    test_idx = rest_idx[test_idx_local]
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    if sample_weight is not None:
        w_train = sample_weight[train_idx]
        w_val = sample_weight[val_idx]
        w_test = sample_weight[test_idx]
        out = (X_train, X_val, X_test, y_train, y_val, y_test, w_train, w_val, w_test)
    else:
        out = (X_train, X_val, X_test, y_train, y_val, y_test, None, None, None)
    if return_indices:
        return out + (train_idx, val_idx, test_idx)
    return out

# ============================================================
# 12. Model bundle / train / eval / predict (모델 저장/학습/평가/추론)
# ============================================================

@dataclass
class ModelBundle:
    model: object
    feature_names: List[str]
    meta: Dict[str, object]

def _require_deps() -> None:
    missing: List[str] = []
    if XGBClassifier is None:
        missing.append(f"xgboost ({_XGBOOST_IMPORT_ERROR})")  # type: ignore[name-defined]
    if joblib is None:
        missing.append(f"joblib ({_JOBLIB_IMPORT_ERROR})")  # type: ignore[name-defined]
    if missing:
        raise RuntimeError(
            "Missing dependencies: "
            + ", ".join(missing)
            + "\nInstall: pip install xgboost scikit-learn numpy joblib"
        )

def train_xgboost_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> object:
    _require_deps()

    # Tuned for typosquatting: lower learning_rate and more trees for stable learning
    # with extra typosquatting features; avoids overfitting when sample_weight is used.
    model = XGBClassifier(  # type: ignore[call-arg]
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.1,
        min_child_weight=2.0,
        gamma=0.1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=max(1, os.cpu_count() or 1),
    )

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    model.fit(**fit_kwargs)
    return model

def _print_feature_importance(model: object, feature_names: Optional[List[str]] = None) -> None:
    """Print feature importance ranking (gain)."""
    names = feature_names or FEATURE_NAMES
    try:
        booster = model.get_booster()
        score = booster.get_score(importance_type="gain")
        if not score:
            return
        # score keys may be f0, f1, ... or feature names depending on xgb version
        idx_to_name = {i: names[i] for i in range(len(names))}
        by_gain = sorted(
            [(idx_to_name.get(int(k.replace("f", "")), k), v) for k, v in score.items()],
            key=lambda x: -x[1],
        )
        print("\n[Feature importance (gain, 특징 중요도)]")
        for name, gain in by_gain[:20]:
            print(f"  {_annotate_feature_name(name)}: {gain:.2f}")
    except Exception:
        pass

def _best_threshold_f1(y_true: np.ndarray, proba: np.ndarray, thresholds: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Find threshold that maximizes F1 on validation. Returns (best_threshold, best_f1)."""
    if thresholds is None:
        thresholds = np.linspace(0.2, 0.8, 31)
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        pred = (proba >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

def evaluate_binary_classifier(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
    optimize_threshold: bool = False,
) -> Dict[str, object]:
    proba = predict_proba(model, X)
    if optimize_threshold:
        threshold, _ = _best_threshold_f1(y, proba)
    pred = (proba >= threshold).astype(int)

    acc = float(accuracy_score(y, pred))
    try:
        auc = float(roc_auc_score(y, proba))
    except Exception:
        auc = float("nan")
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)

    return {
        "accuracy": acc,
        "roc_auc": auc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "report": classification_report(y, pred, digits=4, zero_division=0),
    }

def evaluate_pair_accuracy(proba_normal: np.ndarray, proba_fake: np.ndarray) -> float:
    """Pair-level accuracy: fraction of pairs where proba_fake > proba_normal."""
    if len(proba_normal) != len(proba_fake):
        return 0.0
    correct = np.sum(proba_fake > proba_normal)
    return float(correct) / len(proba_normal)

def evaluate_brand_held_out(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_urls: List[str],
    train_urls: List[str],
) -> Tuple[Optional[Dict[str, float]], int]:
    """
    Evaluate only on test samples whose closest brand was not seen in training.
    Returns (metrics_dict, n_held_out). metrics_dict is None if n_held_out == 0.
    """
    train_brands = set()
    for u in train_urls:
        u = (u or "").strip()
        if not u or "://" not in u:
            u = "http://" + u
        host = (urlsplit(u).hostname or "").lower()
        sld = _get_sld(host)
        b = _closest_brand(sld)
        if b:
            train_brands.add(b)
    held_out_idx = []
    for i, u in enumerate(test_urls):
        if i >= len(y_test):
            break
        u = (u or "").strip()
        if not u or "://" not in u:
            u = "http://" + u
        host = (urlsplit(u).hostname or "").lower()
        sld = _get_sld(host)
        b = _closest_brand(sld)
        if b and b not in train_brands:
            held_out_idx.append(i)
    if not held_out_idx:
        return None, 0
    idx = np.array(held_out_idx)
    X_ho = X_test[idx]
    y_ho = y_test[idx]
    if X_ho.shape[0] != len(y_ho):
        return None, 0
    # Slice features if model was trained with fewer
    n_feat = getattr(model, "n_features_in_", None) or X_test.shape[1]
    if X_ho.shape[1] > n_feat:
        X_ho = X_ho[:, :n_feat]
    metrics = evaluate_binary_classifier(model, X_ho, y_ho)
    return metrics, len(held_out_idx)

def predict_proba(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return np.asarray(proba, dtype=np.float32)
    if hasattr(model, "predict"):
        pred = model.predict(X)
        return np.asarray(pred, dtype=np.float32)
    raise TypeError("Model does not support predict_proba/predict.")

def save_bundle(bundle: ModelBundle, path: str) -> None:
    _require_deps()
    payload = {
        "model": bundle.model,
        "feature_names": bundle.feature_names,
        "meta": bundle.meta,
    }
    joblib.dump(payload, path)  # type: ignore[union-attr]

def load_bundle(path: str) -> ModelBundle:
    _require_deps()
    payload = joblib.load(path)  # type: ignore[union-attr]
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError("Invalid bundle format.")
    return ModelBundle(
        model=payload["model"],
        feature_names=list(payload.get("feature_names", FEATURE_NAMES)),
        meta=dict(payload.get("meta", {})),
    )

def predict_url(
    bundle: ModelBundle,
    url: str,
    enable_domain_age: Optional[bool] = None,
    enable_ssl: Optional[bool] = None,
    domain_only: Optional[bool] = None,
) -> Tuple[int, float, Dict[str, float]]:
    if enable_domain_age is None:
        enable_domain_age = bool(bundle.meta.get("enable_domain_age", False))
    if enable_ssl is None:
        enable_ssl = bool(bundle.meta.get("enable_ssl", False))
    if domain_only is None:
        domain_only = bool(bundle.meta.get("domain_only", False))
    X = featurize_urls(
        [url],
        enable_domain_age=enable_domain_age,
        enable_ssl=enable_ssl,
        domain_only=domain_only,
    )
    feats = extract_features(
        url,
        enable_domain_age=enable_domain_age,
        enable_ssl=enable_ssl,
        domain_only=domain_only,
    )
    base_n = len(bundle.feature_names)
    if X.shape[1] > base_n:
        X = X[:, :base_n]
        feats = feats[:base_n]
    proba = float(predict_proba(bundle.model, X)[0])
    label = 1 if proba >= 0.5 else 0
    feat_map = {name: float(val) for name, val in zip(bundle.feature_names, feats)}
    domain_age_meta = get_domain_age_features_for_mode(url, bool(enable_domain_age))
    for key in (
        "rdap_status_ok",
        "rdap_status_not_registered",
        "rdap_status_lookup_failed",
        "rdap_status_parse_failed",
    ):
        if key in domain_age_meta:
            feat_map[key] = float(domain_age_meta[key])
    return label, proba, feat_map

def predict_url_dom(
    bundle: ModelBundle,
    url: str,
) -> Tuple[int, float, Dict[str, float]]:
    dom_features = extract_dom_features(url)
    dom_feature_array = _dom_feature_dict_to_array(dom_features, feature_names=bundle.feature_names)
    X = dom_feature_array.reshape(1, -1)
    proba = float(predict_proba(bundle.model, X)[0])
    label = 1 if proba >= 0.5 else 0
    dom_feature_map = {
        "dom_max_depth": float(dom_features.get("dom_max_depth", 0.0)),
        "dead_link_ratio": float(dom_features.get("dead_link_ratio", 0.0)),
        "hidden_tags_count": float(dom_features.get("hidden_tags_count", 0.0)),
        "suspicious_form_action": float(dom_features.get("suspicious_form_action", 0.0)),
        "dom_fetch_failed": float(dom_features.get("dom_fetch_failed", 0.0)),
    }
    return label, proba, dom_feature_map

# --- Typosquatting explanation helpers (설명 전용; URL·피처 값은 변경하지 않음) ---

_HOMOGLYPH_DIGIT_TO_LETTERS: Dict[str, Tuple[str, ...]] = {
    "0": ("o",),
    "1": ("l", "i"),
    "!": ("i",),
    "3": ("e",),
    "4": ("a",),
    "@": ("a",),
    "5": ("s",),
    "$": ("s",),
    "6": ("g",),
    "8": ("b",),
    "9": ("g",),
}


def _homoglyph_pair_from_positions(brand_char: str, sld_char: str) -> Optional[Tuple[str, str]]:
    if brand_char == sld_char:
        return None
    letters = _HOMOGLYPH_DIGIT_TO_LETTERS.get(sld_char)
    if letters and brand_char in letters:
        return (brand_char, sld_char)
    return None


def _extract_homoglyph_changes(sld: str, brand: str) -> List[Tuple[str, str]]:
    """Align SLD to brand (equal length); return unique (원래 문자, 조작 문자) pairs."""
    sld_n = (sld or "").lower()
    brand_n = (brand or "").lower()
    if not sld_n or not brand_n or len(sld_n) != len(brand_n):
        return []
    out: List[Tuple[str, str]] = []
    for i in range(len(sld_n)):
        pair = _homoglyph_pair_from_positions(brand_n[i], sld_n[i])
        if pair:
            out.append(pair)
    # dedupe preserving stable order
    seen: Set[Tuple[str, str]] = set()
    deduped: List[Tuple[str, str]] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _sld_homoglyph_indices(sld: str, brand: str) -> List[int]:
    sld_n = (sld or "").lower()
    brand_n = (brand or "").lower()
    if not sld_n or not brand_n or len(sld_n) != len(brand_n):
        return []
    idxs: List[int] = []
    for i in range(len(sld_n)):
        if _homoglyph_pair_from_positions(brand_n[i], sld_n[i]):
            idxs.append(i)
    return idxs


def _find_sld_span_in_url(url: str, host: str, sld: str) -> Optional[Tuple[int, int]]:
    """Return [start, end) indices of the SLD substring inside the original url string."""
    raw = url or ""
    h = (host or "").lower()
    slow = (sld or "").lower()
    if not slow:
        return None
    ulow = raw.lower()
    if h:
        hpos = ulow.find(h)
        if hpos >= 0:
            seg = ulow[hpos : hpos + len(h)]
            off = seg.find(slow)
            if off >= 0:
                start = hpos + off
                return start, start + len(slow)
    pos = ulow.find(slow)
    if pos >= 0:
        return pos, pos + len(slow)
    return None


def _highlight_homoglyph_chars_in_url(url: str, chars: Set[int]) -> str:
    """설명용: chars는 원본 URL 문자열 기준으로 작은따옴표로 감쌀 인덱스 집합."""
    if not chars:
        return url
    s = url
    for i in sorted(chars, reverse=True):
        if 0 <= i < len(s):
            s = s[:i] + "'" + s[i] + "'" + s[i + 1 :]
    return s


def _format_homoglyph_arrow_list(pairs: List[Tuple[str, str]]) -> str:
    uniq = sorted(set(pairs), key=lambda x: (x[0], x[1]))
    return ", ".join(f"{a} → {b}" for a, b in uniq)


def _get_suspicious_keywords_in_host(host: str) -> List[str]:
    h = (host or "").lower()
    found: List[str] = []
    for kw in _SUSPICIOUS_BRAND_KEYWORDS:
        if kw and kw in h:
            found.append(kw)
    return found


def build_typo_explanations(url: str, feat_map: Dict[str, float], probability: float) -> List[str]:
    reasons: List[str] = []
    added: Set[str] = set()

    def _add(text: str) -> None:
        if text not in added:
            reasons.append(text)
            added.add(text)

    host, _path, sld = _split_url_for_analysis(url)
    if not sld or not BRAND_DICTIONARY:
        return reasons

    brand = _closest_brand(sld)
    lev = _levenshtein(sld, brand) if brand else 999

    has_brand_similarity = (
        float(feat_map.get("host_contains_brand_token", 0.0)) >= 1.0
        or float(feat_map.get("brand_token_in_subdomain", 0.0)) >= 1.0
        or float(feat_map.get("sld_3gram_jaccard_closest_brand", 0.0)) >= 0.6
        or float(feat_map.get("sld_damerau_levenshtein_closest_brand", 999.0)) <= 2.0
        or float(feat_map.get("sld_normalized_edit_distance", 1.0)) <= 0.35
    )

    typo_context = (
        has_brand_similarity
        or probability >= 0.5
        or float(feat_map.get("domain_homoglyph_ratio", 0.0)) >= 0.05
        or float(feat_map.get("domain_vowel_like_digit_count", 0.0)) >= 1.0
        or float(feat_map.get("brand_hyphen_compound", 0.0)) >= 1.0
        or float(feat_map.get("brand_target_action_pattern", 0.0)) >= 1.0
        or float(feat_map.get("brand_plus_keyword_pattern", 0.0)) >= 1.0
    )
    if not typo_context:
        return reasons

    max_items = 3
    brand_keyword_impersonation = float(feat_map.get("brand_plus_keyword_pattern", 0.0)) >= 1.0
    homoglyph_feat = float(feat_map.get("domain_homoglyph_ratio", 0.0)) >= 0.05 or float(
        feat_map.get("domain_vowel_like_digit_count", 0.0)
    ) >= 1.0
    hyphen_feat = float(feat_map.get("brand_hyphen_compound", 0.0)) >= 1.0
    subdomain_brand_feat = float(feat_map.get("brand_token_in_subdomain", 0.0)) >= 1.0
    target_action_feat = float(feat_map.get("brand_target_action_pattern", 0.0)) >= 1.0

    pairs = _extract_homoglyph_changes(sld, brand) if brand else []
    sld_idx = _sld_homoglyph_indices(sld, brand) if brand else []
    span = _find_sld_span_in_url(url, host, sld) if sld_idx else None
    wrap_idx: Set[int] = set()
    if span and sld_idx:
        base = span[0]
        for j in sld_idx:
            wrap_idx.add(base + j)

    shown_impersonation = False

    # 1 — 브랜드 유사도 / 위장
    if brand and brand_keyword_impersonation and brand in sld.lower():
        _add(
            f"이 주소에는 {brand} 브랜드명이 포함되어 있어 정상 서비스처럼 보일 수 있습니다.\n"
            "  하지만 실제 공식 도메인이 아닌 곳에 브랜드명을 넣어 사용자를 속이는 방식일 수 있습니다."
        )
        shown_impersonation = True
    elif (
        brand
        and 1 <= lev <= 2
        and float(feat_map.get("sld_damerau_levenshtein_closest_brand", 999.0)) <= 2.0
        and (has_brand_similarity or probability >= 0.5)
    ):
        _add(
            f"이 주소는 {brand}와 매우 비슷하게 만들어져 사용자가 공식 사이트로 착각할 수 있습니다. ({lev}글자 차이)\n"
            "  실제 피싱 사이트는 유명 브랜드 주소와 거의 비슷한 철자를 사용해 접속을 유도하는 경우가 많습니다."
        )

    if len(reasons) >= max_items:
        return reasons[:max_items]

    # 2 — 문자 치환 (homoglyph)
    if brand and homoglyph_feat and pairs and wrap_idx:
        hl = _highlight_homoglyph_chars_in_url(url, wrap_idx)
        ch_str = _format_homoglyph_arrow_list(pairs)
        _add(
            f"이 주소는 문자 치환({ch_str})을 이용해 정상 주소처럼 보이도록 구성되어 있습니다. (조작된 문자: {hl})\n"
            "  숫자나 기호를 글자처럼 보이게 바꾸는 방식은 피싱 사이트에서 자주 사용되는 위장 기법입니다."
        )

    if len(reasons) >= max_items:
        return reasons[:max_items]

    # 3 — 피싱 유도 키워드
    kws = _get_suspicious_keywords_in_host(host)
    if (
        kws
        and not shown_impersonation
        and (
            target_action_feat
            or float(feat_map.get("brand_plus_keyword_pattern", 0.0)) >= 1.0
            or (float(feat_map.get("host_contains_brand_token", 0.0)) >= 1.0 and has_brand_similarity)
        )
    ):
        pick = kws[0]
        _add(
            f'이 주소에는 "{pick}"처럼 로그인을 유도하거나 계정 확인을 요구하는 표현이 포함되어 있습니다.\n'
            "  사용자의 계정 정보 입력을 유도하는 피싱 사이트에서 자주 나타나는 표현입니다."
        )

    if len(reasons) >= max_items:
        return reasons[:max_items]

    # 4 — 하이픈
    if hyphen_feat:
        _add(
            "이 주소는 브랜드명과 추가 단어가 하이픈으로 결합되어 공식 주소처럼 보이도록 만들어져 있습니다.\n"
            "  공격자는 하이픈을 이용해 정상 서비스 주소처럼 보이는 긴 도메인을 만드는 경우가 많습니다."
        )

    if len(reasons) >= max_items:
        return reasons[:max_items]

    # 5 — 서브도메인에 브랜드
    if subdomain_brand_feat:
        _add("브랜드명이 서브도메인에 포함되어 공식 서비스처럼 보이게 배치되었습니다.")

    return reasons[:max_items]

def build_url_structure_explanations(url: str, feat_map: Dict[str, float], probability: float) -> List[str]:
    reasons: List[str] = []
    added = set()

    def _add(text: str) -> None:
        if text not in added:
            reasons.append(text)
            added.add(text)

    host_len_raw = float(feat_map.get("host_len", 0.0))
    host_len_scaled = float(feat_map.get("host_length", 0.0))
    host_len = host_len_raw if host_len_raw > 0.0 else host_len_scaled * 50.0

    url_len = float(feat_map.get("url_length", 0.0))
    num_hyphens = float(feat_map.get("num_hyphens", 0.0))
    num_subdomains = float(feat_map.get("num_subdomains", 0.0))

    if host_len > 45.0 or url_len > 100.0:
        _add(
            "이 주소는 일반적인 사이트 주소보다 길어 한눈에 실제 목적지를 확인하기 어렵습니다.\n"
            "  피싱 주소는 사용자가 의심하지 못하도록 긴 경로나 매개변수를 붙여 실제 도메인을 숨기는 경우가 있습니다."
        )
    elif host_len > 0.0 and host_len < 6.0:
        _add(
            "이 주소의 도메인 이름이 매우 짧아 공식 서비스 주소인지 추가 확인이 필요합니다.\n"
            "  짧은 도메인은 정상 서비스에서도 쓰이지만, 악성 주소가 간단한 이름으로 위장할 때도 사용될 수 있습니다."
        )

    if num_hyphens >= 2.0:
        _add(
            "이 주소는 도메인에 하이픈이 여러 번 포함되어 있어 정상 서비스 주소처럼 보이도록 꾸며졌을 수 있습니다.\n"
            "  공격자는 브랜드명과 보안 관련 단어를 하이픈으로 연결해 사용자를 안심시키는 경우가 있습니다."
        )

    if num_subdomains >= 3.0:
        _add(
            "이 주소는 서브도메인이 여러 단계로 구성되어 실제 도메인을 알아보기 어렵습니다.\n"
            "  피싱 주소는 앞부분에 익숙한 단어를 넣고 실제 도메인을 뒤쪽에 숨기는 경우가 있습니다."
        )

    if probability >= 0.5 and not reasons:
        _add(
            "이 주소는 일반적인 정상 서비스 주소와 다른 구조를 가지고 있어 주의가 필요합니다.\n"
            "  주소 구조가 낯설거나 복잡한 경우, 실제 접속 대상이 사용자가 예상한 사이트와 다를 수 있습니다."
        )

    return reasons

def build_domain_explanations(url: str, feat_map: Dict[str, float], probability: float) -> List[str]:
    rdap_status_ok = float(feat_map.get("rdap_status_ok", 0.0))
    rdap_status_not_registered = float(feat_map.get("rdap_status_not_registered", 0.0))
    rdap_status_lookup_failed = float(feat_map.get("rdap_status_lookup_failed", 0.0))
    rdap_status_parse_failed = float(feat_map.get("rdap_status_parse_failed", 0.0))
    domain_age_days = float(feat_map.get("domain_age_days", 0.0))

    # CASE 1: 도메인 자체 미등록 (강한 위험 신호)
    if rdap_status_not_registered >= 1.0:
        return [
            "현재 이 도메인은 정상적으로 등록된 사이트로 확인되지 않았습니다.\n"
            "  잘못된 주소이거나 임시로 생성된 악성 주소일 가능성이 있습니다."
        ]

    # CASE 2: RDAP 조회 실패/파싱 실패 (중립 상태)
    if rdap_status_lookup_failed >= 1.0 or rdap_status_parse_failed >= 1.0:
        return [
            "이 도메인의 등록 정보를 확인하지 못했습니다.\n"
            "  일시적인 조회 실패일 수 있지만, 신뢰할 수 있는 운영 이력을 확인할 수 없으므로 주의가 필요합니다."
        ]

    # CASE 3: 정상 조회 (수치 기반 설명)
    if rdap_status_ok >= 1.0:
        days = max(0, int(domain_age_days))
        if days <= 30:
            return [
                f"이 도메인은 생성된 지 {days}일밖에 되지 않은 신규 도메인입니다.\n"
                "  피싱 사이트는 차단을 피하기 위해 최근 생성된 도메인을 사용하는 경우가 많습니다."
            ]
        if days <= 180:
            return [
                f"이 도메인은 생성된 지 {days}일 된 비교적 새로운 도메인입니다.\n"
                "  운영 이력이 짧은 도메인은 신뢰도가 충분히 검증되지 않았을 수 있습니다."
            ]
        if days <= 365:
            return [
                f"이 도메인은 생성된 지 {days}일 되어 어느 정도 운영 이력이 확인됩니다.\n"
                "  다만 다른 위험 신호가 함께 나타나는 경우에는 추가 확인이 필요합니다."
            ]
        return [
            f"이 도메인은 생성된 지 {days}일 된 오래 운영된 도메인입니다.\n"
            "  도메인 나이만 보면 비교적 안정적인 편이지만, 다른 검증 결과와 함께 판단해야 합니다."
        ]

    # 안전한 기본값: 상태 정보가 불충분한 경우에도 1줄 보장
    return [
        "이 도메인의 등록 정보를 확인하지 못했습니다.\n"
        "  일시적인 조회 실패일 수 있지만, 신뢰할 수 있는 운영 이력을 확인할 수 없으므로 주의가 필요합니다."
    ]

def build_ssl_explanations(url: str, feat_map: Dict[str, float], probability: float) -> List[str]:
    reasons: List[str] = []
    ssl_missing = float(feat_map.get("ssl_missing", 0.0))
    ssl_valid_days = float(feat_map.get("ssl_valid_days", 0.0))
    ssl_remaining_days = float(feat_map.get("ssl_remaining_days", 0.0))
    ssl_age_days = float(feat_map.get("ssl_age_days", 0.0))
    ssl_status_no_cert = float(feat_map.get("ssl_status_no_cert", 0.0))

    # CASE 1/2: 인증서 미발급 vs 조회 실패 구분
    if ssl_missing >= 1.0:
        if ssl_status_no_cert >= 1.0:
            return [
                "이 사이트는 보안 연결(SSL 인증서)이 정상적으로 설정되어 있지 않습니다.\n"
                "  개인정보 입력 시 보안 위험이 발생할 수 있습니다."
            ]
        return [
            "이 사이트의 보안 인증서 정보를 확인하지 못했습니다.\n"
            "  일시적인 네트워크 문제일 수 있지만, 안전한 연결 여부를 확인할 수 없어 주의가 필요합니다."
        ]

    # CASE 3: 정상 조회 시 수치 기반 설명
    # - 기본 수치 문장과 추가 설명이 같은 값을 중복 출력하지 않도록,
    #   추가 설명이 적용되는 경우 해당 문장만 출력하고
    #   추가 설명이 없을 때에만 기본 수치 문장을 출력한다.
    valid_days_i = max(0, int(ssl_valid_days))
    remaining_days_i = max(0, int(ssl_remaining_days))
    age_days_i = max(0, int(ssl_age_days))

    if remaining_days_i <= 7:
        reasons.append(
            f"이 사이트의 보안 인증서가 매우 곧 만료될 예정입니다. (남은 기간: {remaining_days_i}일)\n"
            "  관리가 제대로 되지 않는 사이트이거나 임시로 운영되는 사이트일 가능성이 있습니다."
        )
    elif remaining_days_i <= 30:
        reasons.append(
            f"이 사이트의 보안 인증서 만료일이 가까워지고 있습니다. (남은 기간: {remaining_days_i}일)\n"
            "  인증서 관리 상태가 불안정할 수 있으므로 주의가 필요합니다."
        )
    elif valid_days_i <= 90:
        reasons.append(
            f"이 사이트는 보안 인증서 사용 기간이 매우 짧아 신뢰도가 낮을 수 있습니다. "
            f"(인증서 유효기간: {valid_days_i}일 / 남은 기간: {remaining_days_i}일)\n"
            "  최근 급하게 생성된 피싱 사이트에서 자주 나타나는 패턴입니다."
        )
    elif age_days_i <= 7:
        reasons.append(
            f"이 사이트의 보안 인증서는 최근에 발급되었습니다. (발급 후 경과: {age_days_i}일)\n"
            "  최근 만들어진 사이트에서 자주 나타나는 특징이므로 다른 위험 신호와 함께 확인해야 합니다."
        )
    else:
        reasons.append(
            f"이 사이트의 보안 인증서 상태를 확인했습니다. "
            f"(인증서 유효기간: {valid_days_i}일 / 남은 기간: {remaining_days_i}일)\n"
            "  인증서 기간은 사이트의 보안 설정 상태를 판단하는 참고 자료로 사용됩니다."
        )

    return reasons

def build_dom_explanations(url: str, dom_feature_map: Dict[str, float], probability: float) -> List[str]:
    reasons: List[str] = []

    if float(dom_feature_map.get("dom_fetch_failed", 0.0)) >= 1.0:
        return [
            "이 사이트의 페이지 구조 정보를 가져오지 못했습니다.\n"
            "  일시적인 접속 문제일 수 있지만, 페이지 내부 구조를 확인할 수 없어 주의가 필요합니다."
        ]

    dom_max_depth = float(dom_feature_map.get("dom_max_depth", 0.0))
    dead_link_ratio = float(dom_feature_map.get("dead_link_ratio", 0.0))
    hidden_tags_count = float(dom_feature_map.get("hidden_tags_count", 0.0))
    suspicious_form_action = float(dom_feature_map.get("suspicious_form_action", 0.0))

    depth_i = max(0, int(dom_max_depth))
    ratio_f = max(0.0, dead_link_ratio)
    hidden_i = max(0, int(hidden_tags_count))
    has_external_form = suspicious_form_action >= 1.0

    # 위험 기준을 넘은 항목만 우선순위로 선택 (최대 2개)
    evidence: List[Tuple[int, str]] = []

    if has_external_form:
        evidence.append(
            (
                1,
                "이 사이트는 사용자가 입력한 정보가 외부 도메인으로 전송될 가능성이 있습니다. (외부 폼 전송 감지)\n"
                "  로그인 정보나 개인정보 입력을 요구하는 경우 특히 주의해야 합니다.",
            )
        )

    if hidden_i > 15:
        evidence.append(
            (
                2,
                f"이 사이트에는 사용자에게 보이지 않는 숨겨진 요소가 많이 포함되어 있습니다. (숨겨진 요소: {hidden_i}개)\n"
                "  악성 스크립트, 추적 코드, 위장 입력값을 숨기기 위해 사용되는 경우가 있습니다.",
            )
        )
    elif hidden_i > 5:
        evidence.append(
            (
                2,
                f"이 사이트에는 일부 숨겨진 요소가 포함되어 있습니다. (숨겨진 요소: {hidden_i}개)\n"
                "  정상 사이트에서도 사용될 수 있지만, 다른 위험 신호와 함께 나타나면 주의가 필요합니다.",
            )
        )

    if ratio_f > 20.0:
        evidence.append(
            (
                3,
                f"이 사이트는 실제로 동작하지 않는 링크 비율이 매우 높습니다. (죽은 링크 비율: {ratio_f:.2f}%)\n"
                "  정상 서비스처럼 보이기 위해 화면만 급하게 구성된 피싱 페이지에서 자주 나타나는 특징입니다.",
            )
        )
    elif ratio_f > 5.0:
        evidence.append(
            (
                3,
                f"이 사이트에는 정상적으로 이동되지 않는 링크가 일부 포함되어 있습니다. (죽은 링크 비율: {ratio_f:.2f}%)\n"
                "  사이트 완성도가 낮거나 임시로 구성된 페이지일 가능성이 있습니다.",
            )
        )

    if depth_i > 20:
        evidence.append(
            (
                4,
                f"이 사이트는 페이지 구조가 비정상적으로 깊고 복잡하게 구성되어 있습니다. (DOM 깊이: {depth_i})\n"
                "  악성 스크립트나 위장 화면을 숨기기 위해 구조를 과도하게 늘리는 경우가 있습니다.",
            )
        )
    elif depth_i > 10:
        evidence.append(
            (
                4,
                f"이 사이트는 일반적인 페이지보다 구조가 다소 복잡한 편입니다. (DOM 깊이: {depth_i})\n"
                "  단독으로 악성이라고 보기는 어렵지만, 다른 위험 신호와 함께 확인할 필요가 있습니다.",
            )
        )

    for _, text in sorted(evidence, key=lambda x: x[0])[:2]:
        reasons.append(text)

    return reasons


def _compose_final_explanations(
    url: str,
    typo_feat_map: Dict[str, float],
    typo_probability: float,
    domain_feat_map: Dict[str, float],
    domain_probability: float,
    dom_feature_map: Dict[str, float],
    dom_probability: float,
) -> List[str]:
    """
    URL·도메인·SSL·DOM 신호를 나열하지 않고 짧은 상황 설명 1~2문단으로 합성한다.
    (판단 임계·피처 정의는 build_* 계열과 동일 조건을 유지한다.)
    """
    out: List[str] = []

    def _narrative_typo_clause() -> Optional[Tuple[int, str]]:
        host, _path, sld = _split_url_for_analysis(url)
        if not sld or not BRAND_DICTIONARY:
            return None

        brand = _closest_brand(sld)
        lev = _levenshtein(sld, brand) if brand else 999

        feat_map = typo_feat_map
        has_brand_similarity = (
            float(feat_map.get("host_contains_brand_token", 0.0)) >= 1.0
            or float(feat_map.get("brand_token_in_subdomain", 0.0)) >= 1.0
            or float(feat_map.get("sld_3gram_jaccard_closest_brand", 0.0)) >= 0.6
            or float(feat_map.get("sld_damerau_levenshtein_closest_brand", 999.0)) <= 2.0
            or float(feat_map.get("sld_normalized_edit_distance", 1.0)) <= 0.35
        )

        typo_context = (
            has_brand_similarity
            or typo_probability >= 0.5
            or float(feat_map.get("domain_homoglyph_ratio", 0.0)) >= 0.05
            or float(feat_map.get("domain_vowel_like_digit_count", 0.0)) >= 1.0
            or float(feat_map.get("brand_hyphen_compound", 0.0)) >= 1.0
            or float(feat_map.get("brand_target_action_pattern", 0.0)) >= 1.0
            or float(feat_map.get("brand_plus_keyword_pattern", 0.0)) >= 1.0
        )
        if not typo_context:
            return None

        brand_keyword_impersonation = float(feat_map.get("brand_plus_keyword_pattern", 0.0)) >= 1.0
        homoglyph_feat = float(feat_map.get("domain_homoglyph_ratio", 0.0)) >= 0.05 or float(
            feat_map.get("domain_vowel_like_digit_count", 0.0)
        ) >= 1.0
        hyphen_feat = float(feat_map.get("brand_hyphen_compound", 0.0)) >= 1.0
        subdomain_brand_feat = float(feat_map.get("brand_token_in_subdomain", 0.0)) >= 1.0
        target_action_feat = float(feat_map.get("brand_target_action_pattern", 0.0)) >= 1.0
        pairs = _extract_homoglyph_changes(sld, brand) if brand else []

        shown_impersonation = False
        if brand and brand_keyword_impersonation and brand in sld.lower():
            shown_impersonation = True
            return (10, f"{brand} 이름을 끼워 넣어 공식 사이트처럼 보이게 꾸민 주소")
        if (
            brand
            and 1 <= lev <= 2
            and float(feat_map.get("sld_damerau_levenshtein_closest_brand", 999.0)) <= 2.0
            and (has_brand_similarity or typo_probability >= 0.5)
        ):
            return (11, f"{brand}와 매우 유사한 철자를 쓰는 주소")
        if brand and homoglyph_feat and pairs:
            return (12, "숫자·기호로 글자를 바꿔 정상 주소처럼 보이게 조작한 주소")

        kws = _get_suspicious_keywords_in_host(host)
        if (
            kws
            and not shown_impersonation
            and (
                target_action_feat
                or float(feat_map.get("brand_plus_keyword_pattern", 0.0)) >= 1.0
                or (float(feat_map.get("host_contains_brand_token", 0.0)) >= 1.0 and has_brand_similarity)
            )
        ):
            pick = kws[0]
            return (
                13,
                f'"{pick}" 같은 표현으로 로그인·계정 확인을 유도하는 주소 구성',
            )
        if hyphen_feat:
            return (14, "브랜드와 단어를 하이픈으로 이어 공식 주소처럼 보이게 만든 구조")
        if subdomain_brand_feat:
            return (15, "브랜드명을 서브도메인에 올려 공식 서비스처럼 보이게 배치한 주소")
        return None

    def _narrative_url_structure_clause() -> Optional[Tuple[int, str]]:
        feat_map = typo_feat_map
        added = set()

        def _add(prio: int, text: str) -> Optional[Tuple[int, str]]:
            if text in added:
                return None
            added.add(text)
            return prio, text

        host_len_raw = float(feat_map.get("host_len", 0.0))
        host_len_scaled = float(feat_map.get("host_length", 0.0))
        host_len = host_len_raw if host_len_raw > 0.0 else host_len_scaled * 50.0
        url_len = float(feat_map.get("url_length", 0.0))
        num_hyphens = float(feat_map.get("num_hyphens", 0.0))
        num_subdomains = float(feat_map.get("num_subdomains", 0.0))

        if host_len > 45.0 or url_len > 100.0:
            hit = _add(40, "일반적인 사이트보다 지나치게 길고 복잡한 주소 구조")
            if hit:
                return hit
        if host_len > 0.0 and host_len < 6.0:
            hit = _add(41, "도메인 이름이 지나치게 짧은 형태의 주소")
            if hit:
                return hit
        if num_hyphens >= 2.0:
            hit = _add(42, "하이픈이 여러 번 끼어든 도메인 구조")
            if hit:
                return hit
        if num_subdomains >= 3.0:
            hit = _add(43, "서브도메인이 여러 단계로 겹쳐 실제 도메인을 가리기 어려운 구성")
            if hit:
                return hit
        if typo_probability >= 0.5:
            hit = _add(44, "정상 서비스와는 다르게 어색한 주소 구조")
            if hit:
                return hit
        return None

    def _narrative_registration_ssl_clause() -> Optional[Tuple[int, str]]:
        feat_map = domain_feat_map
        rdap_status_ok = float(feat_map.get("rdap_status_ok", 0.0))
        rdap_status_not_registered = float(feat_map.get("rdap_status_not_registered", 0.0))
        rdap_status_lookup_failed = float(feat_map.get("rdap_status_lookup_failed", 0.0))
        rdap_status_parse_failed = float(feat_map.get("rdap_status_parse_failed", 0.0))
        domain_age_days = float(feat_map.get("domain_age_days", 0.0))

        ssl_missing = float(feat_map.get("ssl_missing", 0.0))
        ssl_valid_days = float(feat_map.get("ssl_valid_days", 0.0))
        ssl_remaining_days = float(feat_map.get("ssl_remaining_days", 0.0))
        ssl_age_days = float(feat_map.get("ssl_age_days", 0.0))
        ssl_status_no_cert = float(feat_map.get("ssl_status_no_cert", 0.0))

        valid_days_i = max(0, int(ssl_valid_days))
        remaining_days_i = max(0, int(ssl_remaining_days))
        age_days_i = max(0, int(ssl_age_days))
        days = max(0, int(domain_age_days))

        ssl_risky = False
        ssl_mid = ""
        if ssl_missing >= 1.0:
            if ssl_status_no_cert >= 1.0:
                return (2, "보안 연결(SSL)이 제대로 갖춰져 있지 않은 접속 환경")
            return (3, "보안 인증서 정보를 확인하지 못한 채 이어지는 접속")

        if remaining_days_i <= 30:
            ssl_risky = True
            if remaining_days_i <= 7:
                ssl_mid = "만료가 임박한 보안 인증서"
            else:
                ssl_mid = "만료가 가까운 보안 인증서"
        elif valid_days_i <= 90:
            ssl_risky = True
            ssl_mid = "유효 기간이 매우 짧은 보안 인증서"
        elif age_days_i <= 7:
            ssl_risky = True
            ssl_mid = "최근에 막 발급된 보안 인증서"

        if rdap_status_not_registered >= 1.0:
            return (1, "정상 등록 도메인으로 확인되지 않은 주소")

        if rdap_status_lookup_failed >= 1.0 or rdap_status_parse_failed >= 1.0:
            if ssl_risky and ssl_mid:
                return (
                    28,
                    f"등록 정보를 확인하지 못한 주소에 {ssl_mid}까지 겹쳐 붙어 있는 모습",
                )
            return (30, "도메인 등록 정보를 확인하지 못한 주소")

        if rdap_status_ok >= 1.0:
            dom_risky = days <= 365
            if not dom_risky and not ssl_risky:
                return None

            if dom_risky and ssl_risky:
                if days <= 30:
                    dom_bits = "생성된 지 얼마 되지 않은 신규 도메인"
                elif days <= 180:
                    dom_bits = f"생성된 지 {days}일밖에 안 된 비교적 새 도메인"
                else:
                    dom_bits = f"운영 이력이 {days}일 수준으로 아직 길지 않은 도메인"
                return (24, f"{dom_bits}에 {ssl_mid}까지 함께 묶인 등록·SSL 패턴")

            if dom_risky:
                if days <= 30:
                    return (31, f"생성된 지 {days}일밖에 안 된 신규 도메인")
                if days <= 180:
                    return (32, f"생성된 지 {days}일밖에 안 된 비교적 새 도메인")
                return (33, f"운영 이력이 {days}일 수준으로 짧은 도메인")

            if ssl_risky and ssl_mid:
                return (35, ssl_mid)

        return None

    def _narrative_dom_paragraph() -> Optional[str]:
        if float(dom_feature_map.get("dom_fetch_failed", 0.0)) >= 1.0:
            return (
                "페이지 구조를 가져오지 못해 내부 위험을 확인하기 어렵습니다. "
                "일시적 장애일 수 있어도 신중히 살펴보는 편이 좋습니다."
            )

        dom_max_depth = float(dom_feature_map.get("dom_max_depth", 0.0))
        dead_link_ratio = float(dom_feature_map.get("dead_link_ratio", 0.0))
        hidden_tags_count = float(dom_feature_map.get("hidden_tags_count", 0.0))
        suspicious_form_action = float(dom_feature_map.get("suspicious_form_action", 0.0))

        depth_i = max(0, int(dom_max_depth))
        ratio_f = max(0.0, dead_link_ratio)
        hidden_i = max(0, int(hidden_tags_count))
        has_external_form = suspicious_form_action >= 1.0

        evidence: List[Tuple[int, str]] = []

        if has_external_form:
            evidence.append((1, "form_external"))

        if hidden_i > 15:
            evidence.append((2, f"hidden_{hidden_i}"))
        elif hidden_i > 5:
            evidence.append((2, f"hidden_{hidden_i}"))

        if ratio_f > 20.0:
            evidence.append((3, f"dead_{ratio_f:.1f}"))
        elif ratio_f > 5.0:
            evidence.append((3, f"dead_{ratio_f:.1f}"))

        if depth_i > 20:
            evidence.append((4, f"depth_{depth_i}"))
        elif depth_i > 10:
            evidence.append((4, f"depth_{depth_i}"))

        if not evidence:
            return None

        def _slot(tag: str) -> str:
            if tag == "form_external":
                return "사용자가 입력한 정보를 외부 도메인으로 전송할 가능성이 있는 화면 구성"
            if tag.startswith("hidden_"):
                n = tag.split("_", 1)[1]
                hi = int(n)
                if hi > 15:
                    return f"사용자에게 잘 보이지 않는 숨겨진 요소가 많이 끼어 있는(DOM {hi}건)"
                return f"숨겨진 요소가 다소 많이 포함된(DOM {hi}건)"
            if tag.startswith("dead_"):
                r = float(tag.split("_", 1)[1])
                if r > 20.0:
                    return f"실제로는 동작하지 않는 링크가 유독 많은(약 {r:.1f}%)"
                return f"정상 이동이 어려운 링크가 섞인(약 {r:.1f}%)"
            if tag.startswith("depth_"):
                d = int(tag.split("_", 1)[1])
                if d > 20:
                    return f"페이지 구조가 비정상적으로 깊고 복잡하게 꼬인(DOM 깊이 {d})"
                return f"일반 페이지보다 구조가 다소 복잡한(DOM 깊이 {d})"
            return tag

        top_ev = sorted(evidence, key=lambda x: x[0])[:2]
        top = [_slot(t) for _, t in top_ev]

        if len(top) == 1:
            return (
                f"이 사이트는 {top[0]} 점이 확인되어 "
                "위장 화면이나 악성 스크립트가 포함되었을 가능성이 있습니다."
            )
        return (
            f"이 사이트는 {top[0]} 점이 이어지고, {top[1]} 형태까지 겹쳐 있어 "
            "위장 화면이나 악성 스크립트가 포함되었을 가능성이 있습니다."
        )

    typo_t = _narrative_typo_clause()
    url_t = _narrative_url_structure_clause()
    reg_t = _narrative_registration_ssl_clause()

    meta_opts: List[Tuple[int, str]] = []
    for t in (typo_t, url_t, reg_t):
        if t is not None:
            meta_opts.append(t)
    meta_opts.sort(key=lambda x: x[0])

    primary: List[str] = []
    for _, clause in meta_opts[:2]:
        c = clause.strip()
        if c and c not in primary:
            primary.append(c)

    dom_text = _narrative_dom_paragraph()

    if not primary and not dom_text:
        return ["뚜렷한 악성 징후가 발견되지 않았습니다."]

    def _meta_paragraph(parts: List[str]) -> str:
        if not parts:
            return ""
        if len(parts) == 1:
            p0 = parts[0]
            if p0.startswith("정상 등록 도메인으로 확인되지 않은"):
                return (
                    "현재 이 도메인은 정상적으로 등록된 사이트로 확인되지 않았습니다. "
                    "잘못된 주소이거나 임시로 만들어진 악성 주소일 가능성이 있습니다."
                )
            if "보안 연결(SSL)이 제대로 갖춰져 있지 않은" in p0:
                return (
                    "이 사이트는 보안 연결이 제대로 갖춰져 있지 않은 환경입니다. "
                    "개인정보 입력 시 유출 위험이 커질 수 있습니다."
                )
            if "보안 인증서 정보를 확인하지 못한 채 이어지는" in p0:
                return (
                    "이 사이트는 안전한 연결 정보를 확인하지 못한 채 접속해야 하는 형태입니다. "
                    "일시적 문제일 수 있어도 추가 확인이 필요합니다."
                )
            if "일반적인 사이트보다 지나치게 길고 복잡한 주소 구조" in p0:
                return (
                    "이 주소는 일반적인 사이트보다 지나치게 길고 복잡한 구조를 사용하고 있어 "
                    "실제 접속 대상을 숨기려는 피싱 주소일 가능성이 있습니다."
                )
            return (
                f"이 사이트는 {p0}에 해당하는 의심 신호가 보이며 "
                "최근 피싱 사이트에서 자주 나타나는 특징과 유사합니다."
            )

        p0, p1 = parts[0], parts[1]
        return (
            f"이 사이트는 {p0}에 해당하는 의심 신호가 이어지는 한편 {p1}까지 겹쳐 있어 "
            "최근 피싱 사이트에서 자주 나타나는 특징과 유사합니다."
        )

    mp = _meta_paragraph(primary)
    if mp:
        out.append(mp)
    if dom_text:
        out.append(dom_text)

    if len(out) > 2:
        out = out[:2]

    return out


def build_all_explanations(
    url: str,
    typo_feat_map: Dict[str, float],
    typo_probability: float,
    domain_feat_map: Dict[str, float],
    domain_probability: float,
    dom_feature_map: Dict[str, float],
    dom_probability: float,
) -> List[str]:
    return _compose_final_explanations(
        url=url,
        typo_feat_map=typo_feat_map,
        typo_probability=typo_probability,
        domain_feat_map=domain_feat_map,
        domain_probability=domain_probability,
        dom_feature_map=dom_feature_map,
        dom_probability=dom_probability,
    )


def _validate_single_input_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        raise ValueError("URL must not be empty or whitespace.")

    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("URL must include a valid scheme and host (example: https://example.com).")

    return value

# ============================================================
# 13. Dataset I/O (데이터셋 입출력)
# ============================================================

def _read_urls_from_file(path: str) -> List[str]:
    urls: List[str] = []

    for encoding in ("utf-8", "cp949"):
        try:
            with open(path, "r", encoding=encoding, newline="") as f:
                sample = f.readline()
                f.seek(0)

                if "," in sample:
                    reader = csv.DictReader(f)
                    fieldnames = [name.strip().lower() for name in (reader.fieldnames or []) if name]

                    if "url" in fieldnames:
                        for row in reader:
                            url = (row.get("url") or row.get("URL") or "").strip()
                            if url:
                                urls.append(url)
                    else:
                        for row in reader:
                            if row:
                                first_value = next(iter(row.values()), "")
                                url = (first_value or "").strip()
                                if url:
                                    urls.append(url)
                else:
                    for line in f:
                        url = line.strip()
                        if url:
                            urls.append(url)

            return urls
        except UnicodeDecodeError:
            urls = []
            continue

    raise UnicodeDecodeError("cp949", b"", 0, 1, "Unable to decode file with utf-8 or cp949.")

def _dedupe_urls_preserve_order(urls: Iterable[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for url in urls:
        value = (url or "").strip()
        if not value or value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped

def _get_optional_extra_normal_path(normal_path: str) -> Optional[str]:
    normal_abs = os.path.abspath(normal_path)
    if os.path.basename(normal_abs).lower() != "top-500.csv":
        return None
    candidate = os.path.join(os.path.dirname(normal_abs), "extra_normal.csv")
    if os.path.abspath(candidate) == normal_abs:
        return None
    return candidate if os.path.isfile(candidate) else None

def _load_extra_normal_urls(
    normal_path: str,
    *,
    existing_urls: Iterable[str] = (),
    excluded_urls: Iterable[str] = (),
) -> Tuple[List[str], Optional[str]]:
    extra_path = _get_optional_extra_normal_path(normal_path)
    if not extra_path:
        return [], None

    existing = {url.strip() for url in existing_urls if url and url.strip()}
    excluded = {url.strip() for url in excluded_urls if url and url.strip()}
    extra_urls = _read_urls_from_file(extra_path)

    filtered: List[str] = []
    seen = set(existing)
    for url in extra_urls:
        value = (url or "").strip()
        if not value or value in seen or value in excluded:
            continue
        filtered.append(value)
        seen.add(value)
    return filtered, extra_path

def _get_optional_long_fake_path(fake_path: str) -> Optional[str]:
    fake_abs = os.path.abspath(fake_path)
    if os.path.basename(fake_abs).lower() != "top-500-fake.csv":
        return None
    candidate = os.path.join(os.path.dirname(fake_abs), "long_fake.csv")
    if os.path.abspath(candidate) == fake_abs:
        return None
    return candidate if os.path.isfile(candidate) else None

def _get_optional_long_fake_path_synthetic(normal_path: str) -> Optional[str]:
    """With --use-synthetic-fake, pick up long_fake.csv next to top-500.csv (same dir as extra_normal)."""
    normal_abs = os.path.abspath(normal_path)
    if os.path.basename(normal_abs).lower() != "top-500.csv":
        return None
    candidate = os.path.join(os.path.dirname(normal_abs), "long_fake.csv")
    if os.path.abspath(candidate) == normal_abs:
        return None
    return candidate if os.path.isfile(candidate) else None

def _load_long_fake_urls(
    long_path: Optional[str],
    *,
    existing_urls: Iterable[str] = (),
    excluded_urls: Iterable[str] = (),
) -> Tuple[List[str], Optional[str]]:
    if not long_path:
        return [], None

    existing = {url.strip() for url in existing_urls if url and url.strip()}
    excluded = {url.strip() for url in excluded_urls if url and url.strip()}
    extra_urls = _read_urls_from_file(long_path)

    filtered: List[str] = []
    seen = set(existing)
    for url in extra_urls:
        value = (url or "").strip()
        if not value or value in seen or value in excluded:
            continue
        filtered.append(value)
        seen.add(value)
    return filtered, long_path

def _load_paired_url_lists(
    normal_path: str,
    fake_path: str,
    *,
    context: str,
) -> Tuple[List[str], List[str], int]:
    """Load aligned normal/fake URL pairs, truncating to the shared prefix if needed."""
    normal_urls = _read_urls_from_file(normal_path)
    fake_urls = _read_urls_from_file(fake_path)

    if not normal_urls:
        raise SystemExit(f"No normal URLs found for {context}.")
    if not fake_urls:
        raise SystemExit(f"No fake URLs found for {context}.")

    n = min(len(normal_urls), len(fake_urls))
    if n == 0:
        raise SystemExit(f"No aligned URL pairs found for {context}.")

    if len(normal_urls) != len(fake_urls):
        print(
            f"Warning: normal has {len(normal_urls)} lines, fake has {len(fake_urls)}. "
            f"Using first {n} aligned pairs for {context}."
        )

    return normal_urls[:n], fake_urls[:n], n
