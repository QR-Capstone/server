"""Shared XGBoost URL feature/model utilities."""


from __future__ import annotations

import csv
import json
import math
import os
import random
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

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

# ============================================================
# 3. URL length / lexical features (URL 길이 및 구조 특징)
# ============================================================

def extract_length_features(url: str) -> Dict[str, float]:
    """Scaled length helpers only."""
    u = (url or "").strip()
    parsed = urlsplit(u if "://" in u else "http://" + u)
    host = _safe_lower(parsed.hostname or "")
    return {
        "url_length": float(len(u)) / 100.0,
        "host_length": float(len(host)) / 50.0,
    }

# ============================================================
# 4. Domain age / RDAP features (도메인 나이)
# ============================================================

_RDAP_LOOKUP_TIMEOUT_SECONDS = 3.0
_DOMAIN_AGE_MAX_DAYS = 36500.0
_DOMAIN_AGE_FALLBACK = {
    "domain_age_days": 0.0,
    "domain_age_log_days": 0.0,
    "domain_age_missing": 1.0,
}
_DOMAIN_AGE_CACHE: Dict[str, Dict[str, float]] = {}

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

def _fetch_rdap_payload(registered_domain: str) -> Optional[Dict[str, Any]]:
    if not registered_domain:
        return None
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
    except (HTTPError, URLError, TimeoutError, ValueError, OSError):
        return None
    try:
        parsed = json.loads(payload.decode(charset, errors="replace"))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None

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
    registered_domain = _get_registered_domain_for_rdap(url)
    if not registered_domain:
        return dict(_DOMAIN_AGE_FALLBACK)

    cached = _DOMAIN_AGE_CACHE.get(registered_domain)
    if cached is not None:
        return dict(cached)

    payload = _fetch_rdap_payload(registered_domain)
    created_at = _extract_rdap_creation_date(payload)
    if created_at is None:
        _DOMAIN_AGE_CACHE[registered_domain] = dict(_DOMAIN_AGE_FALLBACK)
        return dict(_DOMAIN_AGE_FALLBACK)

    domain_age_days = _compute_domain_age_days(created_at)
    features = {
        "domain_age_days": float(domain_age_days),
        "domain_age_log_days": float(math.log1p(domain_age_days)),
        "domain_age_missing": 0.0,
    }
    _DOMAIN_AGE_CACHE[registered_domain] = dict(features)
    return dict(features)

def get_domain_age_features_for_mode(url: str, enable_domain_age: bool) -> Dict[str, float]:
    if not enable_domain_age:
        return {
            "domain_age_days": 0.0,
            "domain_age_log_days": 0.0,
            "domain_age_missing": 1.0,
        }
    return extract_domain_age_features(url)

def extract_domain_only_features(url: str, enable_domain_age: bool) -> Dict[str, float]:
    return get_domain_age_features_for_mode(url, enable_domain_age)

# ============================================================
# 5. Error analysis (오탐/미탐 분석)
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
# 6. Synthetic typo generation (합성 타이포 URL 생성)
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
# 7. Feature schema / labels (feature 정의 및 라벨)
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
    "domain_age_days",
    "domain_age_log_days",
    "domain_age_missing",
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
    "domain_age_days": "도메인 등록 후 경과 일수",
    "domain_age_log_days": "도메인 나이 로그 변환값",
    "domain_age_missing": "도메인 나이 조회 실패 여부",
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

def _print_metric_summary(metrics: Dict[str, object]) -> None:
    print("[Metrics with Korean explanations]")
    for key in ("accuracy", "roc_auc", "precision", "recall", "f1", "threshold", "confusion_matrix"):
        if key in metrics:
            print(f"  {_annotate_metric_name(key)}: {metrics[key]}")

# ============================================================
# 8. Feature pipeline (최종 feature 벡터 조립)
# ============================================================

def extract_features(
    url: str,
    enable_domain_age: bool = False,
    domain_only: bool = False,
) -> np.ndarray:
    """
    Extract lexical features from a URL.
    Returns:
        np.ndarray shape (len(FEATURE_NAMES),)
    """
    u = (url or "").strip()
    if domain_only:
        features = extract_domain_only_features(u, enable_domain_age)
        return _feature_dict_to_array(features)

    # urlsplit requires scheme to parse netloc well. If missing, prepend.
    parsed = urlsplit(u if "://" in u else "http://" + u)
    host = _safe_lower(parsed.hostname or "")
    path = _safe_lower(parsed.path or "")
    url_lower = _safe_lower(u)
    multi_tld_flag = has_multi_level_tld(host)
    country_code_tld_flag = has_country_code_tld(host)
    safe_second_level_hint_flag = has_safe_second_level_hint(host)
    length_feats = extract_length_features(url)
    domain_age_feats = get_domain_age_features_for_mode(url, enable_domain_age)

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
            float(domain_age_feats["domain_age_days"]),
            float(domain_age_feats["domain_age_log_days"]),
            float(domain_age_feats["domain_age_missing"]),
        ],
        dtype=np.float32,
    )
    return feats

def featurize_urls(
    urls: Iterable[str],
    enable_domain_age: bool = False,
    domain_only: bool = False,
) -> np.ndarray:
    urls_list = list(urls)
    X = np.vstack(
        [
            extract_features(
                u,
                enable_domain_age=enable_domain_age,
                domain_only=domain_only,
            )
            for u in urls_list
        ]
    ) if urls_list else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return X

# ============================================================
# 9. Data split logic (데이터 분할 로직)
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
# 10. Model bundle / train / eval / predict (모델 저장/학습/평가/추론)
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
    domain_only: Optional[bool] = None,
) -> Tuple[int, float, Dict[str, float]]:
    if enable_domain_age is None:
        enable_domain_age = bool(bundle.meta.get("enable_domain_age", False))
    if domain_only is None:
        domain_only = bool(bundle.meta.get("domain_only", False))
    X = featurize_urls([url], enable_domain_age=enable_domain_age, domain_only=domain_only)
    feats = extract_features(url, enable_domain_age=enable_domain_age, domain_only=domain_only)
    base_n = len(bundle.feature_names)
    if X.shape[1] > base_n:
        X = X[:, :base_n]
        feats = feats[:base_n]
    proba = float(predict_proba(bundle.model, X)[0])
    label = 1 if proba >= 0.5 else 0
    feat_map = {name: float(val) for name, val in zip(bundle.feature_names, feats)}
    return label, proba, feat_map

def _validate_single_input_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        raise ValueError("URL must not be empty or whitespace.")

    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("URL must include a valid scheme and host (example: https://example.com).")

    return value

# ============================================================
# 11. Dataset I/O (데이터셋 입출력)
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
