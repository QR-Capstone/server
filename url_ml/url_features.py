"""Small lexical feature transformer for URLML."""

from __future__ import annotations

import math
import re
from urllib.parse import urlsplit

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

HANGUL_RE = re.compile(r"[가-힣]")
PHONE_HOST_RE = re.compile(r"0\d{7,}")
SHORTENER_HOSTS = {
    "bit.ly",
    "goo.gl",
    "t.co",
    "tinyurl.com",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "cutt.ly",
    "rebrand.ly",
    "tr.ee",
}
BENIGN_PLATFORM_HINTS = (
    "cafe24",
    "imweb",
    "modoo",
    "campaignus",
    "wixsite",
    "framer",
)
SUSPICIOUS_TLDS = {
    "click",
    "cyou",
    "icu",
    "live",
    "lol",
    "monster",
    "online",
    "shop",
    "site",
    "sbs",
    "top",
    "vip",
    "xyz",
}


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    total = len(text)
    counts = {ch: text.count(ch) for ch in set(text)}
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _parse(raw_url: str):
    url = (raw_url or "").strip()
    candidate = url if "://" in url else f"//{url}"
    try:
        parsed = urlsplit(candidate)
    except Exception:
        return url, "", "", "", ""
    host = (parsed.hostname or parsed.netloc or "").lower()
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    return url, host, parsed.path or "", parsed.query or "", tld


class URLLexicalFeatures(BaseEstimator, TransformerMixin):
    """Return sparse numeric URL features alongside char n-grams."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows: list[list[float]] = []
        for raw in X:
            url, host, path, query, tld = _parse(str(raw))
            host_compact = host.replace("-", "").replace(".", "")
            host_len = len(host)
            url_len = len(url)
            digit_count = sum(ch.isdigit() for ch in url)
            host_digit_count = sum(ch.isdigit() for ch in host)
            alpha_count = sum(ch.isalpha() for ch in url)
            rows.append(
                [
                    math.log1p(url_len),
                    math.log1p(host_len),
                    math.log1p(len(path)),
                    math.log1p(len(query)),
                    digit_count / max(1, url_len),
                    host_digit_count / max(1, host_len),
                    url.count("-") / max(1, url_len),
                    host.count("-") / max(1, host_len),
                    url.count(".") / max(1, url_len),
                    path.count("/") / max(1, len(path) or 1),
                    1.0 if "@" in url else 0.0,
                    1.0 if "%" in url else 0.0,
                    1.0 if HANGUL_RE.search(url) else 0.0,
                    1.0 if PHONE_HOST_RE.search(host_compact) else 0.0,
                    1.0 if host in SHORTENER_HOSTS else 0.0,
                    1.0 if any(token in host for token in BENIGN_PLATFORM_HINTS) else 0.0,
                    1.0 if tld in SUSPICIOUS_TLDS else 0.0,
                    min(_entropy(host) / 5.0, 1.0),
                    min(_entropy(path) / 5.0, 1.0),
                    digit_count / max(1, alpha_count + digit_count),
                ]
            )
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))
