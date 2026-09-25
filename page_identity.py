"""Page identity check.

URL-only models score the address string. This check reads the page the
address actually opens and compares the service it claims to be with the host.
A short title or site name that is a known service, on a host that is not that
service, is a mismatch. Official hosts and lookup failures stay unchanged.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlsplit

# Official hosts for names a phishing page often puts in the title or site name.
# The match is the name itself plus a short login/web suffix, so a news headline
# that merely mentions the name does not qualify.
_BRANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("네이버", ("naver.com", "navercorp.com")),
    ("naver", ("naver.com", "navercorp.com")),
    ("카카오톡", ("kakao.com", "kakaocorp.com", "daum.net")),
    ("카카오", ("kakao.com", "kakaocorp.com", "daum.net")),
    ("kakao", ("kakao.com", "kakaocorp.com", "daum.net")),
    ("spotify", ("spotify.com",)),
    ("텔레그램", ("telegram.org", "t.me")),
    ("telegram", ("telegram.org", "t.me")),
    ("전자랜드", ("etland.co.kr", "etlandmall.co.kr")),
    ("한국투자증권", ("truefriend.com", "koreainvestment.com")),
    ("whatsapp", ("whatsapp.com",)),
    ("roblox", ("roblox.com",)),
    ("ok저축은행", ("ok.co.kr", "oksb.co.kr")),
)

_SUFFIX = r"(?:로그인|login|sign in|signin|web|인증)?"
_SEPARATOR = r"[\s\-—|:]*"


def _host(url: str) -> str:
    candidate = url if "://" in url else f"//{url}"
    try:
        host = (urlsplit(candidate).hostname or "").lower().strip(".")
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_official(host: str, suffixes: tuple[str, ...]) -> bool:
    return any(host == suffix or host.endswith("." + suffix) for suffix in suffixes)


def _claimed_brand(title: str, site_name: str) -> tuple[str, tuple[str, ...]] | None:
    site = re.sub(r"\s+", " ", (site_name or "").strip().lower())
    for name, suffixes in _BRANDS:
        if site and site == name:
            return name, suffixes
    compact = re.sub(r"\s+", " ", (title or "").strip().lower())
    if not compact or len(compact) > 40:
        return None
    for name, suffixes in _BRANDS:
        pattern = rf"^{re.escape(name)}{_SEPARATOR}{_SUFFIX}$"
        if re.fullmatch(pattern, compact, flags=re.I):
            return name, suffixes
    return None


def claimed_identity_mismatch(url: str, html: str) -> dict[str, Any] | None:
    """Return a dangerous result when the opened page claims another service."""
    if not html:
        return None
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return None
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    site_tag = soup.find("meta", property="og:site_name") or soup.find("meta", attrs={"name": "og:site_name"})
    site_name = (site_tag.get("content") or "").strip() if site_tag else ""
    claimed = _claimed_brand(title, site_name)
    if claimed is None:
        return None
    name, suffixes = claimed
    host = _host(url)
    if not host or _is_official(host, suffixes):
        return None
    return {
        "url": url,
        "riskLevel": "DANGEROUS",
        "judgment": "unnormal",
        "verdict": "malicious",
        "probability": 0.9,
        "claimed_service": name,
        "host": host,
        "title": title[:80],
        "adjustment_reason": f"페이지가 {name}를 표방하지만 도메인은 {host}",
    }


def _read_head(response: Any, limit: int) -> str:
    """Stop once the title is closed or the byte cap is reached."""
    chunks: list[bytes] = []
    size = 0
    for chunk in response.iter_content(chunk_size=2048):
        if not chunk:
            continue
        chunks.append(chunk)
        size += len(chunk)
        head = b"".join(chunks).lower()
        if b"</title>" in head or size >= limit:
            break
    return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")


def _fetch_head(url: str, timeout: float, limit: int) -> dict[str, Any] | None:
    import requests

    response = requests.get(
        url if "://" in url else "https://" + url,
        timeout=(min(0.3, timeout), timeout),
        verify=False,
        allow_redirects=True,
        stream=True,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    try:
        if not (200 <= response.status_code < 400):
            return None
        return claimed_identity_mismatch(url, _read_head(response, limit))
    finally:
        response.close()


def fetch_and_check(url: str, timeout: float | None = None) -> dict[str, Any] | None:
    """Read only the page head. Failure leaves the caller’s verdict unchanged."""
    if timeout is None:
        timeout = float(os.getenv("PAGE_IDENTITY_TIMEOUT", "0.6"))
    limit = int(os.getenv("PAGE_IDENTITY_MAX_BYTES", "8192"))
    try:
        import requests  # noqa: F401
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout
    except Exception:
        return None
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(_fetch_head, url, timeout, limit).result(timeout=timeout)
    except (FuturesTimeout, Exception):
        return None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
