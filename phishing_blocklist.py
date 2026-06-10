"""Local snapshot lookup of public phishing feeds.

Known-phishing URLs reported to public feeds (PhishTank, OpenPhish, URLhaus,
Phishing.Database, Nurilab) are matched exactly by canonical key so the URL
lane catches reported campaigns that look lexically benign. Host-only matching
is intentionally not used: popular legitimate hosts (google.com, github.com,
shorteners, hosting platforms) appear in feeds via deep links and would turn
into false positives.

The snapshot file stores one 16-hex-char BLAKE2b key hash per line
(gzip-compressed). Refresh it with dev/update_blocklist_snapshot.py.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import threading
from urllib.parse import urlsplit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNAPSHOT_PATH = os.getenv(
    "URL_BLOCKLIST_SNAPSHOT_PATH",
    os.path.join(BASE_DIR, "blocklist", "phishing_blocklist_snapshot.txt.gz"),
)
BLOCKLIST_ENABLED = os.getenv("URL_BLOCKLIST_ENABLED", "1") == "1"

_lock = threading.Lock()
_loaded = False
_hashes: frozenset[int] = frozenset()


def blocklist_key_hash(key: str) -> int:
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


def blocklist_keys_for_url(raw_url: str) -> list[str]:
    """Canonical lookup keys: host+path+query and host+path (scheme/www/port agnostic)."""
    raw = (raw_url or "").strip()
    if not raw:
        return []
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
    except Exception:
        return [raw.lower().rstrip("/")]
    host = (parsed.netloc or "").lower().rsplit("@", 1)[-1].split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return []
    path = (parsed.path or "").rstrip("/").lower()
    query = (parsed.query or "").lower()
    keys = [f"{host}{path}?{query}" if query else f"{host}{path}"]
    if query and path:
        # Query-stripped variant catches rotating campaign parameters, but only
        # for deep links: stripping a root URL's query would blocklist the
        # whole host (open-redirect feed entries on legitimate hosts).
        keys.append(f"{host}{path}")
    return keys


def _load_snapshot() -> frozenset[int]:
    global _loaded, _hashes
    if _loaded:
        return _hashes
    with _lock:
        if _loaded:
            return _hashes
        hashes: set[int] = set()
        try:
            with gzip.open(SNAPSHOT_PATH, "rt", encoding="ascii") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        hashes.add(int(line, 16))
        except OSError:
            pass
        _hashes = frozenset(hashes)
        _loaded = True
    return _hashes


def blocklist_snapshot_size() -> int:
    return len(_load_snapshot())


def is_blocklisted_url(raw_url: str) -> bool:
    """True when the URL (or its query-stripped form) is in the feed snapshot."""
    if not BLOCKLIST_ENABLED:
        return False
    hashes = _load_snapshot()
    if not hashes:
        return False
    return any(blocklist_key_hash(key) in hashes for key in blocklist_keys_for_url(raw_url))
