#!/usr/bin/env python3
"""Refresh the local phishing-feed blocklist snapshot.

Pulls every public feed used for training/eval collection, canonicalizes each
URL into lookup keys (host+path+query and host+path), removes anything covered
by trusted-domain rules or the manual allowlist, and writes the BLAKE2b key
hashes to blocklist/phishing_blocklist_snapshot.txt.gz.

Run periodically (e.g. hourly cron) so newly reported phishing is caught:
  python dev/update_blocklist_snapshot.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEV = BASE / "dev"
for path in (BASE, DEV):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import io
import zipfile

from collect_urls import (
    _fetch,
    fetch_nurilab,
    fetch_openphish,
    fetch_phishing_database_kr,
    fetch_phishtank,
    fetch_urlhaus,
)
from phishing_blocklist import blocklist_key_hash, blocklist_keys_for_url
from trusted_domains import is_low_risk_hosted_platform_url, is_trusted_official_url

DEFAULT_OUT = BASE / "blocklist" / "phishing_blocklist_snapshot.txt.gz"
DEFAULT_ALLOWLIST = BASE / "blocklist" / "blocklist_allowlist.txt"

FEEDS = (
    ("urlhaus", fetch_urlhaus),
    ("openphish", fetch_openphish),
    ("phishtank", fetch_phishtank),
    ("phishing_database", fetch_phishing_database_kr),
    ("nurilab", fetch_nurilab),
)


def read_allowlist(path: Path) -> tuple[set[str], set[str]]:
    """Returns (exact_keys, host_prefixes). `host/*` lines drop every key under host."""
    exact: set[str] = set()
    prefixes: set[str] = set()
    if not path.is_file():
        return exact, prefixes
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        if line.endswith("/*"):
            prefixes.add(line[:-2].rstrip("/"))
        else:
            exact.add(line)
    return exact, prefixes


def allowlisted(key: str, exact: set[str], prefixes: set[str]) -> bool:
    if key in exact:
        return True
    host = key.split("/", 1)[0].split("?", 1)[0]
    return host in prefixes


def fetch_popular_hosts(top_n: int) -> set[str]:
    """Tranco top-N hosts: their bare root keys are dropped (deep links still match)."""
    if top_n <= 0:
        return set()
    try:
        raw = _fetch("https://tranco-list.eu/top-1m.csv.zip", timeout=60)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            text = zf.read(zf.namelist()[0]).decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[tranco] 인기 호스트 보호 목록 다운로드 실패: {exc}")
        return set()
    hosts: set[str] = set()
    for line in text.splitlines():
        parts = line.split(",", 1)
        if len(parts) == 2:
            hosts.add(parts[1].strip().lower())
        if len(hosts) >= top_n:
            break
    return hosts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
    parser.add_argument("--protect-top-n", type=int, default=100000)
    args = parser.parse_args(argv)

    exact_allow, prefix_allow = read_allowlist(Path(args.allowlist))
    popular_hosts = fetch_popular_hosts(args.protect_top_n)
    print(f"[tranco] 보호되는 인기 호스트: {len(popular_hosts)}개")
    keys: set[str] = set()
    feed_counts: dict[str, int] = {}
    for name, fetch in FEEDS:
        urls = fetch()
        feed_counts[name] = len(urls)
        for url in urls:
            if is_trusted_official_url(url) or is_low_risk_hosted_platform_url(url):
                continue
            for key in blocklist_keys_for_url(url):
                if not key or allowlisted(key, exact_allow, prefix_allow):
                    continue
                if "/" not in key and "?" not in key and key in popular_hosts:
                    continue
                # Re-check the canonical form: feed URLs with odd schemes or
                # encodings can canonicalize onto a trusted host's key.
                canonical = f"http://{key}"
                if is_trusted_official_url(canonical) or is_low_risk_hosted_platform_url(canonical):
                    continue
                keys.add(key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="ascii") as f:
        for key in sorted(keys):
            f.write(f"{blocklist_key_hash(key):016x}\n")

    meta = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "feed_url_counts": feed_counts,
        "unique_keys": len(keys),
        "allowlist_exact": len(exact_allow),
        "allowlist_prefixes": len(prefix_allow),
    }
    meta_path = out_path.parent / (out_path.name.split(".")[0] + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote={out_path} keys={len(keys)} feeds={feed_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
