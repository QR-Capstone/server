#!/usr/bin/env python3
"""Collect a balanced non-overlap eval set from current public feeds."""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
import zipfile
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DEV = BASE / "dev"
URL_ML_DIR = BASE / "url_ml"
for path in (BASE, DEV, URL_ML_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from collect_urls import (
    _fetch,
    fetch_kisa_phishing,
    fetch_nurilab,
    fetch_openphish,
    fetch_phishing_database_kr,
    fetch_phishstats_kr,
    fetch_phishtank,
    fetch_urlhaus,
)
from train_url_ml import canonical_url_key


def read_reference(paths: list[str]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = canonical_url_key(row.get("url") or "")
                if key:
                    keys.add(key)
    return keys


def add_unique(rows: list[dict[str, str]], seen: set[str], refs: set[str], urls: list[str], label: str, source: str, limit: int) -> None:
    for url in urls:
        key = canonical_url_key(url)
        if not key or key in refs or key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "url": url.strip(),
                "label": label,
                "source": source,
                "is_korean": "1" if ".kr" in url.lower() else "0",
                "canonical_key": key,
            }
        )
        if sum(1 for row in rows if row["label"] == label) >= limit:
            return


def fetch_tranco_deep(n: int, start_rank: int, scan_limit: int) -> list[str]:
    print(f"[tranco-deep] 다운로드 중... start_rank={start_rank} scan_limit={scan_limit}")
    try:
        raw = _fetch("https://tranco-list.eu/top-1m.csv.zip", timeout=60)
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            text = zf.read(name).decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[tranco-deep] 실패: {exc}")
        return []

    urls: list[str] = []
    max_rank = max(start_rank, 1) + max(scan_limit, n)
    for line in text.splitlines():
        parts = line.strip().split(",", 1)
        if len(parts) != 2:
            continue
        try:
            rank = int(parts[0])
        except ValueError:
            continue
        if rank < start_rank:
            continue
        if rank >= max_rank:
            break
        domain = parts[1].strip()
        if domain and "." in domain:
            urls.append(f"https://{domain}")
    print(f"[tranco-deep] {len(urls)}개 후보")
    return urls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-label", type=int, default=1000)
    parser.add_argument("--tranco-n", type=int, default=5000)
    parser.add_argument("--tranco-start-rank", type=int, default=1)
    parser.add_argument("--tranco-scan-limit", type=int, default=250000)
    parser.add_argument("--skip-urlhaus", action="store_true")
    parser.add_argument("--skip-openphish", action="store_true")
    parser.add_argument("--include-kisa", action="store_true")
    parser.add_argument("--include-nurilab", action="store_true")
    parser.add_argument("--include-phishtank", action="store_true")
    parser.add_argument("--include-phishing-db", action="store_true")
    args = parser.parse_args()

    refs = read_reference(args.reference)
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    malicious_sources: list[tuple[str, list[str]]] = []
    if not args.skip_urlhaus:
        malicious_sources.append(("urlhaus_recent", fetch_urlhaus()))
    malicious_sources.append(("phishstats_kr", fetch_phishstats_kr()))
    if not args.skip_openphish:
        malicious_sources.append(("openphish", fetch_openphish()))
    if args.include_kisa:
        malicious_sources.append(("kisa", fetch_kisa_phishing()))
    if args.include_nurilab:
        malicious_sources.append(("nurilab", fetch_nurilab()))
    if args.include_phishtank:
        malicious_sources.append(("phishtank", fetch_phishtank()))
    if args.include_phishing_db:
        malicious_sources.append(("phishing_database_active", fetch_phishing_database_kr()))

    for source, urls in malicious_sources:
        add_unique(rows, seen, refs, urls, "1", source, args.per_label)
        if sum(1 for row in rows if row["label"] == "1") >= args.per_label:
            break

    benign_target = max(args.tranco_n, args.per_label)
    add_unique(
        rows,
        seen,
        refs,
        fetch_tranco_deep(benign_target, args.tranco_start_rank, args.tranco_scan_limit),
        "0",
        "tranco_latest_nonoverlap",
        args.per_label,
    )

    label_counts = Counter(row["label"] for row in rows)
    n = min(label_counts.get("0", 0), label_counts.get("1", 0), args.per_label)
    selected = [row for row in rows if row["label"] == "0"][:n] + [row for row in rows if row["label"] == "1"][:n]
    selected.sort(key=lambda row: row["canonical_key"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "is_korean", "canonical_key"])
        writer.writeheader()
        writer.writerows(selected)

    print(f"reference_unique={len(refs)}")
    print(f"collected={dict(label_counts)} selected_per_label={n} rows={len(selected)}")
    print(f"sources={dict(Counter(row['source'] for row in selected))}")
    print(f"wrote={args.out}")
    return 0 if n == args.per_label else 2


if __name__ == "__main__":
    raise SystemExit(main())
