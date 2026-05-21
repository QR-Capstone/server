#!/usr/bin/env python3
"""Fast unique benign collection from Tranco candidates."""

from __future__ import annotations

import argparse
import csv
import os
import ssl
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from dev.collect_urls import fetch_tranco, save_csv

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"


def load_exclude(paths: Iterable[str]) -> set[str]:
    exclude: set[str] = set()
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                if url:
                    exclude.add(url)
    return exclude


def alive(url: str, timeout: float) -> tuple[str, bool]:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.8,en;q=0.7"},
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            code = int(getattr(resp, "status", 0) or 0)
        return url, 200 <= code < 500
    except urllib.error.HTTPError as e:
        return url, 200 <= int(e.code) < 500
    except Exception:
        req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                code = int(getattr(resp, "status", 0) or 0)
                resp.read(128)
            return url, 200 <= code < 500
        except urllib.error.HTTPError as e:
            return url, 200 <= int(e.code) < 500
        except Exception:
            return url, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", type=int, default=1500)
    parser.add_argument("--tranco-n", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    exclude = load_exclude(args.exclude)
    candidates = [u for u in fetch_tranco(args.tranco_n) if u not in exclude]
    candidates = list(dict.fromkeys(candidates))
    print(f"exclude={len(exclude)} candidates={len(candidates)} target={args.target}", flush=True)

    rows: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(alive, url, args.timeout): url for url in candidates}
        for future in as_completed(futures):
            done += 1
            url, ok = future.result()
            if ok:
                rows.append({"url": url, "label": 0, "source": "tranco_benign_fast", "is_korean": False})
                if len(rows) >= args.target:
                    break
            if done % 250 == 0 or len(rows) >= args.target:
                print(f"checked={done}/{len(candidates)} alive={len(rows)}", flush=True)

    save_csv(rows, args.out)
    for i in range(5):
        chunk = rows[i * 300 : (i + 1) * 300]
        if not chunk:
            break
        split_path = args.out.replace(".csv", f"_r{i+1}.csv")
        save_csv(chunk, split_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
