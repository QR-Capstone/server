#!/usr/bin/env python3
"""Collect live hard-benign Korean SMB, ad landing, shopping, clinic, and academy URLs."""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import ssl
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ML_DIR = os.path.join(BASE, "url_ml")
if BASE not in sys.path:
    sys.path.insert(0, BASE)
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)

from dev.collect_urls import fetch_tranco
from url_ml.train_url_ml import canonical_url_key

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
QUERIES = [
    "site:imweb.me 병원 예약",
    "site:cafe24.com 쇼핑몰",
    "site:modoo.at 학원",
    "site:wixsite.com 펜션",
    "site:campaignus.me 랜딩 이벤트",
    "전화번호 도메인 병원",
    "렌트카 예약 랜딩페이지",
    "법무사 상담 홈페이지",
    "세무사 상담 홈페이지",
    "한국 쇼핑몰 이벤트 utm_source",
]

URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.I)
HARD_HOST_RE = re.compile(r"(cafe24|imweb|campaignus|modoo|wixsite|landing|shop|mall|rent|pension|law|tax)", re.I)
DIGIT_RE = re.compile(r"\d")


def load_exclude(paths: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = canonical_url_key(row.get("url") or "")
                if key:
                    keys.add(key)
    return keys


def load_user_urls(paths: Iterable[str]) -> list[str]:
    urls: list[str] = []
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(2048)
            f.seek(0)
            if sample.lower().lstrip().startswith("url") or "," in sample.splitlines()[0]:
                for row in csv.DictReader(f):
                    url = (row.get("url") or "").strip()
                    if url:
                        urls.append(url)
            else:
                urls.extend(line.strip() for line in f if line.strip())
    return urls


def naver_search(query: str, pages: int, timeout: float) -> list[str]:
    urls: list[str] = []
    for start in range(1, max(1, pages) * 10, 10):
        params = urllib.parse.urlencode({"query": query, "start": start})
        req = urllib.request.Request(
            f"https://search.naver.com/search.naver?where=web&sm=tab_pge&{params}",
            headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.7"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read(250000).decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"naver_search failed query={query!r} start={start}: {exc}")
            continue
        for raw in URL_RE.findall(html.unescape(text)):
            clean = raw.rstrip(").,;'\"")
            parsed = urllib.parse.urlsplit(clean)
            if parsed.netloc and "naver.com" not in parsed.netloc:
                urls.append(clean)
    return urls


def is_hard_candidate(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url if "://" in url else f"https://{url}")
    host = parsed.netloc.lower()
    compact_host = host.replace("-", "").replace(".", "")
    phone_like = bool(re.search(r"0\d{7,}", compact_host))
    long_query = len(parsed.query) >= 40 or any(k in parsed.query.lower() for k in ("utm_", "gclid", "campaign"))
    return bool(
        HARD_HOST_RE.search(host)
        or DIGIT_RE.search(host)
        or phone_like
        or long_query
        or host.endswith((".kr", ".co.kr"))
    )


def live_check(url: str, timeout: float) -> tuple[str, bool]:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            code = int(getattr(resp, "status", 0) or 0)
            resp.read(512)
        return url, 200 <= code < 500
    except Exception:
        return url, False


def source_for(url: str) -> str:
    parsed = urllib.parse.urlsplit(url if "://" in url else f"https://{url}")
    text = f"{parsed.netloc} {parsed.path} {parsed.query}".lower()
    if any(token in text for token in ("utm_", "gclid", "campaign", "landing", "event")):
        return "benign_ad_landing"
    return "benign_hard_korean_smb"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--naver-pages", type=int, default=3)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--user-url-file", action="append", default=[])
    parser.add_argument("--existing-normal", action="append", default=[])
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--tranco-n", type=int, default=5000)
    args = parser.parse_args()

    exclude = load_exclude(args.exclude)
    candidates: list[str] = []
    candidates.extend(load_user_urls(args.user_url_file))
    candidates.extend(load_user_urls(args.existing_normal))
    for query in (args.query or QUERIES):
        candidates.extend(naver_search(query, args.naver_pages, args.timeout))
    candidates.extend(fetch_tranco(args.tranco_n))

    unique: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        key = canonical_url_key(url)
        if not key or key in seen or key in exclude:
            continue
        if is_hard_candidate(url):
            seen.add(key)
            unique.append(url)
    print(f"exclude={len(exclude)} hard_candidates={len(unique)} target={args.target}")

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(live_check, url, args.timeout): url for url in unique}
        for i, future in enumerate(as_completed(futures), start=1):
            url, ok = future.result()
            if ok:
                rows.append({"url": url, "label": "0", "source": source_for(url), "is_korean": "1"})
                if len(rows) >= args.target:
                    break
            if i % 100 == 0 or len(rows) >= args.target:
                print(f"checked={i}/{len(unique)} live={len(rows)}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "is_korean"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.out} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
