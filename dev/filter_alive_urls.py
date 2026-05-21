#!/usr/bin/env python3
"""Filter URL lists to currently reachable sites before evaluation."""

from __future__ import annotations

import argparse
import concurrent.futures
import ssl
import urllib.error
import urllib.request
from typing import Iterable, Tuple


UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def read_urls(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(dict.fromkeys(line.strip() for line in f if line.strip()))


def normalize_url(url: str) -> str:
    raw = (url or "").strip()
    if raw.startswith(("http://", "https://")):
        return raw
    return "https://" + raw


def is_alive(url: str, timeout: float) -> Tuple[str, bool, str]:
    target = normalize_url(url)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        target,
        headers={
            "User-Agent": UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            code = int(getattr(resp, "status", 0) or 0)
            resp.read(512)
        return url, 200 <= code < 500, f"HTTP_{code}"
    except urllib.error.HTTPError as e:
        return url, 200 <= int(e.code) < 500, f"HTTP_{e.code}"
    except Exception as e:
        return url, False, type(e).__name__


def filter_alive(urls: Iterable[str], workers: int, timeout: float) -> list[str]:
    ordered = list(urls)
    alive: list[str] = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(is_alive, url, timeout): url for url in ordered}
        for future in concurrent.futures.as_completed(futures):
            done += 1
            url, ok, reason = future.result()
            if ok:
                alive.append(url)
            if done % 25 == 0 or done == len(ordered):
                print(f"[{done}/{len(ordered)}] alive={len(alive)} last={reason}", flush=True)
    keep = set(alive)
    return [url for url in ordered if url in keep]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=8.0)
    args = parser.parse_args()

    urls = read_urls(args.input)
    alive = filter_alive(urls, workers=args.workers, timeout=args.timeout)
    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for url in alive:
            f.write(url + "\n")
    print(f"saved: {args.output} ({len(alive)}/{len(urls)} alive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
