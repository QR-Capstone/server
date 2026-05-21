#!/usr/bin/env python3
"""Collect live benign SMB-style websites from Naver search results.

This is intentionally a data collector, not an allowlist. URLs found here are
used as normal training/evaluation samples after live checks.
"""

from __future__ import annotations

import argparse
import csv
import html
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlsplit


UA_PC = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
UA_MOBILE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148"

DEFAULT_QUERIES = [
    "청주 과외",
    "목포 맛집",
    "부산 인테리어",
    "대전 법무법인",
    "강남 피부과",
    "분당 영어학원",
    "수원 렌트카",
    "대구 세무사",
    "광주 꽃집",
    "인천 홈페이지 제작",
    "창원 필라테스",
    "천안 치과",
    "제주 펜션",
    "울산 변호사",
    "성남 컴퓨터 수리",
    "일산 네일샵",
    "전주 한정식",
    "안산 자동차 정비",
    "춘천 카페",
    "포항 이사 업체",
    "김해 요양원",
    "구미 학원",
    "평택 공방",
    "용인 애견미용",
]

BLOCKED_HOST_PARTS = (
    "naver.com",
    "pstatic.net",
    "naver.net",
    "daum.net",
    "kakao.com",
    "google.com",
    "youtube.com",
    "instagram.com",
    "facebook.com",
    "twitter.com",
    "x.com",
)


def _fetch(url: str, user_agent: str, timeout: int = 15) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": user_agent, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(400_000)
    return raw.decode("utf-8", errors="replace")


def _hostname(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").strip(".").lower()
    except Exception:
        return ""


def _registered_like(host: str) -> str:
    parts = host.split(".")
    if len(parts) >= 3 and parts[-2] in {"co", "or", "go", "ne", "ac", "re", "pe"} and parts[-1] == "kr":
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return host


def _looks_external(url: str) -> bool:
    host = _hostname(url)
    if not host or "." not in host:
        return False
    if any(part in host for part in BLOCKED_HOST_PARTS):
        return False
    if host.endswith((".jpg", ".png", ".gif", ".css", ".js")):
        return False
    return url.startswith(("http://", "https://"))


def _extract_urls(text: str) -> list[str]:
    candidates: list[str] = []
    patterns = [
        r'href=["\'](https?://[^"\']+)["\']',
        r'"(https?://[^"]+)"',
        r"'(https?://[^']+)'",
        r"url=([^&\"'>\\]+)",
        r"u=([^&\"'>\\]+)",
    ]
    for pattern in patterns:
        for raw in re.findall(pattern, text):
            value = html.unescape(raw)
            value = value.encode("utf-8", errors="ignore").decode("unicode_escape", errors="ignore")
            value = urllib.parse.unquote(value)
            if value.startswith("http"):
                candidates.append(value)
    return candidates


def collect_candidates(queries: list[str], delay: float) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for query in queries:
        encoded = urllib.parse.quote(query)
        urls = [
            (f"https://search.naver.com/search.naver?where=nexearch&query={encoded}", UA_PC),
            (f"https://m.search.naver.com/search.naver?where=m&query={encoded}", UA_MOBILE),
        ]
        for search_url, ua in urls:
            try:
                text = _fetch(search_url, ua)
            except Exception as e:
                print(f"[search] failed query={query!r}: {e}")
                continue
            for url in _extract_urls(text):
                clean = url.split("#", 1)[0].strip()
                host = _hostname(clean)
                if not _looks_external(clean):
                    continue
                key = _registered_like(host)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"url": clean, "label": "0", "source": f"naver_search:{query}", "is_korean": "1"})
        if delay:
            time.sleep(delay)
        print(f"[query] {query}: candidates={len(rows)}", flush=True)
    return rows


def _check_alive(row: dict[str, str], timeout: int) -> dict[str, str] | None:
    url = row["url"]
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(
            url,
            headers={"User-Agent": UA_PC, "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"},
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            if resp.status >= 400:
                return None
            final_url = resp.geturl()
            chunk = resp.read(4096).decode("utf-8", errors="replace")
        if any(part in _hostname(final_url) for part in BLOCKED_HOST_PARTS):
            return None
        out = dict(row)
        out["url"] = final_url.split("#", 1)[0]
        if re.search(r"[가-힣]", chunk) or _hostname(out["url"]).endswith(".kr"):
            out["is_korean"] = "1"
        return out
    except Exception:
        return None


def filter_alive(rows: list[dict[str, str]], target: int, workers: int, timeout: int) -> list[dict[str, str]]:
    alive: list[dict[str, str]] = []
    seen: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {pool.submit(_check_alive, row, timeout): row for row in rows}
        for i, future in enumerate(as_completed(future_map), 1):
            row = future.result()
            if row:
                key = _registered_like(_hostname(row["url"]))
                if key not in seen:
                    seen.add(key)
                    alive.append(row)
                    if len(alive) % 20 == 0:
                        print(f"[alive] {len(alive)} / checked={i}", flush=True)
            if len(alive) >= target:
                break
    return alive[:target]


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "is_korean"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={path} rows={len(rows)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join("dev", "benign_naver_search_20260521.csv"))
    parser.add_argument("--target", type=int, default=120)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--timeout", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--query", action="append", default=[])
    args = parser.parse_args()

    queries = args.query or DEFAULT_QUERIES
    candidates = collect_candidates(queries, args.delay)
    print(f"candidates={len(candidates)}")
    alive = filter_alive(candidates, args.target, args.workers, args.timeout)
    write_csv(args.out, alive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
