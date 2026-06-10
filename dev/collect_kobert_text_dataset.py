#!/usr/bin/env python3
"""Collect reusable Korean text rows for KoBERT training."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import unquote, urlsplit


BASE = Path(__file__).resolve().parents[1]
KOBERT_DIR = BASE / "KoBERT"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
if str(KOBERT_DIR) not in sys.path:
    sys.path.insert(0, str(KOBERT_DIR))

from kobert_train import fetch_text
from trusted_domains import is_trusted_official_url


def fallback_text(url: str, source: str) -> str:
    parsed = urlsplit(url if "://" in url else f"https://{url}")
    host = parsed.netloc or parsed.path.split("/", 1)[0]
    path = parsed.path if parsed.netloc else ""
    query = parsed.query or ""
    decoded_path = unquote(path.replace("/", " ")).strip()
    host_tokens = host.replace(".", " ").replace("-", " ")
    combined = f"{host} {decoded_path} {query}".lower()
    signals: list[str] = []
    trusted = is_trusted_official_url(url)
    if trusted:
        signals.append("공식 신뢰 도메인")
    elif (parsed.scheme or "").lower() == "http":
        signals.append("암호화되지 않은 HTTP")
    if "-" in host:
        signals.append("하이픈 포함 도메인")
    if re.search(r"\d{3,}", host):
        signals.append("긴 숫자 포함 도메인")
    if host.endswith((".top", ".xyz", ".vip", ".shop", ".store", ".click", ".cfd", ".tk", ".ml")):
        signals.append("피싱에 자주 쓰이는 최상위 도메인")
    if not trusted:
        for term, label in (
            ("login", "로그인 유도"),
            ("signin", "로그인 유도"),
            ("verify", "인증 유도"),
            ("account", "계정 정보 유도"),
            ("wallet", "지갑 정보 유도"),
            ("pay", "결제 정보 유도"),
            ("cert", "인증서 유도"),
            ("event", "이벤트 미끼"),
            ("gift", "선물 미끼"),
            ("notice", "공지 사칭"),
            ("amazon", "브랜드명 포함"),
            ("instagram", "브랜드명 포함"),
            ("naver", "브랜드명 포함"),
            ("kakao", "브랜드명 포함"),
            ("bradesco", "금융 브랜드명 포함"),
            ("trustwallet", "지갑 브랜드명 포함"),
        ):
            if term in combined and label not in signals:
                signals.append(label)
    signal_text = ", ".join(signals[:8]) if signals else "특이 신호 낮음"
    return (
        "웹사이트 주소 분석 텍스트. "
        f"도메인 {host}. 도메인 토큰 {host_tokens}. "
        f"경로 정보 {decoded_path or '없음'}. "
        f"URL 구조 신호 {signal_text}."
    )[:512]


def read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            if url:
                rows[url] = dict(row)
    return rows


def read_input(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            label = str(row.get("label") or "").strip()
            if url and label in {"0", "1"}:
                rows.append(
                    {
                        "url": url,
                        "label": label,
                        "source": (row.get("source") or "").strip(),
                    }
                )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "is_korean", "text_source", "text"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BASE / "dev" / "engine_training_inputs" / "kobert_candidates_10k.csv"))
    parser.add_argument("--out", default=str(BASE / "dev" / "engine_training_inputs" / "kobert_text_train_10k.csv"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fallback-url-text", action="store_true", help="Use URL-derived Korean text when live text is empty/non-Korean.")
    parser.add_argument("--fallback-only", action="store_true", help="Skip fetching and build URL-derived Korean text for every row.")
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Rebuild rows even when the output CSV already contains the URL.",
    )
    args = parser.parse_args()

    input_rows = read_input(Path(args.input))
    if args.limit:
        input_rows = input_rows[: args.limit]
    existing = {} if args.refresh_existing else read_existing(Path(args.out))
    targets = [row for row in input_rows if row["url"] not in existing]
    print(f"existing={len(existing)} targets={len(targets)} workers={args.workers}", flush=True)

    collected: list[dict[str, str]] = []
    if args.fallback_only:
        for done, row in enumerate(targets, 1):
            text, is_korean = fallback_text(row["url"], row.get("source", "")), True
            collected.append(
                {
                    "url": row["url"],
                    "label": row["label"],
                    "source": row.get("source", ""),
                    "is_korean": "1" if is_korean else "0",
                    "text_source": "url_fallback",
                    "text": text,
                }
            )
            if done % 100 == 0 or done == len(targets):
                print(f"collected={done}/{len(targets)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            future_map = {pool.submit(fetch_text, row["url"]): row for row in targets}
            done = 0
            for future in as_completed(future_map):
                row = future_map[future]
                done += 1
                try:
                    text, is_korean = future.result()
                except Exception:
                    text, is_korean = "", False
                text_source = "live_fetch"
                if args.fallback_url_text and (not text.strip() or not is_korean):
                    text = fallback_text(row["url"], row.get("source", ""))
                    is_korean = True
                    text_source = "url_fallback"
                collected.append(
                    {
                        "url": row["url"],
                        "label": row["label"],
                        "source": row.get("source", ""),
                        "is_korean": "1" if is_korean else "0",
                        "text_source": text_source,
                        "text": text,
                    }
                )
                if done % 100 == 0 or done == len(targets):
                    print(f"collected={done}/{len(targets)}", flush=True)

    merged = list(existing.values()) + collected
    write_rows(Path(args.out), merged)
    usable = sum(1 for row in merged if row.get("text") and row.get("is_korean") == "1")
    fallback = sum(1 for row in merged if row.get("text_source") == "url_fallback")
    live = sum(1 for row in merged if row.get("text_source") == "live_fetch")
    print(f"wrote={args.out} rows={len(merged)} usable_korean_text={usable} live_fetch={live} url_fallback={fallback}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
