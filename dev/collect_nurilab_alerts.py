#!/usr/bin/env python3
"""Collect real phishing URLs from Nurilab's public phishing-alert list."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime

UA = "Mozilla/5.0 (compatible; phishing-detection-research/1.0)"
API = "https://www.nurilab.com/kr/alert/paginate_alerts.php"
TITLE_RE = re.compile(r'class="content_target_item[^"]*" title="([^"]+)"')
DATE_RE = re.compile(r'class="content_date_item[^"]*">([^<]+)<')
NAME_RE = re.compile(r'class="load-content-phishing"[^>]*>\s*([^<]+?)\s*</a>')
FRAUD_RE = re.compile(r'data-fraud-value="([^"]*)"')
DETECT_RE = re.compile(r'data-detect="([^"]*)"')


def _fetch_json(year: str, page: int, per_page: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "page": page,
            "per_page": per_page,
            "year": year,
            "categories": "",
            "brands": "",
            "search": "",
            "detects": "",
        }
    )
    req = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def refang(raw: str) -> str:
    text = html.unescape(raw or "").strip()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"(?i)\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", ".", text)
    text = re.sub(r"(?i)^hxxp", "http", text)
    text = text.rstrip(".,;)")
    if text and "://" not in text and "." in text:
        text = "http://" + text
    return text


def parse_item(snippet: str, year: str) -> dict | None:
    title_match = TITLE_RE.search(snippet)
    if not title_match:
        return None
    url = refang(title_match.group(1))
    if not url.startswith("http"):
        return None
    date_match = DATE_RE.search(snippet)
    date_text = (date_match.group(1).strip() if date_match else "").replace(" ", "")
    iso_date = ""
    try:
        iso_date = datetime.strptime(date_text, "%Y.%m.%d").date().isoformat()
    except ValueError:
        iso_date = ""
    name_match = NAME_RE.search(snippet)
    fraud_match = FRAUD_RE.search(snippet)
    detect_match = DETECT_RE.search(snippet)
    return {
        "url": url,
        "label": "1",
        "source": "nurilab_phishing_alert",
        "date": iso_date,
        "year": year,
        "detect": detect_match.group(1) if detect_match else "",
        "category": fraud_match.group(1) if fraud_match else "",
        "title": html.unescape(name_match.group(1)).strip() if name_match else "",
    }


def collect(years: list[str], per_page: int) -> list[dict]:
    rows: list[dict] = []
    seen: set[str] = set()
    for year in years:
        first = _fetch_json(year, 1, per_page)
        total_pages = int(first.get("total_pages") or 1)
        print(f"[nurilab] {year}: items={first.get('total_items')} pages={total_pages}", flush=True)
        pages = [first]
        for page in range(2, total_pages + 1):
            pages.append(_fetch_json(year, page, per_page))
        for payload in pages:
            for snippet in payload.get("items") or []:
                row = parse_item(snippet, year)
                if row is None or row["url"] in seen:
                    continue
                seen.add(row["url"])
                rows.append(row)
        print(f"[nurilab] {year}: unique_urls={len(rows)}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2025,2026")
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument("--out", default="dev/nurilab_alerts_real.csv")
    args = parser.parse_args()
    years = [part.strip() for part in args.years.split(",") if part.strip()]
    rows = collect(years, args.per_page)
    fieldnames = ["url", "label", "source", "date", "year", "detect", "category", "title"]
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={args.out} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
