#!/usr/bin/env python3
"""Generate clearly marked Korean phishing-pattern synthetic URL rows."""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import random

BRANDS = [
    "naver", "naverpay", "kakao", "kakaopay", "toss", "payco",
    "kbstar", "shinhan", "wooribank", "nhbank", "kebhana", "kakaobank",
    "hometax", "nts", "gov", "epost", "nhis", "kisa",
    "coupang", "gmarket", "auction", "11st", "ssg", "lotteon",
]

TOKENS = [
    "login", "verify", "cert", "notice", "event", "safe", "pay", "member",
    "auth", "account", "security", "center", "update", "gift", "refund",
]

TLDS = ["shop", "site", "click", "top", "online", "vip", "store", "xyz", "help"]
HOSTERS = ["pages.dev", "vercel.app", "netlify.app", "github.io", "web.app", "wixsite.com"]
PATHS = [
    "/login", "/member/cert", "/pay/auth", "/event/check", "/notice/view",
    "/account/update", "/security/verify", "/refund/apply", "/gift/claim",
]


def generate(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    urls: list[str] = []
    combos = list(itertools.product(BRANDS, TOKENS, TLDS))
    rng.shuffle(combos)
    for brand, token, tld in combos:
        serial = rng.randint(1000, 999999)
        variant = rng.choice(
            [
                f"https://{brand}-{token}-{serial}.{tld}{rng.choice(PATHS)}",
                f"https://{token}-{brand}-kr{serial}.{tld}{rng.choice(PATHS)}?next=login&locale=ko_KR",
                f"https://{brand}{serial}-{token}.{rng.choice(HOSTERS)}{rng.choice(PATHS)}",
                f"http://www.{brand}-{token}.kr-{serial}.{tld}/index.php?mode=cert",
            ]
        )
        urls.append(variant)
        if len(urls) >= count:
            break
    return urls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate(args.count, args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "is_korean"])
        writer.writeheader()
        for url in rows:
            writer.writerow(
                {
                    "url": url,
                    "label": "1",
                    "source": "malicious_synthetic_kr",
                    "is_korean": "1",
                }
            )
    print(f"wrote={args.out} rows={len(rows)} source=malicious_synthetic_kr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
