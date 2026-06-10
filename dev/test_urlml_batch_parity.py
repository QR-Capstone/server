#!/usr/bin/env python3
"""Self-check URLML single and vectorized batch inference parity."""

from __future__ import annotations

import csv
import os
import sys


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ML_DIR = os.path.join(BASE, "url_ml")
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from url_ml_engine import load_url_ml_model, predict_url_ml, predict_url_ml_batch  # noqa: E402


def _read_sample(path: str, limit: int) -> list[str]:
    urls: list[str] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            if url:
                urls.append(url)
            if len(urls) >= limit:
                break
    return urls


def _assert_same(single: dict, batch: dict, url: str) -> None:
    keys = {
        "verdict",
        "riskLevel",
        "probability",
        "raw_probability",
        "ml_probability",
        "heuristic_probability",
        "threshold",
        "adjusted_by_rule",
        "adjustment_reason",
    }
    for key in keys:
        if single.get(key) != batch.get(key):
            raise AssertionError(f"{url}: {key} single={single.get(key)!r} batch={batch.get(key)!r}")


def main() -> int:
    model, status = load_url_ml_model()
    if model is None:
        raise AssertionError(f"URLML model unavailable: {status.reason}")

    urls = [
        "https://trustwallet.com",
        "http://trustwallet-web.at",
        "https://www.apps-offiice-wps.com.cn/login",
        "https://example.com",
        "https://aurumclinic.co.kr/a/",
        "https://nkaeklkub.us16.list-manage.com/track/click?u=e383ad4a4a5b7ac07eb70f0be&id=041a26f896&e=bb0fa0d0f0",
        " https://qshopc.co.kr/ ",
    ]
    urls.extend(_read_sample(os.path.join(BASE, "dev", "dataset_splits", "test_balanced.csv"), 25))
    urls.extend(_read_sample(os.path.join(BASE, "dev", "nonoverlap_crossfeed_eval_20260524.csv"), 25))

    singles = [predict_url_ml(model, url) for url in urls]
    batches = predict_url_ml_batch(model, urls)
    assert len(singles) == len(batches) == len(urls)
    for url, single, batch in zip(urls, singles, batches):
        _assert_same(single, batch, url)

    print(f"urlml batch parity selfcheck passed rows={len(urls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
