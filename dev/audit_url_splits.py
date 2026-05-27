#!/usr/bin/env python3
"""Audit URL overlap between training and evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from urllib.parse import urlsplit, urlunsplit


def canonical_url_key(raw_url: str) -> str:
    """Stable key for duplicate/leakage checks without importing ML deps."""
    raw = (raw_url or "").strip()
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
    except Exception:
        return raw.lower().rstrip("/")
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    query = parsed.query
    if scheme:
        return urlunsplit((scheme, netloc, path, query, "")).lower()
    return urlunsplit(("", netloc, path, query, "")).lower()


def read_keys(path: str) -> dict[str, str]:
    keys: dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            key = canonical_url_key(url)
            if key:
                keys.setdefault(key, url)
    return keys


def read_labels(path: str) -> dict[str, set[str]]:
    labels: dict[str, set[str]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = canonical_url_key(row.get("url") or "")
            label = str(row.get("label") or "").strip()
            if key and label in {"0", "1"}:
                labels.setdefault(key, set()).add(label)
    return labels


def read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_clean_test(paths: list[str], train_keys: set[str], out_path: str) -> int:
    fieldnames = ["url", "label", "source", "is_korean"]
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            key = canonical_url_key(row.get("url") or "")
            if not key or key in train_keys or key in seen:
                continue
            seen.add(key)
            rows.append({name: row.get(name, "") for name in fieldnames})
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--test", action="append", required=True)
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--write-clean-test", help="Write test rows excluding train/test duplicate URL keys.")
    args = parser.parse_args()

    train: dict[str, str] = {}
    test: dict[str, str] = {}
    labels: dict[str, set[str]] = {}
    for path in args.train:
        train.update(read_keys(path))
        for key, values in read_labels(path).items():
            labels.setdefault(key, set()).update(values)
    for path in args.test:
        test.update(read_keys(path))
        for key, values in read_labels(path).items():
            labels.setdefault(key, set()).update(values)

    overlap = sorted(set(train) & set(test))
    conflicts = sorted(key for key, values in labels.items() if len(values) > 1)
    print(
        f"train_unique={len(train)} test_unique={len(test)} "
        f"overlap={len(overlap)} label_conflicts={len(conflicts)}"
    )
    for key in overlap[: args.show]:
        print(f"  OVERLAP {test[key]}")
    for key in conflicts[: args.show]:
        print(f"  LABEL_CONFLICT {key} labels={sorted(labels[key])}")
    if args.write_clean_test:
        rows = write_clean_test(args.test, set(train), args.write_clean_test)
        print(f"wrote_clean_test={args.write_clean_test} rows={rows}")
    return 1 if overlap or conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
