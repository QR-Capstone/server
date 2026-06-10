#!/usr/bin/env python3
"""Audit URL overlap between training and evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from urllib.parse import urlsplit, urlunsplit


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN = os.path.join("dev", "dataset_splits", "train.csv")
DEFAULT_TESTS = [
    os.path.join("dev", "dataset_splits", "validation.csv"),
    os.path.join("dev", "dataset_splits", "test_balanced.csv"),
    os.path.join("dev", "dataset_splits", "test_operational.csv"),
]
DEFAULT_BALANCED_TEST = os.path.join("dev", "dataset_splits", "test_balanced.csv")


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


def audit_file_shape(path: str, *, min_rows: int, min_each_label: int) -> list[str]:
    failures: list[str] = []
    rows = read_rows(path)
    labels: Counter[str] = Counter()
    keys: set[str] = set()
    duplicate_keys: list[str] = []
    usable = 0
    for row in rows:
        key = canonical_url_key(row.get("url") or "")
        label = str(row.get("label") or "").strip()
        if not key or label not in {"0", "1"}:
            continue
        usable += 1
        labels[label] += 1
        if key in keys:
            duplicate_keys.append(key)
        keys.add(key)
    if usable < min_rows:
        failures.append(f"{path}: rows {usable} < min_rows {min_rows}")
    for label in ("0", "1"):
        if labels[label] < min_each_label:
            failures.append(f"{path}: label {label} rows {labels[label]} < min_each_label {min_each_label}")
    if duplicate_keys:
        failures.append(f"{path}: duplicate_keys={len(duplicate_keys)} first={duplicate_keys[0]}")
    print(f"{path}: rows={usable} benign={labels['0']} malicious={labels['1']} duplicate_keys={len(duplicate_keys)}")
    return failures


def audit_balanced(path: str, *, max_delta: int) -> list[str]:
    labels = Counter()
    for row in read_rows(path):
        label = str(row.get("label") or "").strip()
        if label in {"0", "1"} and canonical_url_key(row.get("url") or ""):
            labels[label] += 1
    delta = abs(labels["0"] - labels["1"])
    if delta > max_delta:
        return [f"{path}: benign/malicious delta {delta} > {max_delta}"]
    return []


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", default=[])
    parser.add_argument("--test", action="append", default=[])
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--write-clean-test", help="Write test rows excluding train/test duplicate URL keys.")
    parser.add_argument("--min-train-rows", type=int, default=8000)
    parser.add_argument("--min-test-rows", type=int, default=500)
    parser.add_argument("--min-each-label", type=int, default=1)
    parser.add_argument("--balanced-test", default=DEFAULT_BALANCED_TEST)
    parser.add_argument("--balanced-max-delta", type=int, default=0)
    args = parser.parse_args(argv)

    if not args.train:
        args.train = [DEFAULT_TRAIN]
    if not args.test:
        args.test = list(DEFAULT_TESTS)

    train: dict[str, str] = {}
    test: dict[str, str] = {}
    labels: dict[str, set[str]] = {}
    failures: list[str] = []
    for path in args.train:
        failures.extend(audit_file_shape(path, min_rows=args.min_train_rows, min_each_label=args.min_each_label))
        train.update(read_keys(path))
        for key, values in read_labels(path).items():
            labels.setdefault(key, set()).update(values)
    for path in args.test:
        failures.extend(audit_file_shape(path, min_rows=args.min_test_rows, min_each_label=args.min_each_label))
        test.update(read_keys(path))
        for key, values in read_labels(path).items():
            labels.setdefault(key, set()).update(values)
    if args.balanced_test:
        failures.extend(audit_balanced(args.balanced_test, max_delta=args.balanced_max_delta))

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
    for failure in failures[: args.show]:
        print(f"  FAIL {failure}")
    if args.write_clean_test:
        rows = write_clean_test(args.test, set(train), args.write_clean_test)
        print(f"wrote_clean_test={args.write_clean_test} rows={rows}")
    return 1 if overlap or conflicts or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
