#!/usr/bin/env python3
"""Audit engine-specific training input CSVs for size, labels, duplicates, and eval leakage."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote


BASE = Path(__file__).resolve().parents[1]
URL_ML_DIR = BASE / "url_ml"
if str(URL_ML_DIR) not in sys.path:
    sys.path.insert(0, str(URL_ML_DIR))

from train_url_ml import canonical_url_key


def canonical_key_variants(url: str) -> set[str]:
    key = canonical_url_key(url)
    variants = {key} if key else set()
    decoded_key = canonical_url_key(unquote(url or ""))
    if decoded_key:
        variants.add(decoded_key)
    return variants


DEFAULT_INPUT_DIR = BASE / "dev" / "engine_training_inputs"
DEFAULT_EVALS = [
    BASE / "dev" / "dataset_splits" / "validation.csv",
    BASE / "dev" / "dataset_splits" / "test_balanced.csv",
    BASE / "dev" / "dataset_splits" / "test_operational.csv",
]


def _read_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.is_file():
        return keys
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            keys.update(canonical_key_variants(row.get("url") or ""))
    return keys


def _read_key_labels(path: Path) -> dict[str, set[str]]:
    key_labels: dict[str, set[str]] = defaultdict(set)
    if not path.is_file():
        return key_labels
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = canonical_url_key(row.get("url") or "")
            label = str(row.get("label") or "").strip()
            if key and label in {"0", "1"}:
                key_labels[key].add(label)
    return key_labels


def _audit_csv(path: Path, *, min_rows: int, eval_keys: set[str], require_text: bool = False) -> list[str]:
    failures: list[str] = []
    labels: Counter[str] = Counter()
    seen: set[str] = set()
    duplicate = 0
    overlap = 0
    rows = 0
    text_rows = 0
    if not path.is_file():
        return [f"{path}: missing"]
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            label = str(row.get("label") or "").strip()
            key = canonical_url_key(url)
            if not url or label not in {"0", "1"} or not key:
                continue
            rows += 1
            labels[label] += 1
            if row.get("text"):
                text_rows += 1
            if key in seen:
                duplicate += 1
            if key in eval_keys:
                overlap += 1
            seen.add(key)
    if rows < min_rows:
        failures.append(f"{path}: rows {rows} < {min_rows}")
    for label in ("0", "1"):
        if labels[label] <= 0:
            failures.append(f"{path}: missing label {label}")
    if duplicate:
        failures.append(f"{path}: duplicate_keys={duplicate}")
    if overlap:
        failures.append(f"{path}: eval_overlap={overlap}")
    if require_text and text_rows != rows:
        failures.append(f"{path}: text_rows {text_rows} != rows {rows}")
    print(
        f"{path}: rows={rows} benign={labels['0']} malicious={labels['1']} "
        f"duplicate_keys={duplicate} eval_overlap={overlap} text_rows={text_rows}"
    )
    return failures


def _audit_keyset(
    name: str,
    path: Path,
    *,
    min_unique_rows: int,
    eval_keys: set[str],
    allow_duplicate_keys: bool = False,
) -> tuple[set[str], list[str]]:
    failures: list[str] = []
    key_labels = _read_key_labels(path)
    unique_rows = len(key_labels)
    overlap = len(set(key_labels) & eval_keys)
    label_conflicts = {key: labels for key, labels in key_labels.items() if len(labels) > 1}
    duplicate = 0
    if path.is_file():
        total_rows = 0
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = canonical_url_key(row.get("url") or "")
                label = str(row.get("label") or "").strip()
                if key and label in {"0", "1"}:
                    total_rows += 1
        duplicate = total_rows - unique_rows
    else:
        failures.append(f"{path}: missing")

    print(f"{path}: unique_rows={unique_rows} duplicate_keys={duplicate} eval_overlap={overlap}")
    if unique_rows < min_unique_rows:
        failures.append(f"{path}: unique_rows {unique_rows} < {min_unique_rows}")
    if duplicate and not allow_duplicate_keys:
        failures.append(f"{path}: duplicate_keys={duplicate}")
    if overlap:
        failures.append(f"{path}: eval_overlap={overlap}")
    if label_conflicts:
        failures.append(f"{path}: label_conflicts={len(label_conflicts)}")
    return set(key_labels), failures


def _audit_matching_keysets(left_name: str, left: Path, right_name: str, right: Path) -> list[str]:
    failures: list[str] = []
    left_labels = _read_key_labels(left)
    right_labels = _read_key_labels(right)
    left_keys = set(left_labels)
    right_keys = set(right_labels)
    missing = left_keys - right_keys
    extra = right_keys - left_keys
    label_mismatch = [
        key
        for key in sorted(left_keys & right_keys)
        if left_labels[key] != right_labels[key]
    ]
    print(
        f"{left_name}_vs_{right_name}: missing_in_{right_name}={len(missing)} "
        f"extra_in_{right_name}={len(extra)} label_mismatch={len(label_mismatch)}"
    )
    if missing:
        failures.append(f"{right}: missing_keys_from_{left_name}={len(missing)}")
    if extra:
        failures.append(f"{right}: extra_keys_not_in_{left_name}={len(extra)}")
    if label_mismatch:
        failures.append(f"{right}: label_mismatch_with_{left_name}={len(label_mismatch)}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit engine training input CSVs.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--eval", action="append", default=[str(path) for path in DEFAULT_EVALS])
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--gnn-existing", default=str(BASE / "gnn" / "gnn_total_dataset.csv"))
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    eval_keys: set[str] = set()
    for path in args.eval:
        eval_keys.update(_read_keys(Path(path)))

    failures: list[str] = []
    failures.extend(_audit_csv(input_dir / "xgboost_train_10k.csv", min_rows=args.target, eval_keys=eval_keys))
    kobert_candidates = input_dir / "kobert_candidates_10k.csv"
    kobert_text = input_dir / "kobert_text_train_10k.csv"
    failures.extend(_audit_csv(kobert_candidates, min_rows=args.target, eval_keys=eval_keys))
    failures.extend(
        _audit_csv(
            kobert_text,
            min_rows=args.target,
            eval_keys=eval_keys,
            require_text=True,
        )
    )
    failures.extend(_audit_matching_keysets("kobert_candidates", kobert_candidates, "kobert_text", kobert_text))

    _, gnn_failures = _audit_keyset(
        "gnn_existing",
        Path(args.gnn_existing),
        min_unique_rows=args.target,
        eval_keys=eval_keys,
        allow_duplicate_keys=True,
    )
    failures.extend(gnn_failures)

    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print("PASS engine training inputs audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
