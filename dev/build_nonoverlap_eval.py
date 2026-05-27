#!/usr/bin/env python3
"""Build a balanced evaluation CSV excluding reference canonical URLs."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
URL_ML_DIR = BASE / "url_ml"
if str(URL_ML_DIR) not in sys.path:
    sys.path.insert(0, str(URL_ML_DIR))

from train_url_ml import canonical_url_key


def read_reference_keys(paths: list[str]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = canonical_url_key(row.get("url") or "")
                if key:
                    keys.add(key)
    return keys


def infer_source(label: str, source: str) -> str:
    src = (source or "").strip()
    low = src.lower()
    if label == "1":
        return "malicious_synthetic_kr" if "synthetic" in low else "malicious_real_live"
    if "user" in low or "manual" in low or "fp" in low:
        return "benign_user_confirmed_fp"
    if "naver" in low or "landing" in low or "ad_" in low:
        return "benign_ad_landing"
    if any(token in low for token in ("korean", "kr_benign", "smb", "hard")):
        return "benign_hard_korean_smb"
    return "benign_major_official"


def iter_candidate_paths(root: str, explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    paths = []
    for path in sorted(Path(root).glob("*.csv")):
        name = path.name.lower()
        if name.startswith("exclude_") or name.startswith("train_"):
            continue
        if name in {"warehouse.csv", "train.csv", "validation.csv", "test_balanced.csv", "test_operational.csv"}:
            continue
        paths.append(str(path))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--input-dir", default=str(BASE / "dev"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-label", type=int, default=100)
    args = parser.parse_args()

    reference_keys = read_reference_keys(args.reference)
    candidates_by_key: dict[str, dict[str, str]] = {}
    labels_by_key: defaultdict[str, set[str]] = defaultdict(set)
    source_counts = Counter()
    path_counts = Counter()

    for path in iter_candidate_paths(args.input_dir, args.input):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "url" not in reader.fieldnames or "label" not in reader.fieldnames:
                continue
            for row in reader:
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                key = canonical_url_key(url)
                if not url or label not in {"0", "1"} or not key or key in reference_keys:
                    continue
                source = infer_source(label, row.get("source") or Path(path).stem)
                labels_by_key[key].add(label)
                candidates_by_key.setdefault(
                    key,
                    {
                        "url": url,
                        "label": label,
                        "source": source,
                        "is_korean": str(row.get("is_korean") or "").strip(),
                        "canonical_key": key,
                    },
                )
                source_counts[source] += 1
                path_counts[Path(path).name] += 1

    rows = [row for key, row in candidates_by_key.items() if len(labels_by_key[key]) == 1]
    rows.sort(key=lambda row: (row["source"], row["canonical_key"]))
    by_label = {
        "0": [row for row in rows if row["label"] == "0"],
        "1": [row for row in rows if row["label"] == "1" and row["source"] != "malicious_synthetic_kr"],
    }
    n = min(args.per_label, len(by_label["0"]), len(by_label["1"]))
    out_rows = by_label["0"][:n] + by_label["1"][:n]
    out_rows.sort(key=lambda row: row["canonical_key"])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fieldnames = ["url", "label", "source", "is_korean", "canonical_key"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{name: row.get(name, "") for name in fieldnames} for row in out_rows])

    print(f"reference_unique={len(reference_keys)}")
    print(f"candidate_unique={len(candidates_by_key)} usable_unique={len(rows)}")
    print(f"available labels={dict((label, len(values)) for label, values in by_label.items())}")
    print(f"wrote={args.out} rows={len(out_rows)} per_label={n}")
    print(f"sources={dict(Counter(row['source'] for row in out_rows))}")
    print(f"top_input_files={dict(path_counts.most_common(10))}")
    if len(out_rows) < args.per_label * 2:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
