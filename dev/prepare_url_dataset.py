#!/usr/bin/env python3
"""Build canonical, fixed URL dataset splits for training and evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Iterable

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ML_DIR = os.path.join(BASE, "url_ml")
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)

from train_url_ml import canonical_url_key


SOURCE_TYPES = {
    "malicious_real_live",
    "malicious_real_dead",
    "malicious_synthetic_kr",
    "benign_major_official",
    "benign_hard_korean_smb",
    "benign_ad_landing",
    "benign_user_confirmed_fp",
}


def infer_source(label: str, source: str) -> str:
    src = (source or "").strip().lower()
    if src in SOURCE_TYPES:
        return src
    if label == "1":
        if "synthetic" in src:
            return "malicious_synthetic_kr"
        if "dead" in src:
            return "malicious_real_dead"
        return "malicious_real_live"
    if "user" in src or "manual" in src or "fp" in src:
        return "benign_user_confirmed_fp"
    if "naver" in src or "landing" in src or "ad_" in src:
        return "benign_ad_landing"
    if any(token in src for token in ("hard", "smb", "korean", "kr_benign")):
        return "benign_hard_korean_smb"
    return "benign_major_official"


def read_rows(paths: Iterable[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    by_key: dict[str, dict[str, str]] = {}
    labels_by_key: defaultdict[str, set[str]] = defaultdict(set)
    raw_by_key: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                key = canonical_url_key(url)
                if not url or label not in {"0", "1"} or not key:
                    continue
                source = infer_source(label, row.get("source") or "")
                out = {
                    "url": url,
                    "label": label,
                    "source": source,
                    "is_korean": str(row.get("is_korean") or "").strip(),
                    "canonical_key": key,
                }
                labels_by_key[key].add(label)
                raw_by_key[key].append(out)
                by_key.setdefault(key, out)

    conflicts: list[dict[str, str]] = []
    for key, labels in labels_by_key.items():
        if len(labels) <= 1:
            continue
        for row in raw_by_key[key]:
            conflicts.append(row)
        by_key.pop(key, None)
    rows = list(by_key.values())
    rows.sort(key=lambda r: r["canonical_key"])
    return rows, conflicts


def split_name(key: str, seed: str, val_frac: float, test_frac: float) -> str:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) / float(16**12)
    if bucket < test_frac:
        return "test"
    if bucket < test_frac + val_frac:
        return "validation"
    return "train"


def write_csv(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ["url", "label", "source", "is_korean", "canonical_key", "split"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def balanced(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_label = {
        "0": [r for r in rows if r["label"] == "0"],
        "1": [r for r in rows if r["label"] == "1"],
    }
    n = min(len(by_label["0"]), len(by_label["1"]))
    out = by_label["0"][:n] + by_label["1"][:n]
    out.sort(key=lambda r: r["canonical_key"])
    return out


def operational(rows: list[dict[str, str]], benign_per_malicious: int) -> list[dict[str, str]]:
    benign = [r for r in rows if r["label"] == "0"]
    malicious = [r for r in rows if r["label"] == "1" and r["source"] != "malicious_synthetic_kr"]
    if not benign:
        return malicious
    max_mal = max(1, len(benign) // max(1, benign_per_malicious))
    out = benign + malicious[:max_mal]
    out.sort(key=lambda r: r["canonical_key"])
    return out


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    return {
        "rows": len(rows),
        "labels": dict(Counter(r["label"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "splits": dict(Counter(r.get("split", "") for r in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="CSV with url,label,source. Repeatable.")
    parser.add_argument("--out-dir", default=os.path.join(BASE, "dev", "dataset_splits"))
    parser.add_argument("--seed", default="url-split-v1")
    parser.add_argument("--validation-frac", type=float, default=0.10)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--operational-benign-per-malicious", type=int, default=9)
    args = parser.parse_args()

    rows, conflicts = read_rows(args.input)
    if not rows:
        raise SystemExit("no usable rows")
    for row in rows:
        split = split_name(row["canonical_key"], args.seed, args.validation_frac, args.test_frac)
        if split == "test" and row["source"] == "malicious_synthetic_kr":
            split = "train"
        row["split"] = split

    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    test = [r for r in rows if r["split"] == "test"]
    test_balanced = balanced(test)
    test_operational = operational(test, args.operational_benign_per_malicious)

    write_csv(os.path.join(args.out_dir, "warehouse.csv"), rows)
    write_csv(os.path.join(args.out_dir, "train.csv"), train)
    write_csv(os.path.join(args.out_dir, "validation.csv"), validation)
    write_csv(os.path.join(args.out_dir, "test_balanced.csv"), test_balanced)
    write_csv(os.path.join(args.out_dir, "test_operational.csv"), test_operational)
    write_csv(os.path.join(args.out_dir, "label_conflicts.csv"), conflicts)

    summary = {
        "seed": args.seed,
        "inputs": args.input,
        "warehouse": summarize(rows),
        "train": summarize(train),
        "validation": summarize(validation),
        "test_balanced": summarize(test_balanced),
        "test_operational": summarize(test_operational),
        "label_conflicts": len(conflicts),
        "source_types": sorted(SOURCE_TYPES),
    }
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if conflicts:
        print(f"FAIL label_conflicts={len(conflicts)} wrote={os.path.join(args.out_dir, 'label_conflicts.csv')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
