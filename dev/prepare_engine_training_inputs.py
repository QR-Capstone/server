#!/usr/bin/env python3
"""Prepare per-engine training input CSVs for the 10k-row expansion pass."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit


BASE = Path(__file__).resolve().parents[1]
URL_ML_DIR = BASE / "url_ml"
if str(URL_ML_DIR) not in sys.path:
    sys.path.insert(0, str(URL_ML_DIR))

from train_url_ml import canonical_url_key


class LabelConflictError(ValueError):
    def __init__(self, conflicts: list[str]):
        self.conflicts = conflicts
        super().__init__(f"label conflicts found: {len(conflicts)}")


def canonical_key_variants(url: str) -> set[str]:
    key = canonical_url_key(url)
    variants = {key} if key else set()
    decoded = unquote(url or "")
    decoded_key = canonical_url_key(decoded)
    if decoded_key:
        variants.add(decoded_key)
    return variants


KOREAN_HINTS = (
    ".kr",
    "naver",
    "kakao",
    "daum",
    "coupang",
    "gmarket",
    "11st",
    "tistory",
    "joonggonara",
    "korea",
)


def read_rows(
    paths: list[Path],
    excluded_keys: set[str] | None = None,
    *,
    allow_label_conflicts: bool = False,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_labels: dict[str, str] = {}
    conflicts: list[str] = []
    excluded_keys = excluded_keys or set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                variants = canonical_key_variants(url)
                key = canonical_url_key(url)
                if (
                    not url
                    or label not in {"0", "1"}
                    or not key
                    or any(variant in excluded_keys for variant in variants)
                ):
                    continue
                previous = seen_labels.get(key)
                if previous is not None:
                    if previous != label and len(conflicts) < 20:
                        conflicts.append(f"{key} labels={previous},{label} input={path}")
                    elif previous != label:
                        conflicts.append(f"{key} labels={previous},{label}")
                    continue
                seen_labels[key] = label
                rows.append(
                    {
                        "url": url,
                        "label": label,
                        "source": (row.get("source") or path.stem).strip() or path.stem,
                        "canonical_key": key,
                    }
                )
    if conflicts and not allow_label_conflicts:
        raise LabelConflictError(conflicts)
    return rows


def read_excluded_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                keys.update(canonical_key_variants(row.get("url") or ""))
    return keys


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "canonical_key"])
        writer.writeheader()
        writer.writerows(rows)


def label_limited(rows: list[dict[str, str]], target: int) -> list[dict[str, str]]:
    by_label = {
        "0": [row for row in rows if row["label"] == "0"],
        "1": [row for row in rows if row["label"] == "1"],
    }
    half = target // 2
    selected = by_label["0"][:half] + by_label["1"][:half]
    if len(selected) < target:
        selected_keys = {row["canonical_key"] for row in selected}
        selected.extend(row for row in rows if row["canonical_key"] not in selected_keys)
    selected = selected[:target]
    selected.sort(key=lambda row: row["canonical_key"])
    return selected


def korean_priority(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    def score(row: dict[str, str]) -> tuple[int, str]:
        haystack = f"{row['url']} {row.get('source', '')}".lower()
        hinted = any(hint in haystack for hint in KOREAN_HINTS)
        return (0 if hinted else 1, row["canonical_key"])

    return sorted(rows, key=score)


def existing_keys(path: Path, excluded_keys: set[str] | None = None) -> set[str]:
    if not path.is_file():
        return set()
    excluded_keys = excluded_keys or set()
    return {
        row["canonical_key"]
        for row in read_rows([path], excluded_keys=excluded_keys, allow_label_conflicts=True)
    }


def summarize(name: str, rows: list[dict[str, str]]) -> None:
    labels = Counter(row["label"] for row in rows)
    print(f"{name}: rows={len(rows)} benign={labels.get('0', 0)} malicious={labels.get('1', 0)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--out-dir", default=str(BASE / "dev" / "engine_training_inputs"))
    parser.add_argument("--warehouse", default=str(BASE / "dev" / "dataset_splits" / "warehouse.csv"))
    parser.add_argument(
        "--exclude",
        action="append",
        default=[
            str(BASE / "dev" / "dataset_splits" / "validation.csv"),
            str(BASE / "dev" / "dataset_splits" / "test_balanced.csv"),
            str(BASE / "dev" / "dataset_splits" / "test_operational.csv"),
        ],
        help="CSV whose canonical URL keys must not appear in generated engine training inputs.",
    )
    parser.add_argument("--gnn-existing", default=str(BASE / "gnn" / "gnn_total_dataset.csv"))
    parser.add_argument("--gnn-buffer", type=int, default=50)
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument(
        "--allow-label-conflicts",
        action="store_true",
        help="Ignore later rows whose canonical URL has a conflicting label.",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    source_paths = [Path(args.warehouse), *(Path(path) for path in args.extra)]
    excluded_keys = read_excluded_keys([Path(path) for path in args.exclude])
    try:
        rows = read_rows(
            source_paths,
            excluded_keys=excluded_keys,
            allow_label_conflicts=args.allow_label_conflicts,
        )
    except LabelConflictError as exc:
        for conflict in exc.conflicts[:20]:
            print(f"  LABEL_CONFLICT {conflict}")
        print("FAIL label conflicts found; rerun with --allow-label-conflicts only for intentional experiments")
        return 1
    if len(rows) < args.target:
        raise SystemExit(
            f"not enough unique non-excluded source rows: {len(rows)} < target={args.target} "
            f"(excluded_keys={len(excluded_keys)})"
        )
    print(f"excluded_eval_keys={len(excluded_keys)} source_rows_after_exclusion={len(rows)}")

    xgb_rows = label_limited(rows, args.target)
    write_rows(out_dir / "xgboost_train_10k.csv", xgb_rows)
    summarize("xgboost_train_10k", xgb_rows)

    gnn_existing = Path(args.gnn_existing)
    gnn_keys = existing_keys(gnn_existing, excluded_keys=excluded_keys)
    needed = max(0, args.target - len(gnn_keys))
    gnn_limit = needed + args.gnn_buffer if needed else 0
    gnn_candidates = [row for row in rows if row["canonical_key"] not in gnn_keys][:gnn_limit]
    write_rows(out_dir / "gnn_collect_to_10k.csv", gnn_candidates)
    summarize("gnn_collect_to_10k", gnn_candidates)
    print(f"gnn_existing_non_eval_unique_rows={len(gnn_keys)} gnn_needed_for_target={needed}")

    kobert_rows = korean_priority(rows)[: args.target]
    write_rows(out_dir / "kobert_candidates_10k.csv", kobert_rows)
    summarize("kobert_candidates_10k", kobert_rows)
    print("kobert_note=candidates_only_actual_training_rows_depend_on_korean_text_fetch_filter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
