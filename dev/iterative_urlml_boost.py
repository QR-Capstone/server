#!/usr/bin/env python3
"""Iteratively collect unique live URLs, evaluate, then train URLML on them.

This intentionally follows the requested loop:
  collect 300 malicious + 300 benign live URLs with no duplicates,
  evaluate current fast detectors on that batch,
  add the batch to the training pool,
  retrain URLML,
  repeat.

Use this for aggressive local tuning, not as an unbiased final benchmark.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from typing import Iterable


PYTHON = sys.executable
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(BASE, "dev")


BASE_EXCLUDE_INPUTS = [
    os.path.join(DEV, "train_urls_all_merged.csv"),
    os.path.join(DEV, "train_urls_all_live_20260521.csv"),
    os.path.join(DEV, "train_urls_all_holdout_20260521.csv"),
    os.path.join(DEV, "train_urls_all_holdout2_20260521.csv"),
    os.path.join(BASE, "gnn", "gnn_total_dataset.csv"),
]


def read_rows(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_rows(path: str, rows: list[dict[str, str]]) -> None:
    fieldnames = ["url", "label", "source", "is_korean"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def merge_csvs(paths: Iterable[str], out_path: str) -> int:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_rows(path):
            url = (row.get("url") or "").strip()
            label = str(row.get("label", "")).strip()
            if not url or label not in {"0", "1"} or url in seen:
                continue
            seen.add(url)
            rows.append(
                {
                    "url": url,
                    "label": label,
                    "source": row.get("source", "iterative"),
                    "is_korean": row.get("is_korean", ""),
                }
            )
    write_rows(out_path, rows)
    return len(rows)


def run(cmd: list[str], desc: str) -> None:
    print(f"\n{'=' * 80}\n{desc}\n{' '.join(cmd)}\n{'=' * 80}", flush=True)
    proc = subprocess.run(cmd, cwd=BASE)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def count_labels(path: str) -> tuple[int, int]:
    malicious = benign = 0
    for row in read_rows(path):
        label = str(row.get("label", "")).strip()
        if label == "1":
            malicious += 1
        elif label == "0":
            benign += 1
    return malicious, benign


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--per-label", type=int, default=300)
    parser.add_argument("--workers", type=int, default=35)
    parser.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    iterative_paths: list[str] = []
    cumulative_train = os.path.join(DEV, f"urlml_iterative_train_{args.tag}.csv")

    for round_no in range(1, args.rounds + 1):
        suffix = f"_iter_{args.tag}_r{round_no}"
        exclude_path = os.path.join(DEV, f"exclude_iter_{args.tag}_r{round_no}.csv")
        all_path = os.path.join(DEV, f"train_urls_all{suffix}.csv")

        merge_csvs([*BASE_EXCLUDE_INPUTS, *iterative_paths], exclude_path)

        run(
            [
                PYTHON,
                os.path.join(DEV, "collect_urls.py"),
                "--malicious",
                str(args.per_label),
                "--benign",
                str(args.per_label),
                "--workers",
                str(args.workers),
                "--out-dir",
                DEV,
                "--suffix",
                suffix,
                "--exclude-csv",
                exclude_path,
            ],
            f"Round {round_no}/{args.rounds}: collect unique live URLs",
        )

        mal, ben = count_labels(all_path)
        print(f"[round {round_no}] collected malicious={mal} benign={ben}", flush=True)
        iterative_paths.append(all_path)

        run(
            [
                PYTHON,
                os.path.join(DEV, "evaluate_live_csv.py"),
                "--input",
                all_path,
                "--rules-only",
            ],
            f"Round {round_no}/{args.rounds}: pre-train fast detector stats",
        )

        merge_csvs(
            [
                os.path.join(DEV, "train_urls_all_merged.csv"),
                os.path.join(DEV, "train_urls_all_live_20260521.csv"),
                *iterative_paths,
            ],
            cumulative_train,
        )

        run(
            [
                PYTHON,
                os.path.join(BASE, "url_ml", "train_url_ml.py"),
                "--input",
                cumulative_train,
            ],
            f"Round {round_no}/{args.rounds}: retrain URLML on cumulative data",
        )

    print(f"\nDone. cumulative_train={cumulative_train}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
