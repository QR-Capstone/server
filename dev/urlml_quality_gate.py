#!/usr/bin/env python3
"""Fail-fast quality gate for URLML data splits and threshold behavior."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DEPS = os.path.join(BASE, ".codex_deps")
if os.path.isdir(LOCAL_DEPS) and LOCAL_DEPS not in sys.path:
    sys.path.insert(0, LOCAL_DEPS)
URL_ML_DIR = os.path.join(BASE, "url_ml")
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)

from train_url_ml import canonical_url_key
from url_ml_engine import load_url_ml_model, predict_url_ml

Z_95_ONE_SIDED = 1.6448536269514722


@dataclass(frozen=True)
class Row:
    url: str
    label: int
    key: str
    source: str


def read_labeled(paths: list[str]) -> list[Row]:
    rows: list[Row] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                key = canonical_url_key(url)
                if url and key and label in {"0", "1"} and key not in seen:
                    seen.add(key)
                    rows.append(
                        Row(
                            url=url,
                            label=int(label),
                            key=key,
                            source=(row.get("source") or "unknown_source").strip() or "unknown_source",
                        )
                    )
    return rows


def read_keys(paths: list[str]) -> set[str]:
    return {row.key for row in read_labeled(paths)}


def wilson_lower_bound(successes: int, total: int, z: float = Z_95_ONE_SIDED) -> float:
    if total <= 0:
        return 0.0
    p_hat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = p_hat + z2 / (2.0 * total)
    margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total) ** 0.5
    return max(0.0, (center - margin) / denominator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", default=[])
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--min-coverage", type=float, default=0.99)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--max-fp-rate", type=float, default=1.0)
    parser.add_argument("--max-fn-rate", type=float, default=1.0)
    parser.add_argument("--max-unknown-rate", type=float, default=1.0)
    parser.add_argument("--min-decisive-accuracy", type=float, default=1.0)
    parser.add_argument("--min-accuracy-lower-bound", type=float, default=0.0)
    parser.add_argument("--min-recall-lower-bound", type=float, default=0.0)
    parser.add_argument("--min-specificity-lower-bound", type=float, default=0.0)
    parser.add_argument("--max-fp", type=int, default=0)
    parser.add_argument("--max-fn", type=int, default=0)
    args = parser.parse_args()

    eval_rows = read_labeled(args.eval)
    train_keys = read_keys(args.train)
    overlap = sorted(row.key for row in eval_rows if row.key in train_keys)
    if overlap:
        print(f"FAIL overlap={len(overlap)}")
        for key in overlap[:20]:
            print(f"  {key}")
        return 1

    model, status = load_url_ml_model()
    if model is None:
        raise RuntimeError(status.reason)

    tp = tn = fp = fn = unknown = 0
    misses: list[str] = []
    unknown_rows: list[str] = []
    source_stats: dict[str, dict[str, int]] = {}
    for row in eval_rows:
        stats = source_stats.setdefault(
            row.source,
            {"rows": 0, "malicious": 0, "benign": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "unknown": 0},
        )
        stats["rows"] += 1
        stats["malicious" if row.label else "benign"] += 1
        out = predict_url_ml(model, row.url)
        pred = str(out.get("verdict") or "unknown")
        if pred == "unknown":
            unknown += 1
            stats["unknown"] += 1
            unknown_rows.append(
                f"UNKNOWN label={row.label} raw={out.get('raw_probability')} "
                f"ml={out.get('ml_probability')} heuristic={out.get('heuristic_probability')} {row.url}"
            )
        elif row.label == 1 and pred == "malicious":
            tp += 1
            stats["tp"] += 1
        elif row.label == 0 and pred == "benign":
            tn += 1
            stats["tn"] += 1
        elif row.label == 0 and pred == "malicious":
            fp += 1
            stats["fp"] += 1
            misses.append(f"FP {row.url}")
        elif row.label == 1 and pred == "benign":
            fn += 1
            stats["fn"] += 1
            misses.append(f"FN {row.url}")

    total = max(1, len(eval_rows))
    benign_total = max(1, sum(1 for row in eval_rows if row.label == 0))
    malicious_total = max(1, sum(1 for row in eval_rows if row.label == 1))
    decisive = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    accuracy_lower_95 = wilson_lower_bound(tp + tn, total)
    recall_lower_95 = wilson_lower_bound(tp, malicious_total)
    specificity_lower_95 = wilson_lower_bound(tn, benign_total)
    coverage = decisive / total
    decisive_accuracy = (tp + tn) / max(1, decisive)
    fp_rate = fp / benign_total
    fn_rate = fn / malicious_total
    unknown_rate = unknown / total
    print(
        f"rows={len(eval_rows)} accuracy={accuracy:.4f} coverage={coverage:.4f} "
        f"accuracy_lower_95={accuracy_lower_95:.6f} "
        f"recall_lower_95={recall_lower_95:.6f} "
        f"specificity_lower_95={specificity_lower_95:.6f} "
        f"decisive_accuracy={decisive_accuracy:.4f} TP={tp} TN={tn} "
        f"FP={fp} FP_rate={fp_rate:.4f} FN={fn} FN_rate={fn_rate:.4f} "
        f"unknown={unknown} unknown_rate={unknown_rate:.4f}"
    )
    for miss in misses[:20]:
        print(f"  {miss}")
    for row in unknown_rows[:20]:
        print(f"  {row}")
    for source, stats in sorted(source_stats.items(), key=lambda item: (-item[1]["rows"], item[0])):
        print(
            f"  SOURCE {source}: rows={stats['rows']} malicious={stats['malicious']} benign={stats['benign']} "
            f"TP={stats['tp']} TN={stats['tn']} FP={stats['fp']} FN={stats['fn']} unknown={stats['unknown']}"
        )

    if coverage < args.min_coverage:
        print(f"FAIL coverage {coverage:.4f} < {args.min_coverage:.4f}")
        return 1
    if accuracy < args.min_accuracy:
        print(f"FAIL accuracy {accuracy:.4f} < {args.min_accuracy:.4f}")
        return 1
    if accuracy_lower_95 < args.min_accuracy_lower_bound:
        print(
            f"FAIL accuracy_lower_95 {accuracy_lower_95:.6f} "
            f"< {args.min_accuracy_lower_bound:.6f}"
        )
        return 1
    if recall_lower_95 < args.min_recall_lower_bound:
        print(
            f"FAIL recall_lower_95 {recall_lower_95:.6f} "
            f"< {args.min_recall_lower_bound:.6f}"
        )
        return 1
    if specificity_lower_95 < args.min_specificity_lower_bound:
        print(
            f"FAIL specificity_lower_95 {specificity_lower_95:.6f} "
            f"< {args.min_specificity_lower_bound:.6f}"
        )
        return 1
    if fp_rate > args.max_fp_rate:
        print(f"FAIL fp_rate {fp_rate:.4f} > {args.max_fp_rate:.4f}")
        return 1
    if fn_rate > args.max_fn_rate:
        print(f"FAIL fn_rate {fn_rate:.4f} > {args.max_fn_rate:.4f}")
        return 1
    if unknown_rate > args.max_unknown_rate:
        print(f"FAIL unknown_rate {unknown_rate:.4f} > {args.max_unknown_rate:.4f}")
        return 1
    if decisive_accuracy < args.min_decisive_accuracy:
        print(f"FAIL decisive_accuracy {decisive_accuracy:.4f} < {args.min_decisive_accuracy:.4f}")
        return 1
    if fp > args.max_fp:
        print(f"FAIL fp {fp} > {args.max_fp}")
        return 1
    if fn > args.max_fn:
        print(f"FAIL fn {fn} > {args.max_fn}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
