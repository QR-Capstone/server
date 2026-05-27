#!/usr/bin/env python3
"""Sweep URLML thresholds on labeled CSVs."""

from __future__ import annotations

import argparse
import csv
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ML_DIR = os.path.join(BASE, "url_ml")
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)

from train_url_ml import canonical_url_key
from url_ml_engine import load_url_ml_model, predict_url_ml


def read_rows(paths: list[str]) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                key = canonical_url_key(url)
                if url and key and label in {"0", "1"} and key not in seen:
                    seen.add(key)
                    rows.append((url, int(label)))
    return rows


def classify(score: float, danger_threshold: float, unknown_threshold: float) -> str:
    if score >= danger_threshold:
        return "malicious"
    if score >= unknown_threshold:
        return "unknown"
    return "benign"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--danger", type=float, nargs="*", default=[0.45, 0.50, 0.55, 0.60])
    parser.add_argument("--unknown", type=float, nargs="*", default=[0.20, 0.25, 0.30, 0.35])
    args = parser.parse_args()

    rows = read_rows(args.input)
    model, status = load_url_ml_model()
    if model is None:
        raise RuntimeError(status.reason)

    scored: list[tuple[int, float, str]] = []
    for url, label in rows:
        out = predict_url_ml(model, url)
        score = float(out.get("raw_probability") if out.get("raw_probability") is not None else out.get("probability") or 0.0)
        rule_pred = str(out.get("verdict") or "unknown") if out.get("adjusted_by_rule") else ""
        scored.append((label, score, rule_pred))

    print(f"rows={len(scored)} malicious={sum(1 for label, _, _ in scored if label == 1)} benign={sum(1 for label, _, _ in scored if label == 0)}")
    for danger in args.danger:
        for unknown in args.unknown:
            if unknown >= danger:
                continue
            tp = tn = fp = fn = unk = 0
            for label, score, rule_pred in scored:
                pred = rule_pred or classify(score, danger, unknown)
                if pred == "unknown":
                    unk += 1
                elif label == 1 and pred == "malicious":
                    tp += 1
                elif label == 0 and pred == "benign":
                    tn += 1
                elif label == 0 and pred == "malicious":
                    fp += 1
                elif label == 1 and pred == "benign":
                    fn += 1
            acc_all = (tp + tn) / max(1, len(scored))
            decisive = tp + tn + fp + fn
            acc_decisive = (tp + tn) / max(1, decisive)
            benign_total = max(1, sum(1 for label, _, _ in scored if label == 0))
            malicious_total = max(1, sum(1 for label, _, _ in scored if label == 1))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            print(
                f"danger={danger:.2f} unknown={unknown:.2f} "
                f"accuracy={acc_all:.4f} precision={precision:.4f} recall={recall:.4f} "
                f"fp_rate={fp / benign_total:.4f} fn_rate={fn / malicious_total:.4f} "
                f"unknown_rate={unk / max(1, len(scored)):.4f} "
                f"acc_decisive={acc_decisive:.4f} "
                f"coverage={decisive / max(1, len(scored)):.4f} "
                f"TP={tp} TN={tn} FP={fp} FN={fn} unknown={unk}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
