#!/usr/bin/env python3
"""Estimate current URL-rule impact on a saved batch-test JSON result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from trusted_domains import is_trusted_official_url, strong_url_phishing_score


MODELS = (
    ("XGBoost", "xg_v", "xg"),
    ("GNN", "gn_v", "gn"),
    ("KoBERT", "kb_v", "kb"),
)


def adjusted_prediction(url: str, raw_pred: Any) -> Any:
    if is_trusted_official_url(url):
        return "benign"
    if strong_url_phishing_score(url) >= 0.66:
        return "malicious"
    return raw_pred


def adjusted_score(url: str, raw_score: Any, pred: Any) -> float | None:
    if is_trusted_official_url(url):
        return 0.0
    strong = strong_url_phishing_score(url)
    if strong >= 0.66:
        return strong * 100.0
    try:
        return float(raw_score)
    except (TypeError, ValueError):
        if pred == "malicious":
            return 75.0
        if pred == "benign":
            return 10.0
        return None


def binary_metrics(pairs: list[tuple[str, str]]) -> dict[str, float]:
    tp = sum(1 for exp, pred in pairs if exp == "malicious" and pred == "malicious")
    tn = sum(1 for exp, pred in pairs if exp == "benign" and pred == "benign")
    fp = sum(1 for exp, pred in pairs if exp == "benign" and pred == "malicious")
    fn = sum(1 for exp, pred in pairs if exp == "malicious" and pred == "benign")
    total = max(1, len(pairs))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def final_score_verdict(url: str, row: dict[str, Any]) -> tuple[str, float | None]:
    if is_trusted_official_url(url):
        return "benign", 0.0
    strong = strong_url_phishing_score(url)
    if strong >= 0.66:
        return "malicious", strong * 100.0

    scores = []
    for _, pred_key, score_key in MODELS:
        pred = adjusted_prediction(url, row.get(pred_key))
        score = adjusted_score(url, row.get(score_key), pred)
        if score is not None and pred not in (None, "error", "unavail", "unknown"):
            scores.append(score)
    if not scores:
        return "unknown", None
    avg = sum(scores) / len(scores)
    danger_threshold = float(os.getenv("FINAL_DANGER_THRESHOLD", "0.40")) * 100.0
    unknown_threshold = float(os.getenv("FINAL_UNKNOWN_THRESHOLD", "0.30")) * 100.0
    if avg >= danger_threshold:
        return "malicious", avg
    if avg >= unknown_threshold:
        return "unknown", avg
    return "benign", avg


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join(BASE, "dev", "test_results_local.json"))
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rows = []
    for expected in ("malicious", "benign"):
        for row in data.get(expected, []):
            all_rows.append((expected, row))

    for name, key, score_key in MODELS:
        valid = []
        raw_counter = Counter()
        adjusted_counter = Counter()
        for expected, row in all_rows:
            raw_pred = row.get(key)
            pred = adjusted_prediction(row.get("url", ""), raw_pred)
            raw_counter[raw_pred] += 1
            adjusted_counter[pred] += 1
            if pred not in (None, "error", "unavail", "unknown"):
                valid.append((expected, pred))

        correct = sum(
            1
            for expected, pred in valid
            if (expected == "malicious" and pred == "malicious")
            or (expected == "benign" and pred == "benign")
        )
        total = len(all_rows)
        metrics = binary_metrics(valid)
        acc = metrics["accuracy"] if valid else 0.0
        coverage = len(valid) / total if total else 0.0
        print(
            f"{name}: valid={len(valid)}/{total} accuracy={acc:.3f} "
            f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f} coverage={coverage:.3f} correct_all={correct}/{total}"
        )
        print(f"  raw={dict(raw_counter)}")
        print(f"  adjusted={dict(adjusted_counter)}")

    final_pairs = []
    false_safe = []
    false_danger = []
    final_counter = Counter()
    for expected, row in all_rows:
        verdict, score = final_score_verdict(row.get("url", ""), row)
        final_counter[verdict] += 1
        if verdict != "unknown":
            final_pairs.append((expected, verdict))
        if expected == "malicious" and verdict == "benign":
            false_safe.append((row.get("url", ""), score))
        if expected == "benign" and verdict == "malicious":
            false_danger.append((row.get("url", ""), score))

    final_metrics = binary_metrics(final_pairs)
    total = len(all_rows)
    print("\nFinal(score_weighted_ensemble):")
    print(
        f"  valid={len(final_pairs)}/{total} accuracy={final_metrics['accuracy']:.3f} "
        f"precision={final_metrics['precision']:.3f} recall={final_metrics['recall']:.3f} "
        f"f1={final_metrics['f1']:.3f} coverage={len(final_pairs)/max(1,total):.3f}"
    )
    print(f"  distribution={dict(final_counter)}")
    print(f"  false_safe={len(false_safe)} false_danger={len(false_danger)}")
    for url, score in false_safe[:20]:
        print(f"    FN_SAFE score={score}: {url}")
    for url, score in false_danger[:20]:
        print(f"    FP_DANGER score={score}: {url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
