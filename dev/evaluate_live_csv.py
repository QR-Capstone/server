#!/usr/bin/env python3
"""Evaluate live URL CSVs without starting FastAPI.

The input CSV must contain url,label where label is 0=benign, 1=malicious.
XGBoost is always evaluated. GNN is optional because it fetches pages and can be slow.
KoBERT is intentionally not loaded here; use batch_test_local.py for full server evaluation.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import os
import sys
from collections import Counter
from typing import Any

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XG_DIR = os.path.join(BASE, "xgboost")
GNN_DIR = os.path.join(BASE, "gnn")
URL_ML_DIR = os.path.join(BASE, "url_ml")
for path in (BASE, XG_DIR, GNN_DIR, URL_ML_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from trusted_domains import is_trusted_official_url, strong_url_phishing_score, url_heuristic_phishing_score
from XG_core import load_bundle, predict_url, predict_url_dom, xgboost_weighted_ensemble_verdict
from url_ml_engine import load_url_ml_model, predict_url_ml


def read_rows(path: str, limit: int | None = None) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            url = (row.get("url") or "").strip()
            label = str(row.get("label", "")).strip()
            if url and label in {"0", "1"}:
                rows.append((url, int(label)))
                if limit and len(rows) >= limit:
                    break
    return rows


def apply_url_rule(url: str, pred: str, score: float) -> tuple[str, float, str | None]:
    if is_trusted_official_url(url):
        return "benign", 0.0, "trusted_official"
    strong = float(strong_url_phishing_score(url))
    if strong >= 0.66:
        return "malicious", strong, "strong_url_rule"
    return pred, score, None


def metrics(pairs: list[tuple[int, str]]) -> dict[str, float]:
    tp = sum(1 for label, pred in pairs if label == 1 and pred == "malicious")
    tn = sum(1 for label, pred in pairs if label == 0 and pred == "benign")
    fp = sum(1 for label, pred in pairs if label == 0 and pred == "malicious")
    fn = sum(1 for label, pred in pairs if label == 1 and pred == "benign")
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "accuracy": (tp + tn) / max(1, len(pairs)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(name: str, pairs: list[tuple[int, str]], misses: list[tuple[str, int, str, float, str | None]]) -> None:
    m = metrics(pairs)
    print(
        f"{name}: accuracy={m['accuracy']:.3f} precision={m['precision']:.3f} "
        f"recall={m['recall']:.3f} f1={m['f1']:.3f} "
        f"TP={int(m['tp'])} TN={int(m['tn'])} FP={int(m['fp'])} FN={int(m['fn'])}"
    )
    for url, label, pred, score, rule in misses[:20]:
        print(f"  MISS label={label} pred={pred} score={score:.3f} rule={rule or '-'} url={url}")


def evaluate_xgboost(rows: list[tuple[str, int]]) -> tuple[list[tuple[int, str]], list[tuple[str, int, str, float, str | None]]]:
    typo = load_bundle(os.path.join(XG_DIR, "url_xgb_paired_first.joblib"))
    domain = load_bundle(os.path.join(XG_DIR, "url_xgb_domain_age.joblib"))
    dom = load_bundle(os.path.join(XG_DIR, "url_xgb_dom.joblib"))
    pairs: list[tuple[int, str]] = []
    misses: list[tuple[str, int, str, float, str | None]] = []
    for i, (url, label) in enumerate(rows, 1):
        with contextlib.redirect_stdout(io.StringIO()):
            _, p_typo, _ = predict_url(
                typo,
                url,
                enable_domain_age=False,
                enable_ssl=bool(typo.meta.get("enable_ssl", False)),
                domain_only=False,
            )
            _, p_domain, _ = predict_url(
                domain,
                url,
                enable_domain_age=True,
                enable_ssl=bool(domain.meta.get("enable_ssl", False)),
                domain_only=True,
            )
            _, p_dom, _ = predict_url_dom(dom, url, print_dom_feature_debug=False)
        score, verdict_label = xgboost_weighted_ensemble_verdict(p_typo, p_domain, p_dom)
        pred = "malicious" if verdict_label == 1 else "benign"
        pred, score, rule = apply_url_rule(url, pred, float(score))
        pairs.append((label, pred))
        if (label == 1 and pred != "malicious") or (label == 0 and pred != "benign"):
            misses.append((url, label, pred, score, rule))
        if i % 25 == 0 or i == len(rows):
            print(f"  XGBoost [{i}/{len(rows)}]", flush=True)
    return pairs, misses


def evaluate_url_rules(rows: list[tuple[str, int]]) -> tuple[list[tuple[int, str]], list[tuple[str, int, str, float, str | None]]]:
    pairs: list[tuple[int, str]] = []
    misses: list[tuple[str, int, str, float, str | None]] = []
    for url, label in rows:
        pred, score, rule = apply_url_rule(url, "benign", 0.0)
        pairs.append((label, pred))
        if (label == 1 and pred != "malicious") or (label == 0 and pred != "benign"):
            misses.append((url, label, pred, score, rule))
    return pairs, misses


def evaluate_url_heuristic(rows: list[tuple[str, int]]) -> tuple[list[tuple[int, str]], list[tuple[str, int, str, float, str | None]]]:
    pairs: list[tuple[int, str]] = []
    misses: list[tuple[str, int, str, float, str | None]] = []
    for url, label in rows:
        score = float(url_heuristic_phishing_score(url))
        pred = "malicious" if score >= 0.40 else "benign"
        pairs.append((label, pred))
        if (label == 1 and pred != "malicious") or (label == 0 and pred != "benign"):
            misses.append((url, label, pred, score, "heuristic"))
    return pairs, misses


def evaluate_gnn(rows: list[tuple[str, int]]) -> tuple[list[tuple[int, str]], list[tuple[str, int, str, float, str | None]]]:
    from gnn_engine import GNN_Engine, predict_gnn

    engine = GNN_Engine()
    if not engine.ok:
        raise RuntimeError(engine._load_error or "GNN model not loaded")
    pairs: list[tuple[int, str]] = []
    misses: list[tuple[str, int, str, float, str | None]] = []
    for i, (url, label) in enumerate(rows, 1):
        out = predict_gnn(engine.model, engine.columns, url)
        pred = out.get("verdict", "unknown")
        score = float(out.get("probability", 0.0))
        pred, score, rule = apply_url_rule(url, pred, score)
        if pred in {"malicious", "benign"}:
            pairs.append((label, pred))
            if (label == 1 and pred != "malicious") or (label == 0 and pred != "benign"):
                misses.append((url, label, pred, score, rule))
        if i % 25 == 0 or i == len(rows):
            print(f"  GNN [{i}/{len(rows)}]", flush=True)
    return pairs, misses


def evaluate_url_ml(rows: list[tuple[str, int]]) -> tuple[list[tuple[int, str]], list[tuple[str, int, str, float, str | None]]]:
    model, status = load_url_ml_model()
    if model is None:
        raise RuntimeError(status.reason)
    pairs: list[tuple[int, str]] = []
    misses: list[tuple[str, int, str, float, str | None]] = []
    for url, label in rows:
        out = predict_url_ml(model, url)
        pred = out.get("verdict", "unknown")
        score = float(out.get("probability") or 0.0)
        if pred == "unknown":
            pred = "malicious" if score >= 0.40 else "benign"
        pairs.append((label, pred))
        if (label == 1 and pred != "malicious") or (label == 0 and pred != "benign"):
            misses.append((url, label, pred, score, "url_ml"))
    return pairs, misses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--with-gnn", action="store_true")
    parser.add_argument("--rules-only", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.input, limit=args.limit or None)
    label_counts = Counter(label for _, label in rows)
    print(f"input={args.input} rows={len(rows)} malicious={label_counts[1]} benign={label_counts[0]}")

    rule_pairs, rule_misses = evaluate_url_rules(rows)
    print_metrics("URL-rules-only", rule_pairs, rule_misses)
    heuristic_pairs, heuristic_misses = evaluate_url_heuristic(rows)
    print_metrics("URLHeuristic@0.40", heuristic_pairs, heuristic_misses)
    url_ml_pairs, url_ml_misses = evaluate_url_ml(rows)
    print_metrics("URLML", url_ml_pairs, url_ml_misses)

    if args.rules_only:
        return 0

    xg_pairs, xg_misses = evaluate_xgboost(rows)
    print_metrics("XGBoost+rules", xg_pairs, xg_misses)

    if args.with_gnn:
        gnn_pairs, gnn_misses = evaluate_gnn(rows)
        print_metrics("GNN+rules", gnn_pairs, gnn_misses)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
