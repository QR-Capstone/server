#!/usr/bin/env python3
"""Compare URLML holdout metrics with eval-host rules enabled vs disabled."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Any


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ML_DIR = os.path.join(BASE, "url_ml")
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from url_ml_engine import load_url_ml_model, predict_url_ml_batch  # noqa: E402


@dataclass
class Metrics:
    rows: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    unknown: int = 0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / max(1, self.rows)

    @property
    def coverage(self) -> float:
        return (self.tp + self.tn + self.fp + self.fn) / max(1, self.rows)


def clear_url_caches() -> None:
    import trusted_domains  # noqa: WPS433

    for name in (
        "hostname_from_url",
        "is_trusted_official_url",
        "is_low_risk_hosted_platform_url",
        "strong_url_phishing_score",
        "url_heuristic_phishing_score",
    ):
        fn = getattr(trusted_domains, name, None)
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


def risk_from_result(result: dict[str, Any]) -> str:
    risk = str(result.get("riskLevel") or result.get("verdict") or "").strip().lower()
    if risk in {"dangerous", "malicious", "1"}:
        return "malicious"
    if risk in {"safe", "benign", "0"}:
        return "benign"
    return "unknown"


def read_rows(paths: list[str]) -> list[tuple[str, int, str, str]]:
    rows: list[tuple[str, int, str, str]] = []
    for path in paths:
        seen: set[str] = set()
        eval_name = os.path.relpath(path, BASE).replace("\\", "/")
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                if not url or label not in {"0", "1"} or url in seen:
                    continue
                seen.add(url)
                rows.append((url, int(label), (row.get("source") or "unknown_source").strip(), eval_name))
    return rows


def _add_prediction(metrics: Metrics, label: int, pred: str) -> None:
    metrics.rows += 1
    if pred == "unknown":
        metrics.unknown += 1
    elif label == 1 and pred == "malicious":
        metrics.tp += 1
    elif label == 0 and pred == "benign":
        metrics.tn += 1
    elif label == 0 and pred == "malicious":
        metrics.fp += 1
    elif label == 1 and pred == "benign":
        metrics.fn += 1
    else:
        metrics.unknown += 1


def evaluate(
    model: Any,
    rows: list[tuple[str, int, str, str]],
    enabled: bool,
    batch_size: int,
) -> tuple[Metrics, dict[str, Metrics], dict[str, Metrics]]:
    os.environ["TRUSTED_DOMAIN_EVAL_RULES"] = "1" if enabled else "0"
    clear_url_caches()

    overall = Metrics()
    by_source: dict[str, Metrics] = {}
    by_eval: dict[str, Metrics] = {}
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        predictions = predict_url_ml_batch(model, [url for url, _, _, _ in chunk])
        for (_, label, source, eval_name), result in zip(chunk, predictions):
            pred = risk_from_result(result)
            for metrics in (overall, by_source.setdefault(source, Metrics()), by_eval.setdefault(eval_name, Metrics())):
                _add_prediction(metrics, label, pred)
    return overall, by_source, by_eval


def print_metrics(label: str, metrics: Metrics) -> None:
    print(
        f"{label}: rows={metrics.rows} accuracy={metrics.accuracy:.4f} "
        f"coverage={metrics.coverage:.4f} TP={metrics.tp} TN={metrics.tn} "
        f"FP={metrics.fp} FN={metrics.fn} unknown={metrics.unknown}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="append", required=True)
    parser.add_argument("--max-accuracy-drop", type=float, default=1.0)
    parser.add_argument("--max-new-fp", type=int, default=-1)
    parser.add_argument("--max-new-fn", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    rows = read_rows(args.eval)
    if not rows:
        raise SystemExit("no labeled rows")

    model, status = load_url_ml_model()
    if model is None:
        raise SystemExit(f"URLML model unavailable: {status.reason}")

    on, on_sources, on_evals = evaluate(model, rows, enabled=True, batch_size=max(1, args.batch_size))
    off, off_sources, off_evals = evaluate(model, rows, enabled=False, batch_size=max(1, args.batch_size))
    print_metrics("eval_rules_on", on)
    print_metrics("eval_rules_off", off)
    print(
        f"delta: accuracy_drop={on.accuracy - off.accuracy:.4f} "
        f"new_fp={off.fp - on.fp} new_fn={off.fn - on.fn} "
        f"new_unknown={off.unknown - on.unknown}"
    )
    for source in sorted(set(on_sources) | set(off_sources)):
        on_metrics = on_sources.get(source, Metrics())
        off_metrics = off_sources.get(source, Metrics())
        print_metrics(f"  on  {source}", on_metrics)
        print_metrics(f"  off {source}", off_metrics)
    for eval_name in sorted(set(on_evals) | set(off_evals)):
        on_metrics = on_evals.get(eval_name, Metrics())
        off_metrics = off_evals.get(eval_name, Metrics())
        print_metrics(f"  eval_on  {eval_name}", on_metrics)
        print_metrics(f"  eval_off {eval_name}", off_metrics)

    failed = False
    for eval_name in sorted(set(on_evals) | set(off_evals)):
        on_metrics = on_evals.get(eval_name, Metrics())
        off_metrics = off_evals.get(eval_name, Metrics())
        accuracy_drop = on_metrics.accuracy - off_metrics.accuracy
        if accuracy_drop > args.max_accuracy_drop:
            print(f"FAIL {eval_name}: accuracy_drop {accuracy_drop:.4f} > {args.max_accuracy_drop:.4f}")
            failed = True
        if args.max_new_fp >= 0 and off_metrics.fp - on_metrics.fp > args.max_new_fp:
            print(f"FAIL {eval_name}: new_fp {off_metrics.fp - on_metrics.fp} > {args.max_new_fp}")
            failed = True
        if args.max_new_fn >= 0 and off_metrics.fn - on_metrics.fn > args.max_new_fn:
            print(f"FAIL {eval_name}: new_fn {off_metrics.fn - on_metrics.fn} > {args.max_new_fn}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
