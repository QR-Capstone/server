#!/usr/bin/env python3
"""Run the production-oriented local regression gates."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRAIN = os.path.join("dev", "dataset_splits", "train.csv")
DEFAULT_HOLDOUT = os.path.join("dev", "dataset_splits", "validation.csv")
DEFAULT_BALANCED_HOLDOUT = os.path.join("dev", "dataset_splits", "test_balanced.csv")
DEFAULT_OPERATIONAL_HOLDOUT = os.path.join("dev", "dataset_splits", "test_operational.csv")


def run(name: str, cmd: list[str]) -> None:
    print(f"\n=== operational gate: {name} ===", flush=True)
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=BASE, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    parser.add_argument("--balanced-holdout", default=DEFAULT_BALANCED_HOLDOUT)
    parser.add_argument("--operational-holdout", default=DEFAULT_OPERATIONAL_HOLDOUT)
    parser.add_argument("--skip-urlml", action="store_true")
    parser.add_argument("--skip-api-performance", action="store_true")
    parser.add_argument("--skip-api-holdout", action="store_true")
    parser.add_argument("--skip-kobert-holdout", action="store_true")
    parser.add_argument("--kobert-limit", type=int, default=80)
    parser.add_argument("--api-holdout-per-label-limit", type=int, default=0)
    parser.add_argument("--include-gnn-fetch-cache", action="store_true")
    args = parser.parse_args()

    if not args.skip_urlml:
        run(
            "urlml leakage, retrain, holdout quality",
            [
                sys.executable,
                os.path.join("dev", "run_urlml_regression_gate.py"),
                "--train",
                args.train,
                "--holdout",
                args.holdout,
                "--min-coverage",
                "0.99",
                "--min-accuracy",
                "0.99",
                "--min-decisive-accuracy",
                "0.99",
                "--max-fp-rate",
                "0.001",
                "--max-fn-rate",
                "0.001",
                "--max-unknown-rate",
                "0.002",
                "--max-fp",
                "0",
                "--max-fn",
                "0",
                "--max-micro-ms",
                "0.75",
            ],
        )

    if not args.skip_api_performance:
        cmd = [
            sys.executable,
            os.path.join("dev", "run_api_performance_gate.py"),
        ]
        if not args.include_gnn_fetch_cache:
            cmd.append("--skip-gnn-fetch")
        run("api latency, cache, readiness", cmd)

    balanced_holdout = args.balanced_holdout if os.path.isfile(os.path.join(BASE, args.balanced_holdout)) else args.holdout
    operational_holdout = (
        args.operational_holdout
        if os.path.isfile(os.path.join(BASE, args.operational_holdout))
        else args.holdout
    )

    if not args.skip_api_holdout:
        sample_args = []
        if args.api_holdout_per_label_limit:
            sample_args = ["--per-label-limit", str(args.api_holdout_per_label_limit)]
        run(
            "api balanced holdout metrics",
            [
                sys.executable,
                os.path.join("dev", "evaluate_api_holdout.py"),
                "--spawn-server",
                "--train",
                args.train,
                "--holdout",
                balanced_holdout,
                "--workers",
                "4",
                "--timeout",
                "90",
                "--require-ready",
                "--show-misses",
                "10",
                *sample_args,
                "--engine",
                "urlml",
                "--engine",
                "xgboost",
                "--engine",
                "gnn",
                "--engine",
                "ensemble",
                "--max-unknown-ratio",
                "1.0",
                "--min-accuracy",
                "0.0",
                "--max-fp-rate",
                "1.0",
                "--max-fn-rate",
                "1.0",
                "--engine-max-unknown-ratio",
                "urlml=0.0",
                "--engine-min-accuracy",
                "urlml=1.0",
                "--engine-max-fn-rate",
                "urlml=0.0",
                "--engine-max-fp-rate",
                "urlml=0.0",
                "--engine-max-unknown-ratio",
                "xgboost=0.0",
                "--engine-min-accuracy",
                "xgboost=0.95",
                "--engine-max-fp-rate",
                "xgboost=0.10",
                "--engine-max-fn-rate",
                "xgboost=0.05",
                "--engine-max-unknown-ratio",
                "gnn=0.0",
                "--engine-min-accuracy",
                "gnn=0.90",
                "--engine-max-fp-rate",
                "gnn=0.10",
                "--engine-max-fn-rate",
                "gnn=0.01",
                "--engine-min-accuracy",
                "ensemble=1.0",
                "--engine-max-fp-rate",
                "ensemble=0.0",
                "--engine-max-fn-rate",
                "ensemble=0.0",
                "--engine-max-unknown-ratio",
                "ensemble=0.0",
                "--engine-min-precision",
                "urlml=0.98",
                "--engine-min-recall",
                "urlml=0.97",
                "--engine-min-precision",
                "xgboost=0.90",
                "--engine-min-recall",
                "xgboost=0.95",
                "--engine-min-precision",
                "gnn=0.90",
                "--engine-min-recall",
                "gnn=1.0",
                "--engine-min-precision",
                "ensemble=1.0",
                "--engine-min-recall",
                "ensemble=1.0",
                "--engine-max-fp",
                "ensemble=0",
                "--engine-max-fn",
                "ensemble=0",
                "--engine-max-p95-ms",
                "ensemble=6000",
            ],
        )
        if not args.skip_kobert_holdout:
            run(
                "api KoBERT sampled balanced holdout metrics",
                [
                    sys.executable,
                    os.path.join("dev", "evaluate_api_holdout.py"),
                    "--spawn-server",
                    "--train",
                    args.train,
                    "--holdout",
                    balanced_holdout,
                    "--engine",
                    "kobert",
                    "--limit",
                    str(args.kobert_limit),
                    "--per-label-limit",
                    str(max(1, args.kobert_limit // 2)),
                    "--workers",
                    "2",
                    "--timeout",
                    "90",
                    "--require-ready",
                    "--show-misses",
                    "10",
                    "--engine-min-precision",
                    "kobert=1.0",
                    "--engine-min-recall",
                    "kobert=1.0",
                    "--engine-max-unknown-ratio",
                    "kobert=0.0",
                    "--engine-max-p95-ms",
                    "kobert=1000",
                ],
            )
        run(
            "api normal-heavy operational holdout metrics",
            [
                sys.executable,
                os.path.join("dev", "evaluate_api_holdout.py"),
                "--spawn-server",
                "--train",
                args.train,
                "--holdout",
                operational_holdout,
                "--workers",
                "4",
                "--timeout",
                "90",
                "--require-ready",
                "--show-misses",
                "10",
                *sample_args,
                "--engine",
                "urlml",
                "--engine",
                "xgboost",
                "--engine",
                "gnn",
                "--engine",
                "ensemble",
                "--max-unknown-ratio",
                "1.0",
                "--min-accuracy",
                "0.0",
                "--max-fp-rate",
                "1.0",
                "--max-fn-rate",
                "1.0",
                "--engine-max-unknown-ratio",
                "urlml=0.0",
                "--engine-min-accuracy",
                "urlml=1.0",
                "--engine-max-fn-rate",
                "urlml=0.0",
                "--engine-max-fp-rate",
                "urlml=0.0",
                "--engine-max-unknown-ratio",
                "xgboost=0.0",
                "--engine-min-accuracy",
                "xgboost=0.95",
                "--engine-max-fp-rate",
                "xgboost=0.10",
                "--engine-max-fn-rate",
                "xgboost=0.05",
                "--engine-max-unknown-ratio",
                "gnn=0.0",
                "--engine-min-accuracy",
                "gnn=0.90",
                "--engine-max-fp-rate",
                "gnn=0.10",
                "--engine-max-fn-rate",
                "gnn=0.01",
                "--engine-min-accuracy",
                "ensemble=1.0",
                "--engine-max-fp-rate",
                "ensemble=0.0",
                "--engine-max-fn-rate",
                "ensemble=0.0",
                "--engine-max-unknown-ratio",
                "ensemble=0.0",
                "--engine-min-precision",
                "ensemble=1.0",
                "--engine-min-recall",
                "ensemble=1.0",
                "--engine-max-fp",
                "ensemble=0",
                "--engine-max-fn",
                "ensemble=0",
                "--engine-max-p95-ms",
                "ensemble=6000",
            ],
        )

    print("\nPASS operational gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
