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
DEFAULT_HARD_CASES = os.path.join("dev", "model_quality", "hard_cases.csv")
DEFAULT_XGBOOST_PROVENANCE_WAIVERS = os.path.join("dev", "model_quality", "xgboost_provenance_waivers.json")


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
    parser.add_argument("--hard-cases", default=DEFAULT_HARD_CASES)
    parser.add_argument("--xgboost-provenance-waivers", default=DEFAULT_XGBOOST_PROVENANCE_WAIVERS)
    parser.add_argument("--skip-urlml", action="store_true")
    parser.add_argument("--skip-api-performance", action="store_true")
    parser.add_argument("--skip-api-holdout", action="store_true")
    parser.add_argument("--skip-hard-cases", action="store_true")
    parser.add_argument("--skip-kobert-holdout", action="store_true")
    parser.add_argument("--kobert-limit", type=int, default=1000)
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
    run(
        "dataset split integrity",
        [
            sys.executable,
            os.path.join("dev", "audit_url_splits.py"),
            "--train",
            args.train,
            "--test",
            args.holdout,
            "--test",
            balanced_holdout,
            "--test",
            operational_holdout,
            "--min-train-rows",
            "8000",
            "--min-test-rows",
            "500",
            "--balanced-test",
            balanced_holdout,
            "--balanced-max-delta",
            "0",
        ],
    )
    run(
        "engine training input integrity",
        [
            sys.executable,
            os.path.join("dev", "audit_engine_training_inputs.py"),
            "--target",
            "10000",
        ],
    )
    run(
        "XGBoost artifact provenance",
        [
            sys.executable,
            os.path.join("dev", "audit_xgboost_artifact_provenance.py"),
            "--waiver-file",
            args.xgboost_provenance_waivers,
        ],
    )

    if not args.skip_api_holdout:
        sample_args = []
        if args.api_holdout_per_label_limit:
            sample_args = ["--per-label-limit", str(args.api_holdout_per_label_limit)]
        if not args.skip_hard_cases:
            run(
                "api curated hard cases",
                [
                    sys.executable,
                    os.path.join("dev", "evaluate_api_holdout.py"),
                    "--spawn-server",
                    "--holdout",
                    args.hard_cases,
                    "--engine",
                    "urlml",
                    "--engine",
                    "xgboost",
                    "--engine",
                    "gnn",
                    "--engine",
                    "kobert",
                    "--workers",
                    "4",
                    "--timeout",
                    "90",
                    "--require-ready",
                    "--show-misses",
                    "20",
                    "--min-accuracy",
                    "1.0",
                    "--max-fp",
                    "0",
                    "--max-fn",
                    "0",
                    "--max-unknown-ratio",
                    "0.0",
                    "--max-p95-ms",
                    "1000",
                ],
            )
        quality_dir = os.path.join("dev", "model_quality")
        os.makedirs(os.path.join(BASE, quality_dir), exist_ok=True)
        core_quality_json = os.path.join(quality_dir, "api_holdout_core.json")
        kobert_quality_json = os.path.join(quality_dir, "api_holdout_kobert.json")
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
                "xgboost=0.99",
                "--engine-max-fp-rate",
                "xgboost=0.01",
                "--engine-max-fn-rate",
                "xgboost=0.01",
                "--engine-max-unknown-ratio",
                "gnn=0.0",
                "--engine-min-accuracy",
                "gnn=0.99",
                "--engine-max-fp-rate",
                "gnn=0.01",
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
                "xgboost=0.99",
                "--engine-min-recall",
                "xgboost=0.99",
                "--engine-min-precision",
                "gnn=0.99",
                "--engine-min-recall",
                "gnn=0.99",
                "--engine-min-precision",
                "ensemble=1.0",
                "--engine-min-recall",
                "ensemble=1.0",
                "--engine-min-accuracy-lower-bound",
                "urlml=0.99",
                "--engine-min-accuracy-lower-bound",
                "xgboost=0.99",
                "--engine-min-accuracy-lower-bound",
                "gnn=0.99",
                "--engine-min-accuracy-lower-bound",
                "ensemble=0.99",
                "--engine-min-recall-lower-bound",
                "urlml=0.99",
                "--engine-min-recall-lower-bound",
                "xgboost=0.99",
                "--engine-min-recall-lower-bound",
                "gnn=0.99",
                "--engine-min-recall-lower-bound",
                "ensemble=0.99",
                "--engine-min-specificity-lower-bound",
                "urlml=0.99",
                "--engine-min-specificity-lower-bound",
                "xgboost=0.99",
                "--engine-min-specificity-lower-bound",
                "gnn=0.99",
                "--engine-min-specificity-lower-bound",
                "ensemble=0.99",
                "--engine-max-fp",
                "ensemble=0",
                "--engine-max-fn",
                "ensemble=0",
                "--engine-max-p95-ms",
                "ensemble=6000",
                "--json-out",
                core_quality_json,
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
                    "--engine-min-accuracy",
                    "kobert=0.99",
                    "--engine-min-accuracy-lower-bound",
                    "kobert=0.99",
                    "--engine-min-recall-lower-bound",
                    "kobert=0.99",
                    "--engine-min-specificity-lower-bound",
                    "kobert=0.99",
                    "--engine-max-unknown-ratio",
                    "kobert=0.0",
                    "--engine-max-p95-ms",
                    "kobert=1000",
                    "--json-out",
                    kobert_quality_json,
                ],
            )
        stamp_cmd = [
            sys.executable,
            os.path.join("dev", "stamp_model_quality_metadata.py"),
            "--api-json",
            core_quality_json,
            "--engine",
            "urlml",
            "--engine",
            "xgboost",
            "--engine",
            "gnn",
            "--min-accuracy",
            "0.99",
            "--min-accuracy-lower",
            "0.99",
            "--min-recall",
            "0.99",
            "--min-specificity",
            "0.99",
            "--max-unknown-ratio",
            "0.0",
            "--max-fp-rate",
            "0.01",
            "--max-fn-rate",
            "0.01",
        ]
        if not args.skip_kobert_holdout:
            stamp_cmd.extend(["--api-json", kobert_quality_json, "--engine", "kobert"])
        run("stamp current model quality metadata", stamp_cmd)
        if not args.skip_kobert_holdout:
            run(
                "sync KoBERT operational quality metadata",
                [
                    sys.executable,
                    os.path.join("dev", "sync_kobert_quality_metadata.py"),
                ],
            )
        run(
            "audit current model quality evidence",
            [
                sys.executable,
                os.path.join("dev", "audit_model_quality.py"),
                "--min-rows",
                "900",
                "--min-accuracy",
                "0.99",
                "--min-accuracy-lower",
                "0.99",
                "--min-recall-lower",
                "0.99",
                "--min-specificity-lower",
                "0.99",
                "--max-false-positive",
                "0",
                "--max-false-negative",
                "0",
                "--max-unknown",
                "0",
                "--max-p95-ms",
                "1000",
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
                "xgboost=0.99",
                "--engine-max-fp-rate",
                "xgboost=0.01",
                "--engine-max-fn-rate",
                "xgboost=0.01",
                "--engine-max-unknown-ratio",
                "gnn=0.0",
                "--engine-min-accuracy",
                "gnn=0.99",
                "--engine-max-fp-rate",
                "gnn=0.01",
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
        run(
            "api batch consistency",
            [
                sys.executable,
                os.path.join("dev", "evaluate_api_batch_consistency.py"),
                "--spawn-server",
                "--holdout",
                balanced_holdout,
                "--per-label-limit",
                "100",
                "--engine",
                "urlml",
                "--engine",
                "ensemble",
                "--batch-size",
                "16",
                "--include-duplicate-contract",
                "--timeout",
                "90",
                "--require-ready",
            ],
        )
        run(
            "api crossfeed batch consistency",
            [
                sys.executable,
                os.path.join("dev", "evaluate_api_batch_consistency.py"),
                "--spawn-server",
                "--port-start",
                "8070",
                "--holdout",
                os.path.join("dev", "nonoverlap_crossfeed_eval_20260524.csv"),
                "--per-label-limit",
                "150",
                "--engine",
                "urlml",
                "--engine",
                "ensemble",
                "--batch-size",
                "32",
                "--include-duplicate-contract",
                "--timeout",
                "90",
                "--require-ready",
            ],
        )

    print("\nPASS operational gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
