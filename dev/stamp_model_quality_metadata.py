#!/usr/bin/env python3
"""Attach API holdout quality evidence to the currently deployed model artifacts.

The evaluator writes engine-level metrics. This script binds those metrics to
the exact artifact files by recording each file's SHA-256 and size in sidecar
JSON files. It refuses to stamp metrics that do not meet the configured floor.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]

ENGINE_ARTIFACTS: dict[str, list[Path]] = {
    "urlml": [BASE / "url_ml" / "url_ml_model.joblib"],
    "xgboost": [
        BASE / "xgboost" / "url_xgb_paired_first.joblib",
        BASE / "xgboost" / "url_xgb_domain_age.joblib",
        BASE / "xgboost" / "url_xgb_dom.joblib",
    ],
    "gnn": [
        BASE / "gnn" / "gnn_model.pkl",
        BASE / "gnn" / "gnn_model_features.pkl",
    ],
    "kobert": [BASE / "KoBERT" / "kobert_phishing_model_weights.pt"],
}

ENGINE_TRAINING_INPUTS: dict[str, list[Path]] = {
    "urlml": [BASE / "dev" / "engine_training_inputs" / "expanded_url_train_all.csv"],
    "xgboost": [
        BASE / "dev" / "engine_training_inputs" / "expanded_url_train_all.csv",
        BASE / "dev" / "engine_training_inputs" / "xgboost_train_10k.csv",
    ],
    "gnn": [BASE / "gnn" / "gnn_total_dataset.csv"],
    "kobert": [BASE / "dev" / "engine_training_inputs" / "kobert_text_train_10k.csv"],
}

RUNTIME_RULE_FILES: list[Path] = [
    BASE / "trusted_domains.py",
    BASE / "main.py",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE)).replace("\\", "/")
    except ValueError:
        return str(path)


def artifact_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
    }


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        if not f.readline():
            return 0
        return sum(1 for _ in f)


def csv_label_counts(path: Path) -> dict[str, int]:
    counts = {"0": 0, "1": 0}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            return counts
        for row in reader:
            label = str(row.get("label") or "").strip()
            if label in counts:
                counts[label] += 1
    return counts


def training_input_record(path: Path) -> dict[str, Any]:
    record = artifact_record(path)
    record["rows"] = csv_row_count(path)
    record["label_counts"] = csv_label_counts(path)
    return record


def runtime_rule_record(path: Path) -> dict[str, Any]:
    return artifact_record(path)


def _result_by_engine(api_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    results = api_report.get("results")
    if not isinstance(results, list):
        raise ValueError("api report must contain a results list")
    out: dict[str, dict[str, Any]] = {}
    for result in results:
        if isinstance(result, dict) and result.get("engine"):
            out[str(result["engine"])] = result
    return out


def _failures(result: dict[str, Any], args: argparse.Namespace) -> list[str]:
    checks = [
        ("accuracy", args.min_accuracy, "below"),
        ("accuracy_lower_95", args.min_accuracy_lower, "below"),
        ("recall", args.min_recall, "below"),
        ("specificity", args.min_specificity, "below"),
        ("unknown_ratio", args.max_unknown_ratio, "above"),
        ("false_positive_rate", args.max_fp_rate, "above"),
        ("false_negative_rate", args.max_fn_rate, "above"),
    ]
    failures: list[str] = []
    for key, threshold, direction in checks:
        value = float(result.get(key, 0.0))
        if direction == "below" and value < threshold:
            failures.append(f"{key}={value:.6f} < {threshold:.6f}")
        elif direction == "above" and value > threshold:
            failures.append(f"{key}={value:.6f} > {threshold:.6f}")
    return failures


def write_quality_sidecar(engine: str, result: dict[str, Any], out_dir: Path) -> Path:
    artifacts = ENGINE_ARTIFACTS[engine]
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{engine} artifact missing: {missing}")
    training_inputs = ENGINE_TRAINING_INPUTS.get(engine, [])
    missing_inputs = [str(path) for path in training_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"{engine} training input missing: {missing_inputs}")
    missing_rules = [str(path) for path in RUNTIME_RULE_FILES if not path.is_file()]
    if missing_rules:
        raise FileNotFoundError(f"{engine} runtime rule file missing: {missing_rules}")
    payload = {
        "engine": engine,
        "stamped_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifacts": [artifact_record(path) for path in artifacts],
        "training_inputs": [training_input_record(path) for path in training_inputs],
        "runtime_rule_files": [runtime_rule_record(path) for path in RUNTIME_RULE_FILES],
        "api_holdout_metrics": result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{engine}_quality.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stamp model artifact quality sidecars from evaluate_api_holdout JSON.")
    parser.add_argument(
        "--api-json",
        action="append",
        required=True,
        help="JSON produced by dev/evaluate_api_holdout.py --json-out. May be provided more than once.",
    )
    parser.add_argument("--out-dir", default=str(BASE / "dev" / "model_quality"))
    parser.add_argument("--engine", action="append", choices=sorted(ENGINE_ARTIFACTS), default=[])
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    parser.add_argument("--min-accuracy-lower", type=float, default=0.99)
    parser.add_argument("--min-recall", type=float, default=0.99)
    parser.add_argument("--min-specificity", type=float, default=0.99)
    parser.add_argument("--max-unknown-ratio", type=float, default=0.0)
    parser.add_argument("--max-fp-rate", type=float, default=0.01)
    parser.add_argument("--max-fn-rate", type=float, default=0.01)
    args = parser.parse_args(argv)

    results: dict[str, dict[str, Any]] = {}
    for api_json in args.api_json:
        with open(api_json, "r", encoding="utf-8") as f:
            api_report = json.load(f)
        results.update(_result_by_engine(api_report))
    engines = args.engine or sorted(ENGINE_ARTIFACTS)
    failed = False
    for engine in engines:
        result = results.get(engine)
        if not result:
            print(f"FAIL {engine}: missing from API report")
            failed = True
            continue
        failures = _failures(result, args)
        if failures:
            print(f"FAIL {engine}: {'; '.join(failures)}")
            failed = True
            continue
        out_path = write_quality_sidecar(engine, result, Path(args.out_dir))
        print(f"stamped {engine}: {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
