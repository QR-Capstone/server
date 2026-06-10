#!/usr/bin/env python3
"""Audit deployed model quality evidence for all four phishing engines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import stamp_model_quality_metadata as stamp
import sync_kobert_quality_metadata as kobert_sync


BASE = Path(__file__).resolve().parents[1]
QUALITY_DIR = BASE / "dev" / "model_quality"
KOBERT_META = BASE / "KoBERT" / "kobert_phishing_model_weights.meta.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _metric_failures(engine: str, metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    checks = [
        ("rows", int(args.min_rows), "below"),
        ("accuracy", float(args.min_accuracy), "below"),
        ("accuracy_lower_95", float(args.min_accuracy_lower), "below"),
        ("precision", float(args.min_precision), "below"),
        ("recall", float(args.min_recall), "below"),
        ("recall_lower_95", float(args.min_recall_lower), "below"),
        ("specificity", float(args.min_specificity), "below"),
        ("specificity_lower_95", float(args.min_specificity_lower), "below"),
        ("false_positive", int(args.max_false_positive), "above"),
        ("false_negative", int(args.max_false_negative), "above"),
        ("unknown", int(args.max_unknown), "above"),
        ("unknown_ratio", float(args.max_unknown_ratio), "above"),
        ("p95_latency_ms", float(args.max_p95_ms), "above"),
    ]
    failures: list[str] = []
    for key, threshold, direction in checks:
        value = metrics.get(key)
        if value is None:
            failures.append(f"{key}=missing")
            continue
        value_f = float(value)
        threshold_f = float(threshold)
        if direction == "below" and value_f < threshold_f:
            failures.append(f"{key}={value_f:.6f} < {threshold_f:.6f}")
        elif direction == "above" and value_f > threshold_f:
            failures.append(f"{key}={value_f:.6f} > {threshold_f:.6f}")
    return failures


def _artifact_failures(payload: dict[str, Any], artifact_paths: list[Path]) -> list[str]:
    failures: list[str] = []
    by_path = {item.get("path"): item for item in payload.get("artifacts") or []}
    for artifact_path in artifact_paths:
        rel = str(artifact_path.relative_to(BASE)).replace("\\", "/")
        record = by_path.get(rel)
        if not artifact_path.is_file():
            failures.append(f"{rel}=missing_file")
            continue
        if not record:
            failures.append(f"{rel}=missing_sidecar_record")
            continue
        actual_sha = stamp.sha256_file(artifact_path)
        if record.get("sha256") != actual_sha:
            failures.append(f"{rel}=sha256_mismatch")
        if int(record.get("size_bytes", -1)) != int(artifact_path.stat().st_size):
            failures.append(f"{rel}=size_mismatch")
    return failures


def _training_input_failures(payload: dict[str, Any], input_paths: list[Path]) -> list[str]:
    failures: list[str] = []
    by_path = {item.get("path"): item for item in payload.get("training_inputs") or []}
    for input_path in input_paths:
        rel = str(input_path.relative_to(BASE)).replace("\\", "/")
        record = by_path.get(rel)
        if not input_path.is_file():
            failures.append(f"{rel}=missing_file")
            continue
        if not record:
            failures.append(f"{rel}=missing_sidecar_record")
            continue
        actual_sha = stamp.sha256_file(input_path)
        if record.get("sha256") != actual_sha:
            failures.append(f"{rel}=sha256_mismatch")
        if int(record.get("size_bytes", -1)) != int(input_path.stat().st_size):
            failures.append(f"{rel}=size_mismatch")
        if int(record.get("rows", -1)) != stamp.csv_row_count(input_path):
            failures.append(f"{rel}=row_count_mismatch")
        if record.get("label_counts") != stamp.csv_label_counts(input_path):
            failures.append(f"{rel}=label_counts_mismatch")
    return failures


def _runtime_rule_failures(payload: dict[str, Any], rule_paths: list[Path]) -> list[str]:
    failures: list[str] = []
    by_path = {item.get("path"): item for item in payload.get("runtime_rule_files") or []}
    for rule_path in rule_paths:
        rel = stamp.display_path(rule_path)
        record = by_path.get(rel)
        if not rule_path.is_file():
            failures.append(f"{rel}=missing_file")
            continue
        if not record:
            failures.append(f"{rel}=missing_sidecar_record")
            continue
        actual_sha = stamp.sha256_file(rule_path)
        if record.get("sha256") != actual_sha:
            failures.append(f"{rel}=sha256_mismatch")
        if int(record.get("size_bytes", -1)) != int(rule_path.stat().st_size):
            failures.append(f"{rel}=size_mismatch")
    return failures


def _kobert_meta_failures(kobert_payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    meta = _load_json(KOBERT_META)
    op = meta.get("operational_api_holdout")
    if not isinstance(op, dict):
        return ["operational_api_holdout=missing"]
    sidecar_artifact = kobert_sync._kobert_artifact_record(kobert_payload)
    if op.get("artifact_sha256") != sidecar_artifact.get("sha256"):
        failures.append("operational_api_holdout.artifact_sha256=sidecar_mismatch")
    sidecar_metrics = kobert_payload.get("api_holdout_metrics") or {}
    for key in ("rows", "accuracy", "accuracy_lower_95", "recall_lower_95", "specificity_lower_95", "false_positive", "false_negative", "unknown"):
        if op.get(key) != sidecar_metrics.get(key):
            failures.append(f"operational_api_holdout.{key}=sidecar_mismatch")
    if "best_val_accuracy" not in meta:
        failures.append("best_val_accuracy=missing")
    return failures


def audit(args: argparse.Namespace) -> list[str]:
    engines = args.engine or sorted(stamp.ENGINE_ARTIFACTS)
    failures: list[str] = []
    for engine in engines:
        sidecar_path = QUALITY_DIR / f"{engine}_quality.json"
        if not sidecar_path.is_file():
            failures.append(f"{engine}: sidecar_missing:{sidecar_path}")
            continue
        payload = _load_json(sidecar_path)
        if payload.get("engine") != engine:
            failures.append(f"{engine}: sidecar_engine={payload.get('engine')!r}")
            continue
        metrics = payload.get("api_holdout_metrics")
        if not isinstance(metrics, dict):
            failures.append(f"{engine}: api_holdout_metrics=missing")
            continue
        for failure in _metric_failures(engine, metrics, args):
            failures.append(f"{engine}: {failure}")
        for failure in _artifact_failures(payload, stamp.ENGINE_ARTIFACTS[engine]):
            failures.append(f"{engine}: {failure}")
        for failure in _training_input_failures(payload, stamp.ENGINE_TRAINING_INPUTS.get(engine, [])):
            failures.append(f"{engine}: training_input:{failure}")
        for failure in _runtime_rule_failures(payload, stamp.RUNTIME_RULE_FILES):
            failures.append(f"{engine}: runtime_rule:{failure}")
        if engine == "kobert":
            for failure in _kobert_meta_failures(payload):
                failures.append(f"{engine}: {failure}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit current model quality sidecars and artifact hashes.")
    parser.add_argument("--engine", action="append", choices=sorted(stamp.ENGINE_ARTIFACTS), default=[])
    parser.add_argument("--min-rows", type=int, default=900)
    parser.add_argument("--min-accuracy", type=float, default=0.99)
    parser.add_argument("--min-accuracy-lower", type=float, default=0.99)
    parser.add_argument("--min-precision", type=float, default=0.99)
    parser.add_argument("--min-recall", type=float, default=0.99)
    parser.add_argument("--min-recall-lower", type=float, default=0.99)
    parser.add_argument("--min-specificity", type=float, default=0.99)
    parser.add_argument("--min-specificity-lower", type=float, default=0.99)
    parser.add_argument("--max-false-positive", type=int, default=0)
    parser.add_argument("--max-false-negative", type=int, default=0)
    parser.add_argument("--max-unknown", type=int, default=0)
    parser.add_argument("--max-unknown-ratio", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float, default=1000.0)
    args = parser.parse_args(argv)

    failures = audit(args)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    engines = args.engine or sorted(stamp.ENGINE_ARTIFACTS)
    print("PASS model quality audit: " + ", ".join(engines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
