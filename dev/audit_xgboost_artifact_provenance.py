#!/usr/bin/env python3
"""Audit XGBoost artifact metadata against retained training input CSVs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import stamp_model_quality_metadata as quality


BASE = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = [
    BASE / "xgboost" / "url_xgb_paired_first.joblib",
    BASE / "xgboost" / "url_xgb_domain_age.joblib",
    BASE / "xgboost" / "url_xgb_dom.joblib",
]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _input_signatures(paths: list[Path]) -> dict[str, dict[str, Any]]:
    signatures: dict[str, dict[str, Any]] = {}
    for path in paths:
        rel = _display_path(path)
        signatures[rel] = {
            "rows": quality.csv_row_count(path),
            "label_counts": quality.csv_label_counts(path),
        }
    return signatures


def _artifact_meta(path: Path) -> dict[str, Any]:
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected joblib dict")
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: meta missing")
    return meta


def audit(args: argparse.Namespace) -> list[str]:
    input_paths = [Path(path) for path in args.training_input]
    signatures = _input_signatures(input_paths)
    waivers: dict[str, dict[str, Any]] = {}
    if args.waiver_file:
        with Path(args.waiver_file).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for path, waiver in (payload.get("waivers") or {}).items():
            if isinstance(waiver, dict):
                waivers[str(path).replace("\\", "/")] = waiver
    failures: list[str] = []
    used_waivers: set[str] = set()
    for artifact in [Path(path) for path in args.artifact]:
        rel = _display_path(artifact)
        meta = _artifact_meta(artifact)
        rows = int(meta.get("training_rows", -1))
        labels = {str(k): int(v) for k, v in (meta.get("label_counts") or {}).items()}
        matched = [
            name
            for name, signature in signatures.items()
            if rows == int(signature["rows"]) and labels == signature["label_counts"]
        ]
        print(f"{rel}: training_rows={rows} label_counts={labels} matched_inputs={matched}")
        if not matched and rel not in args.allow_unmatched:
            waiver = waivers.get(rel)
            if waiver is not None:
                reason = str(waiver.get("reason") or "")
                action = str(waiver.get("required_action") or "")
                expected_sha = str(waiver.get("artifact_sha256") or "")
                actual_sha = quality.sha256_file(artifact)
                used_waivers.add(rel)
                if not reason:
                    failures.append(f"{rel}: waiver reason missing")
                    continue
                if not action:
                    failures.append(f"{rel}: waiver required_action missing")
                    continue
                if expected_sha != actual_sha:
                    failures.append(f"{rel}: waiver artifact_sha256 mismatch")
                    continue
                print(f"WAIVED {rel}: {reason}")
                continue
            failures.append(f"{rel}: no retained training input matches artifact metadata")
    for rel in sorted(set(waivers) - used_waivers):
        failures.append(f"{rel}: unused waiver")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit XGBoost artifact training provenance.")
    parser.add_argument("--artifact", action="append", default=None)
    parser.add_argument(
        "--training-input",
        action="append",
        default=None,
    )
    parser.add_argument("--allow-unmatched", action="append", default=[])
    parser.add_argument("--waiver-file", default="")
    args = parser.parse_args(argv)
    if args.artifact is None:
        args.artifact = [str(path) for path in DEFAULT_ARTIFACTS]
    if args.training_input is None:
        args.training_input = [str(path) for path in quality.ENGINE_TRAINING_INPUTS["xgboost"]]

    failures = audit(args)
    if failures:
        for failure in failures:
            print("FAIL " + failure)
        return 1
    print("PASS XGBoost artifact provenance audit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
