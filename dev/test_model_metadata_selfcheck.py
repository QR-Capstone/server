#!/usr/bin/env python3
"""Smoke-check that regenerated model artifacts carry validation metadata."""
from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import csv

import joblib


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(cmd: list[str]) -> None:
    completed = subprocess.run(cmd, cwd=BASE_DIR, text=True, capture_output=True)
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise AssertionError(f"command failed: {' '.join(cmd)}")


def _write_balanced_subset(src: str, dst: str, per_label: int = 300) -> None:
    kept = {"0": 0, "1": 0}
    with open(src, "r", encoding="utf-8-sig", newline="") as f_in, open(
        dst, "w", encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames or ["url", "label"])
        writer.writeheader()
        for row in reader:
            label = str(row.get("label", "")).strip()
            if label not in kept or kept[label] >= per_label:
                continue
            writer.writerow(row)
            kept[label] += 1
            if all(count >= per_label for count in kept.values()):
                break
    assert all(count >= per_label for count in kept.values()), kept


def test_xgboost_metadata(tmp_dir: str) -> None:
    train_csv = os.path.join(tmp_dir, "xgboost_subset.csv")
    _write_balanced_subset(
        os.path.join(BASE_DIR, "dev", "engine_training_inputs", "xgboost_train_10k.csv"),
        train_csv,
    )
    _run(
        [
            sys.executable,
            "xgboost/XG_train.py",
            "--input",
            train_csv,
            "--out-dir",
            tmp_dir,
            "--skip-domain",
            "--skip-dom",
            "--min-test-accuracy",
            "0.50",
        ]
    )
    artifact = joblib.load(os.path.join(tmp_dir, "url_xgb_paired_first.joblib"))
    meta = artifact["meta"]
    metrics = meta.get("internal_test_metrics") or {}
    assert "accuracy" in metrics, meta
    assert "confusion_matrix" in metrics, metrics
    assert meta.get("split_rows", {}).get("test", 0) > 0, meta


def test_gnn_metadata(tmp_dir: str) -> None:
    model_out = os.path.join(tmp_dir, "gnn_model.pkl")
    features_out = os.path.join(tmp_dir, "gnn_model_features.pkl")
    train_csv = os.path.join(tmp_dir, "gnn_subset.csv")
    _write_balanced_subset(
        os.path.join(BASE_DIR, "gnn", "gnn_total_dataset.csv"),
        train_csv,
        per_label=150,
    )
    _run(
        [
            sys.executable,
            "gnn/regenerate_gnn_model.py",
            "--csv",
            train_csv,
            "--model-out",
            model_out,
            "--features-out",
            features_out,
            "--epochs",
            "3",
            "--min-holdout-accuracy",
            "0.50",
        ]
    )
    with open(model_out, "rb") as f:
        artifact = pickle.load(f)
    metadata = artifact["metadata"]
    metrics = metadata.get("holdout_metrics") or {}
    assert "accuracy" in metrics, metadata
    assert metrics.get("test_rows", 0) > 0, metrics
    assert "confusion_matrix" in metrics, metrics


def main() -> int:
    tmp_dir = tempfile.mkdtemp(prefix="model_metadata_selfcheck_")
    try:
        test_xgboost_metadata(tmp_dir)
        test_gnn_metadata(tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("model metadata selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
