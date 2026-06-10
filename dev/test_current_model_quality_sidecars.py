#!/usr/bin/env python3
"""Verify quality sidecars still match the current deployed artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import stamp_model_quality_metadata as stamp


BASE = Path(__file__).resolve().parents[1]
QUALITY_DIR = BASE / "dev" / "model_quality"


def _assert_metrics(metrics: dict) -> None:
    assert float(metrics["accuracy"]) >= 0.99, metrics
    assert float(metrics["accuracy_lower_95"]) >= 0.99, metrics
    assert float(metrics["recall"]) >= 0.99, metrics
    assert float(metrics["specificity"]) >= 0.99, metrics
    assert float(metrics["unknown_ratio"]) == 0.0, metrics
    assert float(metrics["false_positive_rate"]) <= 0.01, metrics
    assert float(metrics["false_negative_rate"]) <= 0.01, metrics


def main() -> int:
    for engine, artifact_paths in stamp.ENGINE_ARTIFACTS.items():
        sidecar_path = QUALITY_DIR / f"{engine}_quality.json"
        assert sidecar_path.is_file(), sidecar_path
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        assert payload["engine"] == engine, payload
        _assert_metrics(payload["api_holdout_metrics"])

        by_path = {item["path"]: item for item in payload["artifacts"]}
        for artifact_path in artifact_paths:
            rel = str(artifact_path.relative_to(BASE)).replace("\\", "/")
            record = by_path.get(rel)
            assert record, f"missing {rel} in {sidecar_path}"
            assert record["sha256"] == stamp.sha256_file(artifact_path), rel
            assert int(record["size_bytes"]) == artifact_path.stat().st_size, rel
        input_by_path = {item["path"]: item for item in payload.get("training_inputs", [])}
        for input_path in stamp.ENGINE_TRAINING_INPUTS.get(engine, []):
            rel = str(input_path.relative_to(BASE)).replace("\\", "/")
            record = input_by_path.get(rel)
            assert record, f"missing training input {rel} in {sidecar_path}"
            assert record["sha256"] == stamp.sha256_file(input_path), rel
            assert int(record["size_bytes"]) == input_path.stat().st_size, rel
            assert int(record["rows"]) == stamp.csv_row_count(input_path), rel
            assert record["label_counts"] == stamp.csv_label_counts(input_path), rel
        rule_by_path = {item["path"]: item for item in payload.get("runtime_rule_files", [])}
        for rule_path in stamp.RUNTIME_RULE_FILES:
            rel = str(rule_path.relative_to(BASE)).replace("\\", "/")
            record = rule_by_path.get(rel)
            assert record, f"missing runtime rule {rel} in {sidecar_path}"
            assert record["sha256"] == stamp.sha256_file(rule_path), rel
            assert int(record["size_bytes"]) == rule_path.stat().st_size, rel

    print("current model quality sidecars passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
