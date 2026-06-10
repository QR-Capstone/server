#!/usr/bin/env python3
"""Self-check quality sidecar stamping without touching deployed artifacts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import stamp_model_quality_metadata as stamp


def _write_report(path: Path, accuracy_lower: float) -> None:
    result = {
        "engine": "urlml",
        "rows": 10,
        "accuracy": 1.0,
        "accuracy_lower_95": accuracy_lower,
        "precision": 1.0,
        "recall": 1.0,
        "recall_lower_95": 1.0,
        "specificity": 1.0,
        "specificity_lower_95": 1.0,
        "false_positive_rate": 0.0,
        "false_negative_rate": 0.0,
        "unknown_ratio": 0.0,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump({"ok": True, "rows": 10, "results": [result]}, f)


def main() -> int:
    original = stamp.ENGINE_ARTIFACTS
    original_inputs = stamp.ENGINE_TRAINING_INPUTS
    original_rules = stamp.RUNTIME_RULE_FILES
    with tempfile.TemporaryDirectory(prefix="quality_stamp_selfcheck_") as tmp_raw:
        tmp = Path(tmp_raw)
        artifact = tmp / "model.bin"
        artifact.write_bytes(b"model")
        training_input = tmp / "train.csv"
        training_input.write_text("url,label\nhttps://example.com,0\n", encoding="utf-8")
        runtime_rule = tmp / "rules.py"
        runtime_rule.write_text("KNOWN = {'example.com'}\n", encoding="utf-8")
        stamp.ENGINE_ARTIFACTS = {"urlml": [artifact]}
        stamp.ENGINE_TRAINING_INPUTS = {"urlml": [training_input]}
        stamp.RUNTIME_RULE_FILES = [runtime_rule]
        try:
            passing = tmp / "passing.json"
            _write_report(passing, 0.991)
            out_dir = tmp / "quality"
            assert stamp.main(["--api-json", str(passing), "--out-dir", str(out_dir), "--engine", "urlml"]) == 0
            sidecar = out_dir / "urlml_quality.json"
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            assert payload["artifacts"][0]["sha256"]
            assert payload["training_inputs"][0]["sha256"]
            assert payload["training_inputs"][0]["rows"] == 1
            assert payload["training_inputs"][0]["label_counts"] == {"0": 1, "1": 0}
            assert payload["runtime_rule_files"][0]["path"].endswith("rules.py")
            assert payload["runtime_rule_files"][0]["sha256"] == stamp.sha256_file(runtime_rule)
            assert payload["api_holdout_metrics"]["accuracy_lower_95"] == 0.991

            failing = tmp / "failing.json"
            _write_report(failing, 0.98)
            assert stamp.main(["--api-json", str(failing), "--out-dir", str(out_dir), "--engine", "urlml"]) == 1
        finally:
            stamp.ENGINE_ARTIFACTS = original
            stamp.ENGINE_TRAINING_INPUTS = original_inputs
            stamp.RUNTIME_RULE_FILES = original_rules
    print("model quality stamp selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
