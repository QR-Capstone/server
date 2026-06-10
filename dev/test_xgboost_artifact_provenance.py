#!/usr/bin/env python3
"""Self-check XGBoost artifact provenance audit behavior."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import joblib
import audit_xgboost_artifact_provenance as audit
import stamp_model_quality_metadata as quality


def _write_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label"])
        writer.writeheader()
        writer.writerow({"url": "https://example.com", "label": "0"})
        writer.writerow({"url": "https://bad.test", "label": "1"})


def _write_artifact(path: Path, rows: int, labels: dict[str, int]) -> None:
    joblib.dump({"model": None, "feature_names": [], "meta": {"training_rows": rows, "label_counts": labels}}, path)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="xgb_provenance_") as tmp_raw:
        tmp = Path(tmp_raw)
        train_csv = tmp / "train.csv"
        good = tmp / "good.joblib"
        bad = tmp / "bad.joblib"
        waiver = tmp / "waiver.json"
        _write_csv(train_csv)
        _write_artifact(good, 2, {"0": 1, "1": 1})
        _write_artifact(bad, 2, {"0": 2, "1": 0})

        assert audit.main(["--artifact", str(good), "--training-input", str(train_csv)]) == 0
        assert audit.main(["--artifact", str(bad), "--training-input", str(train_csv)]) == 1
        waiver.write_text(
            json.dumps(
                {
                    "waivers": {
                        str(bad): {
                            "artifact_sha256": quality.sha256_file(bad),
                            "reason": "unit-test retained legacy artifact",
                            "required_action": "replace the artifact with one trained from a retained input",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        assert audit.main(["--artifact", str(bad), "--training-input", str(train_csv), "--waiver-file", str(waiver)]) == 0

    print("XGBoost artifact provenance selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
