#!/usr/bin/env python3
"""Self-check KoBERT quality metadata sync."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import sync_kobert_quality_metadata as sync


def main() -> int:
    meta = sync.sync(sync.DEFAULT_META, sync.DEFAULT_QUALITY)
    op = meta.get("operational_api_holdout") or {}
    assert op["accuracy"] >= 0.99, op
    assert op["accuracy_lower_95"] >= 0.99, op
    assert op["recall_lower_95"] >= 0.99, op
    assert op["specificity_lower_95"] >= 0.99, op
    assert op["false_positive"] == 0, op
    assert op["false_negative"] == 0, op
    assert op["unknown"] == 0, op
    assert op["artifact_sha256"], op
    assert "best_val_accuracy" in meta, meta

    with tempfile.TemporaryDirectory(prefix="kobert_quality_meta_") as tmp_raw:
        tmp = Path(tmp_raw)
        meta_path = tmp / "meta.json"
        quality_path = tmp / "quality.json"
        meta_path.write_text(json.dumps({"best_val_accuracy": 0.1}), encoding="utf-8")
        quality_payload = json.loads(sync.DEFAULT_QUALITY.read_text(encoding="utf-8"))
        quality_payload["artifacts"][0]["sha256"] = "0" * 64
        quality_path.write_text(json.dumps(quality_payload), encoding="utf-8")
        try:
            sync.sync(meta_path, quality_path)
        except ValueError as exc:
            assert "sha256" in str(exc)
        else:
            raise AssertionError("expected sha mismatch to fail")

    print("sync KoBERT quality metadata selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
