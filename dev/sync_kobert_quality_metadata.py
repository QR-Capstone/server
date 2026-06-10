#!/usr/bin/env python3
"""Copy verified KoBERT API-quality evidence into the KoBERT training meta file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import stamp_model_quality_metadata as quality


BASE = Path(__file__).resolve().parents[1]
DEFAULT_META = BASE / "KoBERT" / "kobert_phishing_model_weights.meta.json"
DEFAULT_QUALITY = BASE / "dev" / "model_quality" / "kobert_quality.json"
WEIGHTS = BASE / "KoBERT" / "kobert_phishing_model_weights.pt"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _kobert_artifact_record(sidecar: dict[str, Any]) -> dict[str, Any]:
    for record in sidecar.get("artifacts") or []:
        if record.get("path") == "KoBERT/kobert_phishing_model_weights.pt":
            return record
    raise ValueError("kobert weights artifact record missing from quality sidecar")


def sync(meta_path: Path, quality_path: Path) -> dict[str, Any]:
    meta = _load_json(meta_path)
    sidecar = _load_json(quality_path)
    if sidecar.get("engine") != "kobert":
        raise ValueError(f"quality sidecar engine must be kobert, got {sidecar.get('engine')!r}")

    artifact = _kobert_artifact_record(sidecar)
    actual_sha = quality.sha256_file(WEIGHTS)
    if artifact.get("sha256") != actual_sha:
        raise ValueError("quality sidecar does not match current KoBERT weights sha256")

    metrics = sidecar.get("api_holdout_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("quality sidecar missing api_holdout_metrics")

    meta["operational_api_holdout"] = {
        "scope": "KoBERT API engine, including trusted-domain and URL-rule preflight adjustments",
        "artifact_sha256": actual_sha,
        "stamped_at_utc": sidecar.get("stamped_at_utc"),
        "rows": int(metrics.get("rows", 0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "accuracy_lower_95": float(metrics.get("accuracy_lower_95", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "recall_lower_95": float(metrics.get("recall_lower_95", 0.0)),
        "specificity": float(metrics.get("specificity", 0.0)),
        "specificity_lower_95": float(metrics.get("specificity_lower_95", 0.0)),
        "false_positive": int(metrics.get("false_positive", 0)),
        "false_negative": int(metrics.get("false_negative", 0)),
        "unknown": int(metrics.get("unknown", 0)),
        "p95_latency_ms": float(metrics.get("p95_latency_ms", 0.0)),
        "source_stats": metrics.get("source_stats", {}),
    }
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sync KoBERT training metadata with verified API holdout quality.")
    parser.add_argument("--meta", default=str(DEFAULT_META))
    parser.add_argument("--quality", default=str(DEFAULT_QUALITY))
    args = parser.parse_args(argv)

    meta_path = Path(args.meta)
    quality_path = Path(args.quality)
    meta = sync(meta_path, quality_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"synced KoBERT operational quality: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
