#!/usr/bin/env python3
"""Run fast local self-checks for model quality guardrails."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAST_CHECKS = [
    ("URL split audit selfcheck", ["dev/test_url_split_audit.py"]),
    ("engine training inputs audit selfcheck", ["dev/test_engine_training_inputs_audit.py"]),
    ("prepare engine training inputs selfcheck", ["dev/test_prepare_engine_training_inputs_selfcheck.py"]),
    ("expanded training builder selfcheck", ["dev/test_build_expanded_training_selfcheck.py"]),
    ("url rule selfcheck", ["dev/test_url_rule_selfcheck.py"]),
    ("phishing blocklist selfcheck", ["dev/test_blocklist_selfcheck.py"]),
    ("URL rule adjustment selfcheck", ["dev/test_url_rule_adjustment_selfcheck.py"]),
    ("URLML endpoint rule contract selfcheck", ["dev/test_url_ml_endpoint_rule_contract.py"]),
    ("URLML batch parity selfcheck", ["dev/test_urlml_batch_parity.py"]),
    ("hard cases selfcheck", ["dev/test_hard_cases_selfcheck.py"]),
    ("eval host rule dependency selfcheck", ["dev/test_eval_host_rule_dependency.py"]),
    ("batch dedupe selfcheck", ["dev/test_batch_dedupe_selfcheck.py"]),
    ("KoBERT text dataset selfcheck", ["dev/test_kobert_text_dataset_selfcheck.py"]),
    ("full pipeline helper selfcheck", ["dev/test_full_pipeline_selfcheck.py"]),
    ("model quality stamp selfcheck", ["dev/test_model_quality_stamp_selfcheck.py"]),
    ("XGBoost artifact provenance selfcheck", ["dev/test_xgboost_artifact_provenance.py"]),
    ("current model quality sidecars", ["dev/test_current_model_quality_sidecars.py"]),
    ("KoBERT quality metadata sync selfcheck", ["dev/test_sync_kobert_quality_metadata.py"]),
    ("model quality audit selfcheck", ["dev/test_model_quality_audit.py"]),
    ("evidence gate selfcheck", ["dev/test_evidence_gate_selfcheck.py"]),
    ("live probe gate selfcheck", ["dev/test_live_probe_gate_selfcheck.py"]),
    ("live probe quality gate", ["dev/run_live_probe_gates.py"]),
    ("live probe quality report selfcheck", ["dev/test_live_probe_quality_report_selfcheck.py"]),
    ("live residual API gate", ["dev/run_live_residual_api_gate.py"]),
    ("live residual API report selfcheck", ["dev/test_live_residual_api_report_selfcheck.py"]),
    ("live API quality sample selfcheck", ["dev/test_live_api_quality_sample_selfcheck.py"]),
    ("live API batch sample selfcheck", ["dev/test_live_api_batch_sample_selfcheck.py"]),
    ("live API batch gate", ["dev/run_live_api_batch_gate.py"]),
    ("live API batch report selfcheck", ["dev/test_live_api_batch_report_selfcheck.py"]),
    ("live API quality gate", ["dev/run_live_api_quality_gate.py"]),
    ("live API quality report selfcheck", ["dev/test_live_api_quality_report_selfcheck.py"]),
    ("text URL extraction selfcheck", ["dev/test_text_url_extraction.py"]),
    ("text API contract selfcheck", ["dev/test_text_api_contract_selfcheck.py"]),
]

SLOW_CHECKS = [
    ("model metadata training artifact selfcheck", ["dev/test_model_metadata_selfcheck.py"]),
]


def run_check(name: str, script_args: list[str]) -> None:
    cmd = [sys.executable, *script_args]
    print(f"\n=== quality selfcheck: {name} ===", flush=True)
    print("+ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=BASE, check=True)
    print(f"[OK] {name} ({time.perf_counter() - t0:.1f}s)", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run local model quality selfchecks.")
    parser.add_argument("--include-slow", action="store_true", help="Also run small temporary training artifact checks.")
    parser.add_argument("--only", action="append", default=[], help="Run checks whose name contains this text.")
    args = parser.parse_args(argv)

    checks = list(FAST_CHECKS)
    if args.include_slow:
        checks.extend(SLOW_CHECKS)
    if args.only:
        lowered = [part.lower() for part in args.only]
        checks = [(name, cmd) for name, cmd in checks if any(part in name.lower() for part in lowered)]
    if not checks:
        print("FAIL no selfchecks selected")
        return 1
    for name, cmd in checks:
        run_check(name, cmd)
    print("\nPASS quality selfchecks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
