#!/usr/bin/env python3
"""Evaluate live probe samples through production API batch endpoints."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEV = BASE / "dev"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from run_live_api_batch_gate import build_source_balanced_sample, canonical_url_key, discover  # noqa: E402


def build_quality_sample(probes: list[Path], per_source_label: int) -> list[dict[str, str]]:
    rows = build_source_balanced_sample(probes, per_source_label)
    selected_keys = {row["canonical_key"] for row in rows}
    malicious_count = sum(1 for row in rows if row["label"] == "1")
    benign_count = sum(1 for row in rows if row["label"] == "0")
    if benign_count >= malicious_count:
        return rows

    needed = malicious_count - benign_count
    extra_benign: list[dict[str, str]] = []
    for path in probes:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                source = (row.get("source") or path.stem).strip() or path.stem
                key = canonical_url_key(url)
                if not url or label != "0" or not key or key in selected_keys:
                    continue
                selected_keys.add(key)
                extra_benign.append({"url": url, "label": label, "source": source, "canonical_key": key})
                if len(extra_benign) >= needed:
                    return sorted([*rows, *extra_benign], key=lambda item: (item["label"], item["source"], item["canonical_key"]))
    return sorted([*rows, *extra_benign], key=lambda item: (item["label"], item["source"], item["canonical_key"]))


def write_sample(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "canonical_key"])
        writer.writeheader()
        writer.writerows(rows)


def sample_balance_failures(rows: list[dict[str, str]]) -> list[str]:
    malicious = sum(1 for row in rows if row["label"] == "1")
    benign = sum(1 for row in rows if row["label"] == "0")
    failures: list[str] = []
    if not rows:
        failures.append("no live probe rows sampled")
    if malicious == 0:
        failures.append("no malicious rows sampled")
    if benign == 0:
        failures.append("no benign rows sampled")
    if malicious != benign:
        failures.append(f"sample labels are imbalanced: malicious={malicious} benign={benign}")
    if len({row["canonical_key"] for row in rows}) != len(rows):
        failures.append("duplicate canonical keys in sample")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--probe-glob", action="append", default=["dev/latest_*probe_20260607_*.csv"])
    parser.add_argument("--per-source-label", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--port-start", type=int, default=8110)
    parser.add_argument("--max-fp", type=int, default=2)
    parser.add_argument("--max-unknown-ratio", type=float, default=0.01)
    parser.add_argument("--json-out", default="dev/model_quality/live_api_quality.json")
    args = parser.parse_args(argv)

    probes = [BASE / path for path in args.probe] if args.probe else discover(args.probe_glob)
    rows = build_quality_sample(probes, args.per_source_label)
    failures = sample_balance_failures(rows)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(
        "live_api_quality_sample "
        f"rows={len(rows)} malicious={sum(1 for row in rows if row['label'] == '1')} "
        f"benign={sum(1 for row in rows if row['label'] == '0')} "
        f"sources={sorted({row['source'] for row in rows})}"
    )

    with tempfile.TemporaryDirectory(prefix="live_api_quality_gate_") as tmp:
        sample_path = Path(tmp) / "live_sample.csv"
        report_path = BASE / args.json_out
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_sample(rows, sample_path)
        cmd = [
            sys.executable,
            "dev/evaluate_api_holdout.py",
            "--spawn-server",
            "--port-start",
            str(args.port_start),
            "--holdout",
            str(sample_path),
            "--engine",
            "urlml",
            "--engine",
            "ensemble",
            "--batch-size",
            str(args.batch_size),
            "--workers",
            "4",
            "--timeout",
            str(args.timeout),
            "--require-ready",
            "--show-misses",
            "20",
            "--max-unknown-ratio",
            str(args.max_unknown_ratio),
            "--min-accuracy",
            "0.99",
            "--max-fp-rate",
            "0.01",
            "--max-fn-rate",
            "0.0",
            "--engine-max-fn",
            "urlml=0",
            "--engine-max-fn",
            "ensemble=0",
            "--engine-max-fp",
            f"urlml={args.max_fp}",
            "--engine-max-fp",
            f"ensemble={args.max_fp}",
            "--engine-max-unknown-ratio",
            f"urlml={args.max_unknown_ratio}",
            "--engine-max-unknown-ratio",
            f"ensemble={args.max_unknown_ratio}",
            "--engine-min-recall",
            "urlml=1.0",
            "--engine-min-recall",
            "ensemble=1.0",
            "--json-out",
            str(report_path),
        ]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=BASE, check=True)
    print("PASS live API quality gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
