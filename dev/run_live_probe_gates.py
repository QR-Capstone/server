#!/usr/bin/env python3
"""Validate current live non-overlap probe CSVs against URLML.

The live probes are intentionally kept out of training inputs. This gate turns
them into repeatable evidence: no malicious URL may be missed, while a tiny
number of suspicious Tranco-labeled benign URLs can remain non-SAFE instead of
being forced into trusted exceptions.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
URL_ML_DIR = BASE / "url_ml"
if str(URL_ML_DIR) not in sys.path:
    sys.path.insert(0, str(URL_ML_DIR))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from train_url_ml import canonical_url_key  # noqa: E402
from url_ml_engine import load_url_ml_model, predict_url_ml  # noqa: E402


DEFAULT_PROBE_GLOB = "dev/latest_*probe_20260607_*.csv"
DEFAULT_REFERENCE_FILES = [
    "dev/dataset_splits/train.csv",
    "dev/dataset_splits/validation.csv",
    "dev/dataset_splits/test_balanced.csv",
    "dev/dataset_splits/test_operational.csv",
]
DEFAULT_ALLOWED_RESIDUAL_MISSES = [
    "fp|tranco_latest_nonoverlap|https://blog44investment.shop",
    "fp|tranco_latest_nonoverlap|https://y-y-z-y-w-101.top",
    "fp|tranco_latest_nonoverlap|https://y-y-z-y-w-1.top",
    "fp|tranco_latest_nonoverlap|https://confirm-payment-id694787.com",
]


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(BASE)).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_reference_keys(paths: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        for row in _read_rows(path):
            key = canonical_url_key(row.get("url") or "")
            if key:
                keys.add(key)
    return keys


def _discover_probes(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matched = glob.glob(str(BASE / pattern))
        paths.extend(Path(path) for path in matched)
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def _empty_stats() -> dict[str, int]:
    return {
        "rows": 0,
        "benign": 0,
        "malicious": 0,
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "unknown": 0,
        "benign_unknown": 0,
        "malicious_unknown": 0,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _miss_key(miss: dict[str, Any]) -> str:
    return "|".join(
        [
            str(miss.get("kind") or ""),
            str(miss.get("source") or ""),
            str(miss.get("url") or ""),
        ]
    )


def evaluate(probes: list[Path], reference_keys: set[str]) -> dict[str, Any]:
    model, status = load_url_ml_model()
    if model is None:
        raise RuntimeError(status.reason)

    seen: dict[str, str] = {}
    overlap: list[str] = []
    conflicts: list[str] = []
    duplicates = 0
    stats = _empty_stats()
    by_source: dict[str, dict[str, int]] = {}
    by_probe: dict[str, dict[str, int]] = {}
    misses: list[dict[str, Any]] = []

    for probe in probes:
        probe_key = _display(probe)
        by_probe.setdefault(probe_key, _empty_stats())
        for row in _read_rows(probe):
            url = (row.get("url") or "").strip()
            label = str(row.get("label") or "").strip()
            source = (row.get("source") or probe.stem).strip() or probe.stem
            key = canonical_url_key(url)
            if not url or label not in {"0", "1"} or not key:
                continue
            if key in reference_keys:
                overlap.append(key)
                continue
            previous = seen.get(key)
            if previous is not None:
                duplicates += 1
                if previous != label:
                    conflicts.append(key)
                continue
            seen[key] = label

            out = predict_url_ml(model, url)
            pred = str(out.get("verdict") or "unknown")
            expected_malicious = label == "1"
            source_stats = by_source.setdefault(source, _empty_stats())
            for target in (stats, source_stats, by_probe[probe_key]):
                target["rows"] += 1
                target["malicious" if expected_malicious else "benign"] += 1

            if pred == "malicious" and expected_malicious:
                for target in (stats, source_stats, by_probe[probe_key]):
                    target["tp"] += 1
            elif pred == "benign" and not expected_malicious:
                for target in (stats, source_stats, by_probe[probe_key]):
                    target["tn"] += 1
            elif pred == "malicious":
                for target in (stats, source_stats, by_probe[probe_key]):
                    target["fp"] += 1
                misses.append({"kind": "fp", "url": url, "source": source, "probe": probe_key, "result": out})
            elif pred == "benign":
                for target in (stats, source_stats, by_probe[probe_key]):
                    target["fn"] += 1
                misses.append({"kind": "fn", "url": url, "source": source, "probe": probe_key, "result": out})
            else:
                field = "malicious_unknown" if expected_malicious else "benign_unknown"
                for target in (stats, source_stats, by_probe[probe_key]):
                    target["unknown"] += 1
                    target[field] += 1
                misses.append({"kind": field, "url": url, "source": source, "probe": probe_key, "result": out})

    decisive = stats["tp"] + stats["tn"] + stats["fp"] + stats["fn"]
    stats["accuracy_ppm"] = int(round(_rate(stats["tp"] + stats["tn"], max(1, stats["rows"])) * 1_000_000))
    stats["coverage_ppm"] = int(round(_rate(decisive, max(1, stats["rows"])) * 1_000_000))
    return {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "probes": [_display(path) for path in probes],
        "reference_rows": len(reference_keys),
        "overlap": overlap[:50],
        "label_conflicts": conflicts[:50],
        "duplicates_skipped": duplicates,
        "stats": stats,
        "by_source": dict(sorted(by_source.items())),
        "by_probe": dict(sorted(by_probe.items())),
        "misses": misses[:100],
    }


def _failures(report: dict[str, Any], args: argparse.Namespace) -> list[str]:
    stats = report["stats"]
    failures: list[str] = []
    checks = [
        ("rows", stats["rows"], args.min_rows, "lt"),
        ("malicious", stats["malicious"], args.min_malicious, "lt"),
        ("benign", stats["benign"], args.min_benign, "lt"),
        ("fp", stats["fp"], args.max_fp, "gt"),
        ("fn", stats["fn"], args.max_fn, "gt"),
        ("malicious_unknown", stats["malicious_unknown"], args.max_malicious_unknown, "gt"),
        ("benign_unknown", stats["benign_unknown"], args.max_benign_unknown, "gt"),
    ]
    for name, value, threshold, direction in checks:
        if direction == "lt" and value < threshold:
            failures.append(f"{name}={value} < {threshold}")
        elif direction == "gt" and value > threshold:
            failures.append(f"{name}={value} > {threshold}")
    if report["overlap"]:
        failures.append(f"reference_overlap={len(report['overlap'])}")
    if report["label_conflicts"]:
        failures.append(f"label_conflicts={len(report['label_conflicts'])}")
    allowed_misses = set(args.allowed_residual_miss or [])
    unexpected_misses = [
        _miss_key(miss)
        for miss in report.get("misses", [])
        if _miss_key(miss) not in allowed_misses
    ]
    if unexpected_misses:
        failures.append(f"unexpected_residual_misses={len(unexpected_misses)}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--probe-glob", action="append", default=[DEFAULT_PROBE_GLOB])
    parser.add_argument("--reference", action="append", default=DEFAULT_REFERENCE_FILES)
    parser.add_argument("--out", default="dev/model_quality/live_probe_quality.json")
    parser.add_argument("--min-rows", type=int, default=7000)
    parser.add_argument("--min-malicious", type=int, default=3500)
    parser.add_argument("--min-benign", type=int, default=3500)
    parser.add_argument("--max-fp", type=int, default=4)
    parser.add_argument("--max-fn", type=int, default=0)
    parser.add_argument("--max-malicious-unknown", type=int, default=0)
    parser.add_argument("--max-benign-unknown", type=int, default=0)
    parser.add_argument(
        "--allowed-residual-miss",
        action="append",
        default=DEFAULT_ALLOWED_RESIDUAL_MISSES,
        help="Allowed residual miss fingerprint: kind|source|url. New residual misses fail the gate.",
    )
    args = parser.parse_args(argv)

    probes = [BASE / path for path in args.probe] if args.probe else _discover_probes(args.probe_glob)
    if not probes:
        raise SystemExit("no live probe files found")
    references = [BASE / path for path in args.reference]
    report = evaluate(probes, _read_reference_keys(references))
    allowed_misses = set(args.allowed_residual_miss or [])
    report["allowed_residual_misses"] = sorted(allowed_misses)
    report["unexpected_residual_misses"] = [
        _miss_key(miss)
        for miss in report.get("misses", [])
        if _miss_key(miss) not in allowed_misses
    ]
    out = BASE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    stats = report["stats"]
    print(
        "live_probe rows={rows} malicious={malicious} benign={benign} "
        "TP={tp} TN={tn} FP={fp} FN={fn} unknown={unknown} "
        "malicious_unknown={malicious_unknown} benign_unknown={benign_unknown} "
        "accuracy_ppm={accuracy_ppm} coverage_ppm={coverage_ppm}".format(**stats)
    )
    for miss in report["misses"][:20]:
        print(f"  {miss['kind'].upper()} {miss['source']} {miss['url']}")
    failures = _failures(report, args)
    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"PASS live probe gates report={_display(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
