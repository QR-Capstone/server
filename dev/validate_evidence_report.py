#!/usr/bin/env python3
"""Validate the machine-readable full evidence API report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MIN_SOURCE_COUNTS = {
    "tranco_latest_nonoverlap": 45500,
    "phishing_database_active": 20000,
    "urlhaus_recent": 20000,
    "phishtank": 4705,
    "nurilab": 500,
    "openphish": 295,
}


def parse_min_source_counts(specs: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"invalid --min-source-count {spec!r}; use source=count")
        source, raw_count = spec.split("=", 1)
        source = source.strip()
        if not source:
            raise SystemExit(f"invalid --min-source-count {spec!r}; source is empty")
        parsed[source] = int(raw_count)
    return parsed


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def validate_engine(
    result: dict[str, Any],
    min_rows: int,
    min_accuracy_lower_bound: float,
    min_class_lower_bound: float,
    min_source_counts: dict[str, int],
) -> None:
    engine = str(result.get("engine") or "")
    require(engine in {"urlml", "ensemble"}, f"unexpected engine {engine!r}")
    require(int(result.get("rows") or 0) >= min_rows, f"{engine}.rows too small")
    require(float(result.get("accuracy") or 0.0) == 1.0, f"{engine}.accuracy != 1")
    require(float(result.get("coverage") or 0.0) == 1.0, f"{engine}.coverage != 1")
    require(float(result.get("decisive_accuracy") or 0.0) == 1.0, f"{engine}.decisive_accuracy != 1")
    require(
        float(result.get("accuracy_lower_95") or 0.0) >= min_accuracy_lower_bound,
        f"{engine}.accuracy_lower_95 too low",
    )
    require(
        float(result.get("recall_lower_95") or 0.0) >= min_class_lower_bound,
        f"{engine}.recall_lower_95 too low",
    )
    require(
        float(result.get("specificity_lower_95") or 0.0) >= min_class_lower_bound,
        f"{engine}.specificity_lower_95 too low",
    )
    for field in ("false_positive", "false_negative", "unknown", "errors"):
        require(int(result.get(field) or 0) == 0, f"{engine}.{field} != 0")

    source_stats = result.get("source_stats")
    require(isinstance(source_stats, dict), f"{engine}.source_stats missing")
    for source, minimum in min_source_counts.items():
        stats = source_stats.get(source)
        require(isinstance(stats, dict), f"{engine}.source_stats[{source!r}] missing")
        require(int(stats.get("rows") or 0) >= minimum, f"{engine}.{source}.rows too small")
        for field in ("fp", "fn", "unknown", "errors"):
            require(int(stats.get(field) or 0) == 0, f"{engine}.{source}.{field} != 0")


def validate_report(
    report: dict[str, Any],
    min_rows: int,
    min_accuracy_lower_bound: float,
    min_class_lower_bound: float,
    min_source_counts: dict[str, int],
) -> None:
    require(report.get("ok") is True, "report.ok is not true")
    require(int(report.get("rows") or 0) >= min_rows, "report.rows too small")
    results = report.get("results")
    require(isinstance(results, list), "report.results missing")
    engines = {str(result.get("engine") or "") for result in results if isinstance(result, dict)}
    require(engines == {"urlml", "ensemble"}, f"unexpected engines {sorted(engines)!r}")
    for result in results:
        require(isinstance(result, dict), "report result is not an object")
        validate_engine(result, min_rows, min_accuracy_lower_bound, min_class_lower_bound, min_source_counts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="dev/evidence_api_report_20260524.json")
    parser.add_argument("--min-rows", type=int, default=91000)
    parser.add_argument("--min-accuracy-lower-bound", type=float, default=0.9999)
    parser.add_argument("--min-class-lower-bound", type=float, default=0.9999)
    parser.add_argument("--min-source-count", action="append", default=[])
    args = parser.parse_args()

    min_source_counts = parse_min_source_counts(args.min_source_count) if args.min_source_count else DEFAULT_MIN_SOURCE_COUNTS
    with Path(args.report).open("r", encoding="utf-8") as f:
        report = json.load(f)
    validate_report(report, args.min_rows, args.min_accuracy_lower_bound, args.min_class_lower_bound, min_source_counts)
    print(f"PASS evidence report validation report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
