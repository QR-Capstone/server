#!/usr/bin/env python3
"""Run the current high-confidence non-overlap evidence gates."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


BASE = Path(__file__).resolve().parents[1]
LOCAL_DEPS = BASE / ".codex_deps"


DEFAULT_HOLDOUTS = [
    "dev/nonoverlap_feed_eval_40k_20260524.csv",
    "dev/nonoverlap_crossfeed_eval_20260524.csv",
    "dev/nonoverlap_phishingdb_eval_40k_20260524.csv",
    "dev/nonoverlap_korean_sources_eval_20260524.csv",
]

DEFAULT_MIN_COMBINED_ROWS = 91000
DEFAULT_MIN_SOURCE_COUNTS = {
    "tranco_latest_nonoverlap": 45500,
    "phishing_database_active": 20000,
    "urlhaus_recent": 20000,
    "phishtank": 4705,
    "nurilab": 500,
    "openphish": 295,
}


def canonical_url_key(raw_url: str) -> str:
    """Stable key for duplicate/leakage checks without importing ML deps."""
    raw = (raw_url or "").strip()
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
    except Exception:
        return raw.lower().rstrip("/")
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    query = parsed.query
    if scheme:
        return urlunsplit((scheme, netloc, path, query, "")).lower()
    return urlunsplit(("", netloc, path, query, "")).lower()


def python_env() -> dict[str, str]:
    env = os.environ.copy()
    if LOCAL_DEPS.exists():
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(LOCAL_DEPS) if not existing else f"{LOCAL_DEPS}{os.pathsep}{existing}"
    return env


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=BASE, check=True, env=python_env())


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


def validate_combined_counts(row_count: int, source_counts: dict[str, int], min_rows: int, min_source_counts: dict[str, int]) -> None:
    if row_count < min_rows:
        raise SystemExit(f"combined holdout too small: rows={row_count} < min_rows={min_rows}")
    for source, expected in min_source_counts.items():
        actual = source_counts.get(source, 0)
        if actual < expected:
            raise SystemExit(f"source {source!r} too small: rows={actual} < min_rows={expected}")
    print(
        "combined_count_gate=PASS "
        f"rows={row_count} min_rows={min_rows} "
        f"sources={dict(sorted(source_counts.items()))}",
        flush=True,
    )


def build_combined_holdout(holdouts: list[str], out_path: Path) -> tuple[int, dict[str, int]]:
    seen: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    for holdout in holdouts:
        path = BASE / holdout
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                key = canonical_url_key(url)
                if not url or label not in {"0", "1"} or not key:
                    continue
                previous = seen.get(key)
                if previous is not None:
                    if previous != label:
                        raise SystemExit(f"label conflict in combined holdout: {key}")
                    continue
                seen[key] = label
                source = (row.get("source") or Path(holdout).stem).strip() or Path(holdout).stem
                source_counts[source] = source_counts.get(source, 0) + 1
                rows.append(
                    {
                        "url": url,
                        "label": label,
                        "source": source,
                        "is_korean": row.get("is_korean") or ("1" if ".kr" in url.lower() else "0"),
                        "canonical_key": key,
                    }
                )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "is_korean", "canonical_key"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"combined_holdout={out_path.relative_to(BASE)} rows={len(rows)}", flush=True)
    return len(rows), source_counts


def api_gate_cmd(
    holdout: str,
    show_misses: str = "20",
    progress_every: str = "25",
    min_accuracy_lower_bound: str | None = None,
    min_class_lower_bound: str | None = None,
    json_out: str | None = None,
    batch_size: int = 0,
) -> list[str]:
    cmd = [
        sys.executable,
        "dev/evaluate_api_holdout.py",
        "--spawn-server",
        "--port",
        "0",
        "--holdout",
        holdout,
        "--engine",
        "urlml",
        "--engine",
        "ensemble",
        "--workers",
        "8",
        "--timeout",
        "90",
        "--progress-every",
        progress_every,
        "--ready-timeout",
        "90",
        "--require-ready",
        "--show-misses",
        show_misses,
        "--max-unknown-ratio",
        "0",
        "--min-accuracy",
        "1",
        "--max-fp-rate",
        "0",
        "--max-fn-rate",
        "0",
        "--engine-max-fp",
        "urlml=0",
        "--engine-max-fn",
        "urlml=0",
        "--engine-max-unknown-ratio",
        "urlml=0",
        "--engine-min-accuracy",
        "urlml=1",
        "--engine-max-fp",
        "ensemble=0",
        "--engine-max-fn",
        "ensemble=0",
        "--engine-max-unknown-ratio",
        "ensemble=0",
        "--engine-min-accuracy",
        "ensemble=1",
        "--engine-max-p95-ms",
        "ensemble=6000",
    ]
    if batch_size > 1:
        cmd.extend(
            [
                "--batch-size",
                str(batch_size),
                "--engine-max-p95-ms",
                "urlml=2",
                "--engine-max-p95-ms",
                "ensemble=5",
            ]
        )
    if min_accuracy_lower_bound is not None:
        cmd.extend(
            [
                "--engine-min-accuracy-lower-bound",
                f"urlml={min_accuracy_lower_bound}",
                "--engine-min-accuracy-lower-bound",
                f"ensemble={min_accuracy_lower_bound}",
            ]
        )
    if min_class_lower_bound is not None:
        cmd.extend(
            [
                "--engine-min-recall-lower-bound",
                f"urlml={min_class_lower_bound}",
                "--engine-min-recall-lower-bound",
                f"ensemble={min_class_lower_bound}",
                "--engine-min-specificity-lower-bound",
                f"urlml={min_class_lower_bound}",
                "--engine-min-specificity-lower-bound",
                f"ensemble={min_class_lower_bound}",
            ]
        )
    if json_out is not None:
        cmd.extend(["--json-out", json_out])
    return cmd


def report_gate_cmd(report: str, min_rows: int, min_source_counts: dict[str, int]) -> list[str]:
    cmd = [
        sys.executable,
        "dev/validate_evidence_report.py",
        "--report",
        report,
        "--min-rows",
        str(min_rows),
        "--min-accuracy-lower-bound",
        "0.9999",
        "--min-class-lower-bound",
        "0.9999",
    ]
    for source, count in sorted(min_source_counts.items()):
        cmd.extend(["--min-source-count", f"{source}={count}"])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="dev/dataset_splits/warehouse.csv")
    parser.add_argument("--holdout", action="append", default=[])
    parser.add_argument("--skip-api-smoke", action="store_true")
    parser.add_argument("--full-api", action="store_true", help="Run API verification on the combined evidence holdout.")
    parser.add_argument(
        "--batch-api",
        action="store_true",
        help="Use /analyze/url-ml/batch and /analyze/batch for API verification.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--combined-out", default="dev/evidence_combined_holdout_20260524.csv")
    parser.add_argument("--api-report-out", default="dev/evidence_api_report_20260524.json")
    parser.add_argument("--api-smoke-holdout", default="dev/nonoverlap_korean_sources_eval_20260524.csv")
    parser.add_argument("--min-combined-rows", type=int, default=DEFAULT_MIN_COMBINED_ROWS)
    parser.add_argument("--min-source-count", action="append", default=[], help="Minimum combined rows by source, as source=count.")
    args = parser.parse_args()

    holdouts = args.holdout or DEFAULT_HOLDOUTS
    min_source_counts = parse_min_source_counts(args.min_source_count) if args.min_source_count else (
        DEFAULT_MIN_SOURCE_COUNTS if not args.holdout else {}
    )

    audit_cmd = [sys.executable, "dev/audit_url_splits.py", "--train", args.train]
    for holdout in holdouts:
        audit_cmd.extend(["--test", holdout])
    run(audit_cmd)

    quality_cmd = [
        sys.executable,
        "dev/urlml_quality_gate.py",
        "--train",
        args.train,
        "--min-coverage",
        "1",
        "--min-accuracy",
        "1",
        "--min-accuracy-lower-bound",
        "0.9999",
        "--min-recall-lower-bound",
        "0.9999",
        "--min-specificity-lower-bound",
        "0.9999",
        "--min-decisive-accuracy",
        "1",
        "--max-fp-rate",
        "0",
        "--max-fn-rate",
        "0",
        "--max-unknown-rate",
        "0",
        "--max-fp",
        "0",
        "--max-fn",
        "0",
    ]
    for holdout in holdouts:
        quality_cmd.extend(["--eval", holdout])
    run(quality_cmd)

    if args.full_api:
        combined_path = BASE / args.combined_out
        row_count, source_counts = build_combined_holdout(holdouts, combined_path)
        validate_combined_counts(row_count, source_counts, args.min_combined_rows, min_source_counts)
        run(
            api_gate_cmd(
                args.combined_out,
                show_misses="40",
                progress_every="1000",
                min_accuracy_lower_bound="0.9999",
                min_class_lower_bound="0.9999",
                json_out=args.api_report_out,
                batch_size=args.batch_size if args.batch_api else 0,
            )
        )
        run(report_gate_cmd(args.api_report_out, args.min_combined_rows, min_source_counts))
    elif not args.skip_api_smoke:
        run(api_gate_cmd(args.api_smoke_holdout, batch_size=args.batch_size if args.batch_api else 0))

    print("PASS evidence gates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
