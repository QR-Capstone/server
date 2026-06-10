#!/usr/bin/env python3
"""Run API batch consistency on a source-balanced live probe sample."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


BASE = Path(__file__).resolve().parents[1]
DEFAULT_PROBE_GLOB = "dev/latest_*probe_20260607_*.csv"


def canonical_url_key(raw_url: str) -> str:
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


def discover(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(str(BASE / pattern)))
    return sorted(dict.fromkeys(path.resolve() for path in paths))


def build_source_balanced_sample(probes: list[Path], per_source_label: int) -> list[dict[str, str]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    seen: set[str] = set()
    for path in probes:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label") or "").strip()
                source = (row.get("source") or path.stem).strip() or path.stem
                key = canonical_url_key(url)
                if not url or label not in {"0", "1"} or not key or key in seen:
                    continue
                bucket_key = (source, label)
                if len(buckets[bucket_key]) >= per_source_label:
                    continue
                seen.add(key)
                buckets[bucket_key].append({"url": url, "label": label, "source": source, "canonical_key": key})

    rows: list[dict[str, str]] = []
    for key in sorted(buckets):
        rows.extend(buckets[key])
    rows.sort(key=lambda row: (row["source"], row["label"], row["canonical_key"]))
    return rows


def write_sample(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "canonical_key"])
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="append", default=[])
    parser.add_argument("--probe-glob", action="append", default=[DEFAULT_PROBE_GLOB])
    parser.add_argument("--per-source-label", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--port-start", type=int, default=8090)
    parser.add_argument("--json-out", default="dev/model_quality/live_api_batch_consistency.json")
    args = parser.parse_args(argv)

    probes = [BASE / path for path in args.probe] if args.probe else discover(args.probe_glob)
    if not probes:
        print("FAIL no live probe files found")
        return 1
    rows = build_source_balanced_sample(probes, args.per_source_label)
    if not rows:
        print("FAIL no live probe rows sampled")
        return 1
    labels = {row["label"] for row in rows}
    sources = sorted({row["source"] for row in rows})
    if labels != {"0", "1"}:
        print(f"FAIL sampled labels={sorted(labels)}")
        return 1
    print(f"live_api_batch_sample rows={len(rows)} sources={sources}")

    with tempfile.TemporaryDirectory(prefix="live_api_batch_gate_") as tmp:
        sample_path = Path(tmp) / "live_sample.csv"
        report_path = BASE / args.json_out
        report_path.parent.mkdir(parents=True, exist_ok=True)
        write_sample(rows, sample_path)
        cmd = [
            sys.executable,
            "dev/evaluate_api_batch_consistency.py",
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
            "--include-duplicate-contract",
            "--timeout",
            str(args.timeout),
            "--require-ready",
            "--json-out",
            str(report_path),
        ]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=BASE, check=True)
    print("PASS live API batch gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
