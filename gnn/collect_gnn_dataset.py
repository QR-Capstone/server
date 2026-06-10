#!/usr/bin/env python3
"""
Collect web-structure GNN training rows.

Examples:

  python collect_gnn_dataset.py --add-url https://example.com --label 0 --source manual
  python collect_gnn_dataset.py --input urls.csv
  python collect_gnn_dataset.py --refresh-existing --only-label 1 --drop-dead-phishing

The CSV remains compatible with regenerate_gnn_model.py. If feature columns are
present, training uses these stored collection-time graph features instead of
re-fetching dead phishing pages later.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional
from urllib.parse import unquote

from gnn_engine import FEATURE_NAMES, build_web_graph, feature_map_from_graph

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ML_DIR = os.path.join(BASE_DIR, "url_ml")
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)

from train_url_ml import canonical_url_key

def _canonical_key_variants(url: str) -> set[str]:
    key = canonical_url_key(url)
    variants = {key} if key else set()
    decoded_key = canonical_url_key(unquote(url or ""))
    if decoded_key:
        variants.add(decoded_key)
    return variants


BASE_COLUMNS = [
    "url",
    "label",
    "source",
    "collected_at",
    "final_url",
    "status_code",
    "fetch_error",
    "fetch_method",
    "graph_nodes",
    "graph_edges",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _read_excluded_keys(paths: List[str]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                keys.update(_canonical_key_variants(row.get("url") or ""))
    return keys


def _canonical_url_key(url: str) -> str:
    return canonical_url_key(url)


def _fieldnames(rows: Iterable[Dict[str, str]]) -> List[str]:
    names: List[str] = []
    for name in BASE_COLUMNS + FEATURE_NAMES:
        if name not in names:
            names.append(name)
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    return names


def _write_csv_atomic(path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = _fieldnames(rows)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".gnn_dataset_", suffix=".csv", dir=directory)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _is_dead_phishing(row: Dict[str, str]) -> bool:
    if str(row.get("label", "")).strip() != "1":
        return False
    status_raw = str(row.get("status_code", "")).strip()
    fetch_error = str(row.get("fetch_error", "")).strip()
    try:
        status = int(float(status_raw)) if status_raw else 0
    except ValueError:
        status = 0
    return bool(fetch_error) or status == 0 or status >= 400


def _collect_row(row: Dict[str, str], fetch: bool = True) -> Dict[str, str]:
    url = (row.get("url") or "").strip()
    if not url:
        return row
    out = dict(row)
    graph = build_web_graph(url, fetch=fetch)
    features = feature_map_from_graph(graph)
    out["url"] = graph.page_url
    out["collected_at"] = _now_iso()
    out["final_url"] = graph.final_url
    out["status_code"] = str(graph.status)
    out["fetch_error"] = graph.fetch_error or ""
    out["fetch_method"] = graph.fetch_method
    out["graph_nodes"] = str(len(graph.nodes))
    out["graph_edges"] = str(len(graph.edges))
    for name in FEATURE_NAMES:
        out[name] = f"{float(features.get(name, 0.0)):.10g}"
    return out


def _merge_rows(existing: List[Dict[str, str]], new_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_url: Dict[str, Dict[str, str]] = {}
    order: List[str] = []
    for row in existing + new_rows:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        key = _canonical_url_key(url)
        if key not in by_url:
            order.append(key)
        by_url[key] = row
    return [by_url[key] for key in order]


def _filter_excluded_rows(rows: List[Dict[str, str]], excluded_keys: set[str]) -> List[Dict[str, str]]:
    if not excluded_keys:
        return rows
    return [
        row
        for row in rows
        if not (_canonical_key_variants(row.get("url") or "") & excluded_keys)
    ]


def _load_input_rows(path: str, default_source: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = "url" in sample.splitlines()[0].lower() if sample.splitlines() else False
        if has_header:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                if not row.get("url"):
                    continue
                rows.append(
                    {
                        "url": (row.get("url") or "").strip(),
                        "label": str(row.get("label", "")).strip(),
                        "source": (row.get("source") or default_source).strip(),
                    }
                )
            return rows
        rows = []
        for line in f:
            url = line.strip()
            if url:
                rows.append({"url": url, "label": "", "source": default_source})
        return rows


def _collect_many(rows: List[Dict[str, str]], workers: int, *, fetch: bool = True) -> List[Dict[str, str]]:
    collected: List[Optional[Dict[str, str]]] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        future_map = {pool.submit(_collect_row, row, fetch): i for i, row in enumerate(rows)}
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                collected[idx] = future.result()
            except Exception as e:
                failed = dict(rows[idx])
                failed["collected_at"] = _now_iso()
                failed["fetch_error"] = str(e)
                failed["status_code"] = "0"
                collected[idx] = failed
    return [row for row in collected if row is not None]


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect/update GNN web-structure dataset CSV.")
    parser.add_argument("--csv", default="gnn_total_dataset.csv", help="Dataset CSV path")
    parser.add_argument("--add-url", default=None, help="Single URL to collect and append/update")
    parser.add_argument("--label", choices=["0", "1"], default=None, help="0=benign, 1=phishing")
    parser.add_argument("--source", default="manual", help="Source name for new rows")
    parser.add_argument("--input", default=None, help="CSV or newline URL file. CSV may contain url,label,source.")
    parser.add_argument("--refresh-existing", action="store_true", help="Re-fetch existing CSV rows")
    parser.add_argument("--only-label", choices=["0", "1"], default=None, help="Refresh only one label")
    parser.add_argument("--drop-dead-phishing", action="store_true", help="Remove phishing rows that fail fetch or return >=400")
    parser.add_argument("--exclude-csv", action="append", default=[], help="CSV whose canonical URL keys must be pruned from the dataset.")
    parser.add_argument("--no-fetch", action="store_true", help="Build URL-structure features without live HTTP fetches.")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base, args.csv)
    existing = _read_csv(csv_path)
    exclude_paths = [path if os.path.isabs(path) else os.path.join(base, path) for path in args.exclude_csv]
    excluded_keys = _read_excluded_keys(exclude_paths)
    if excluded_keys:
        before_exclude = len(existing)
        existing = _filter_excluded_rows(existing, excluded_keys)
        print(f"Excluded keys: {len(excluded_keys)}")
        print(f"Pruned existing rows: {before_exclude - len(existing)}")

    targets: List[Dict[str, str]] = []
    if args.add_url:
        if args.label is None:
            raise SystemExit("--label is required with --add-url")
        targets.append({"url": args.add_url.strip(), "label": args.label, "source": args.source})
    if args.input:
        input_path = args.input if os.path.isabs(args.input) else os.path.join(base, args.input)
        targets.extend(_filter_excluded_rows(_load_input_rows(input_path, args.source), excluded_keys))
    if args.refresh_existing:
        for row in existing:
            if args.only_label is not None and str(row.get("label", "")).strip() != args.only_label:
                continue
            targets.append(row)

    if not targets:
        if excluded_keys:
            _write_csv_atomic(csv_path, existing)
            print(f"Wrote: {csv_path}")
            print("Collected: 0")
            print(f"Total rows: {len(existing)}")
            return 0
        print("No rows to collect. Use --add-url, --input, or --refresh-existing.")
        return 0

    for row in targets:
        if not row.get("label"):
            raise SystemExit(f"missing label for URL: {row.get('url')}")

    print(f"Collecting {len(targets)} URL(s) with workers={args.workers} ...")
    collected = _collect_many(targets, args.workers, fetch=not args.no_fetch)
    merged = _merge_rows(existing, collected)
    before_drop = len(merged)
    if args.drop_dead_phishing:
        merged = [row for row in merged if not _is_dead_phishing(row)]
    _write_csv_atomic(csv_path, merged)

    dead_count = sum(1 for row in collected if _is_dead_phishing(row))
    print(f"Wrote: {csv_path}")
    print(f"Collected: {len(collected)}")
    print(f"Dead phishing seen: {dead_count}")
    if args.drop_dead_phishing:
        print(f"Dropped rows: {before_drop - len(merged)}")
    print(f"Total rows: {len(merged)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
