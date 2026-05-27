#!/usr/bin/env python3
"""Evaluate API endpoints on a labeled holdout CSV.

The input CSV must contain at least url,label where label is 0=benign and
1=malicious.  Optional --train files are used only for leakage checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DEPS = os.path.join(BASE, ".codex_deps")
if os.path.isdir(LOCAL_DEPS) and LOCAL_DEPS not in sys.path:
    sys.path.insert(0, LOCAL_DEPS)
URL_ML_DIR = os.path.join(BASE, "url_ml")
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)

from train_url_ml import canonical_url_key

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


ENDPOINTS = {
    "urlml": "/analyze/url-ml",
    "xgboost": "/analyze/xgboost",
    "gnn": "/analyze/gnn",
    "kobert": "/analyze/engine",
    "ensemble": "/analyze",
}
BATCH_ENDPOINTS = {
    "urlml": "/analyze/url-ml/batch",
    "ensemble": "/analyze/batch",
}

Z_95_ONE_SIDED = 1.6448536269514722
_HTTP_LOCAL = threading.local()


@dataclass(frozen=True)
class Row:
    url: str
    label: int
    key: str
    source: str


def read_labeled(paths: list[str], limit: int = 0, per_label_limit: int = 0) -> list[Row]:
    rows: list[Row] = []
    seen: set[str] = set()
    per_label_counts = {0: 0, 1: 0}
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            for raw in csv.DictReader(handle):
                url = (raw.get("url") or "").strip()
                label = str(raw.get("label") or "").strip()
                key = canonical_url_key(url)
                if not url or label not in {"0", "1"} or not key or key in seen:
                    continue
                label_int = int(label)
                if per_label_limit and per_label_counts[label_int] >= per_label_limit:
                    if all(count >= per_label_limit for count in per_label_counts.values()):
                        return rows
                    continue
                seen.add(key)
                per_label_counts[label_int] += 1
                rows.append(
                    Row(
                        url=url,
                        label=label_int,
                        key=key,
                        source=(raw.get("source") or "unknown_source").strip() or "unknown_source",
                    )
                )
                if limit and len(rows) >= limit:
                    return rows
    return rows


def read_keys(paths: list[str]) -> set[str]:
    return {row.key for row in read_labeled(paths)}


def check_no_leakage(train_paths: list[str], rows: list[Row]) -> None:
    if not train_paths:
        return
    train_keys = read_keys(train_paths)
    overlap = [row for row in rows if row.key in train_keys]
    if not overlap:
        return
    print(f"FAIL leakage overlap={len(overlap)}")
    for row in overlap[:20]:
        print(f"  OVERLAP {row.url}")
    raise SystemExit(1)


def normalize_risk(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"1", "true", "malicious", "dangerous", "unnormal", "phishing"}:
        return "malicious"
    if text in {"0", "false", "benign", "safe", "normal", "low"}:
        return "benign"
    if text in {"unknown", "unavailable", "skipped", ""}:
        return "unknown"
    return "unknown"


def risk_from_payload(engine: str, payload: dict[str, Any]) -> str:
    if engine == "ensemble":
        for key in ("riskLevel", "risklevel", "verdict", "judgment"):
            if key in payload:
                return normalize_risk(payload.get(key))
    nested_key = {
        "urlml": "url_ml",
        "xgboost": "xgboost",
        "gnn": "gnn",
    }.get(engine)
    candidates: list[dict[str, Any]] = []
    if nested_key and isinstance(payload.get(nested_key), dict):
        candidates.append(payload[nested_key])
    candidates.append(payload)
    for obj in candidates:
        for key in ("riskLevel", "risklevel", "verdict", "judgment", "label"):
            if key in obj:
                return normalize_risk(obj.get(key))
    return "unknown"


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return ordered[idx]


def wilson_lower_bound(successes: int, total: int, z: float = Z_95_ONE_SIDED) -> float:
    if total <= 0:
        return 0.0
    p_hat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = p_hat + z2 / (2.0 * total)
    margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total) ** 0.5
    return max(0.0, (center - margin) / denominator)


def post_json(api: str, endpoint: str, url: str, timeout: float) -> tuple[dict[str, Any], float, float | None]:
    if requests is not None:
        session = getattr(_HTTP_LOCAL, "session", None)
        if session is None:
            session = requests.Session()
            _HTTP_LOCAL.session = session
        t0 = time.perf_counter()
        response = session.post(
            f"{api.rstrip('/')}{endpoint}",
            json={"url": url},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        server_header = response.headers.get("X-Process-Time")
        latency = time.perf_counter() - t0
        server_latency = float(server_header) if server_header else None
        return payload, latency, server_latency

    body = json.dumps({"url": url}).encode("utf-8")
    request = urllib.request.Request(
        f"{api.rstrip('/')}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
        server_header = response.headers.get("X-Process-Time")
    latency = time.perf_counter() - t0
    server_latency = float(server_header) if server_header else None
    return payload, latency, server_latency


def post_batch_json(api: str, endpoint: str, urls: list[str], timeout: float) -> tuple[dict[str, Any], float, float | None]:
    if requests is not None:
        session = getattr(_HTTP_LOCAL, "session", None)
        if session is None:
            session = requests.Session()
            _HTTP_LOCAL.session = session
        t0 = time.perf_counter()
        response = session.post(
            f"{api.rstrip('/')}{endpoint}",
            json={"urls": urls},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        server_header = response.headers.get("X-Process-Time")
        latency = time.perf_counter() - t0
        server_latency = float(server_header) if server_header else None
        return payload, latency, server_latency

    body = json.dumps({"urls": urls}).encode("utf-8")
    request = urllib.request.Request(
        f"{api.rstrip('/')}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
        server_header = response.headers.get("X-Process-Time")
    latency = time.perf_counter() - t0
    server_latency = float(server_header) if server_header else None
    return payload, latency, server_latency


def get_ready(api: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(f"{api.rstrip('/')}/ready", method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def find_free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found from {start}")


def wait_ready(api: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = ""
    while time.time() < deadline:
        try:
            ready = get_ready(api, timeout=2.0)
            if ready.get("ready"):
                return
            last_error = json.dumps(ready.get("issues") or ready, ensure_ascii=False)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"server not ready after {timeout:.1f}s: {last_error}")


def tail_file(path: str, max_lines: int = 80) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    return "".join(lines[-max_lines:]).rstrip() if lines else "<empty>"


def evaluate_endpoint(
    engine: str,
    endpoint: str,
    api: str,
    rows: list[Row],
    timeout: float,
    workers: int,
    progress_every: int,
) -> dict[str, Any]:
    predictions: list[tuple[Row, str, float, float | None, str | None]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(post_json, api, endpoint, row.url, timeout): row
            for row in rows
        }
        done = 0
        for future in as_completed(futures):
            row = futures[future]
            done += 1
            try:
                payload, latency, server_latency = future.result()
                pred = risk_from_payload(engine, payload)
                error = None
            except (urllib.error.URLError, TimeoutError, Exception) as exc:
                pred = "unknown"
                latency = timeout
                server_latency = None
                error = str(exc)
            predictions.append((row, pred, latency, server_latency, error))
            if progress_every > 0 and (done % progress_every == 0 or done == len(rows)):
                print(f"  {engine} [{done}/{len(rows)}]", flush=True)

    tp = tn = fp = fn = unknown = 0
    errors = 0
    misses: list[tuple[Row, str, str | None]] = []
    latencies: list[float] = []
    server_latencies: list[float] = []
    source_stats: dict[str, dict[str, int]] = {}
    for row, pred, latency, server_latency, error in predictions:
        stats = source_stats.setdefault(
            row.source,
            {"rows": 0, "malicious": 0, "benign": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "unknown": 0, "errors": 0},
        )
        stats["rows"] += 1
        stats["malicious" if row.label else "benign"] += 1
        latencies.append(latency)
        if server_latency is not None:
            server_latencies.append(server_latency)
        if error:
            errors += 1
            stats["errors"] += 1
        if pred == "unknown":
            unknown += 1
            stats["unknown"] += 1
            misses.append((row, pred, error))
        elif row.label == 1 and pred == "malicious":
            tp += 1
            stats["tp"] += 1
        elif row.label == 0 and pred == "benign":
            tn += 1
            stats["tn"] += 1
        elif row.label == 0 and pred == "malicious":
            fp += 1
            stats["fp"] += 1
            misses.append((row, pred, error))
        elif row.label == 1 and pred == "benign":
            fn += 1
            stats["fn"] += 1
            misses.append((row, pred, error))
        else:
            unknown += 1
            stats["unknown"] += 1
            misses.append((row, pred, error))

    decisive = tp + tn + fp + fn
    total = max(1, len(rows))
    malicious_total = max(1, sum(1 for row, *_ in predictions if row.label == 1))
    benign_total = max(1, sum(1 for row, *_ in predictions if row.label == 0))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    result = {
        "engine": engine,
        "rows": len(rows),
        "accuracy": (tp + tn) / total,
        "accuracy_lower_95": wilson_lower_bound(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "recall_lower_95": wilson_lower_bound(tp, malicious_total),
        "specificity": specificity,
        "specificity_lower_95": wilson_lower_bound(tn, benign_total),
        "false_positive": fp,
        "false_positive_rate": fp / benign_total,
        "false_negative": fn,
        "false_negative_rate": fn / malicious_total,
        "unknown": unknown,
        "unknown_ratio": unknown / total,
        "decisive_accuracy": (tp + tn) / max(1, decisive),
        "coverage": decisive / total,
        "p95_latency_ms": p95(latencies) * 1000.0,
        "p95_server_ms": p95(server_latencies) * 1000.0 if server_latencies else None,
        "errors": errors,
        "misses": misses,
        "tp": tp,
        "tn": tn,
        "source_stats": source_stats,
    }
    return result


def _row_chunks(rows: list[Row], size: int) -> list[list[Row]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def _batch_item_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("results")
    if not isinstance(items, list):
        return []
    ordered: list[tuple[int, dict[str, Any]]] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        index = item.get("index", position)
        result = item.get("result")
        if isinstance(result, dict):
            ordered.append((int(index), result))
    return [result for _index, result in sorted(ordered, key=lambda pair: pair[0])]


def evaluate_batch_endpoint(
    engine: str,
    endpoint: str,
    api: str,
    rows: list[Row],
    timeout: float,
    workers: int,
    progress_every: int,
    batch_size: int,
) -> dict[str, Any]:
    predictions: list[tuple[Row, str, float, float | None, str | None]] = []
    request_latencies: list[float] = []
    request_server_latencies: list[float] = []
    chunks = _row_chunks(rows, batch_size)
    next_progress = progress_every if progress_every > 0 else 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(post_batch_json, api, endpoint, [row.url for row in chunk], timeout): chunk
            for chunk in chunks
        }
        done_rows = 0
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                payload, latency, server_latency = future.result()
                item_payloads = _batch_item_payloads(payload)
                if len(item_payloads) != len(chunk):
                    raise RuntimeError(f"batch result length {len(item_payloads)} != request length {len(chunk)}")
                request_latencies.append(latency)
                if server_latency is not None:
                    request_server_latencies.append(server_latency)
                per_item_latency = latency / max(1, len(chunk))
                per_item_server_latency = (
                    server_latency / max(1, len(chunk))
                    if server_latency is not None
                    else None
                )
                for row, item_payload in zip(chunk, item_payloads):
                    predictions.append(
                        (
                            row,
                            risk_from_payload(engine, item_payload),
                            per_item_latency,
                            per_item_server_latency,
                            None,
                        )
                    )
            except (urllib.error.URLError, TimeoutError, Exception) as exc:
                error = str(exc)
                for row in chunk:
                    predictions.append((row, "unknown", timeout, None, error))
            done_rows += len(chunk)
            if progress_every > 0 and (done_rows >= next_progress or done_rows >= len(rows)):
                print(f"  {engine}/batch [{min(done_rows, len(rows))}/{len(rows)}]", flush=True)
                while next_progress <= done_rows:
                    next_progress += progress_every

    result = summarize_predictions(engine, rows, predictions)
    result["batch_size"] = batch_size
    result["batch_requests"] = len(chunks)
    result["batch_request_p95_ms"] = p95(request_latencies) * 1000.0
    result["batch_request_server_p95_ms"] = (
        p95(request_server_latencies) * 1000.0 if request_server_latencies else None
    )
    return result


def summarize_predictions(
    engine: str,
    rows: list[Row],
    predictions: list[tuple[Row, str, float, float | None, str | None]],
) -> dict[str, Any]:
    tp = tn = fp = fn = unknown = 0
    errors = 0
    misses: list[tuple[Row, str, str | None]] = []
    latencies: list[float] = []
    server_latencies: list[float] = []
    source_stats: dict[str, dict[str, int]] = {}
    for row, pred, latency, server_latency, error in predictions:
        stats = source_stats.setdefault(
            row.source,
            {"rows": 0, "malicious": 0, "benign": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0, "unknown": 0, "errors": 0},
        )
        stats["rows"] += 1
        stats["malicious" if row.label else "benign"] += 1
        latencies.append(latency)
        if server_latency is not None:
            server_latencies.append(server_latency)
        if error:
            errors += 1
            stats["errors"] += 1
        if pred == "unknown":
            unknown += 1
            stats["unknown"] += 1
            misses.append((row, pred, error))
        elif row.label == 1 and pred == "malicious":
            tp += 1
            stats["tp"] += 1
        elif row.label == 0 and pred == "benign":
            tn += 1
            stats["tn"] += 1
        elif row.label == 0 and pred == "malicious":
            fp += 1
            stats["fp"] += 1
            misses.append((row, pred, error))
        elif row.label == 1 and pred == "benign":
            fn += 1
            stats["fn"] += 1
            misses.append((row, pred, error))
        else:
            unknown += 1
            stats["unknown"] += 1
            misses.append((row, pred, error))

    decisive = tp + tn + fp + fn
    total = max(1, len(rows))
    malicious_total = max(1, sum(1 for row, *_ in predictions if row.label == 1))
    benign_total = max(1, sum(1 for row, *_ in predictions if row.label == 0))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "engine": engine,
        "rows": len(rows),
        "accuracy": (tp + tn) / total,
        "accuracy_lower_95": wilson_lower_bound(tp + tn, total),
        "precision": precision,
        "recall": recall,
        "recall_lower_95": wilson_lower_bound(tp, malicious_total),
        "specificity": specificity,
        "specificity_lower_95": wilson_lower_bound(tn, benign_total),
        "false_positive": fp,
        "false_positive_rate": fp / benign_total,
        "false_negative": fn,
        "false_negative_rate": fn / malicious_total,
        "unknown": unknown,
        "unknown_ratio": unknown / total,
        "decisive_accuracy": (tp + tn) / max(1, decisive),
        "coverage": decisive / total,
        "p95_latency_ms": p95(latencies) * 1000.0,
        "p95_server_ms": p95(server_latencies) * 1000.0 if server_latencies else None,
        "errors": errors,
        "misses": misses,
        "tp": tp,
        "tn": tn,
        "source_stats": source_stats,
    }


def print_result(result: dict[str, Any], show_misses: int) -> None:
    server_p95 = result["p95_server_ms"]
    server_text = "NA" if server_p95 is None else f"{server_p95:.1f}"
    print(
        f"{result['engine']}: rows={result['rows']} accuracy={result['accuracy']:.4f} "
        f"accuracy_lower_95={result['accuracy_lower_95']:.6f} "
        f"precision={result['precision']:.4f} recall={result['recall']:.4f} "
        f"recall_lower_95={result['recall_lower_95']:.6f} "
        f"specificity={result['specificity']:.4f} "
        f"specificity_lower_95={result['specificity_lower_95']:.6f} "
        f"FP={result['false_positive']} FP_rate={result['false_positive_rate']:.4f} "
        f"FN={result['false_negative']} FN_rate={result['false_negative_rate']:.4f} "
        f"unknown={result['unknown']} "
        f"unknown_ratio={result['unknown_ratio']:.4f} "
        f"coverage={result['coverage']:.4f} decisive_accuracy={result['decisive_accuracy']:.4f} "
        f"p95_ms={result['p95_latency_ms']:.1f} server_p95_ms={server_text} "
        f"errors={result['errors']}"
    )
    if result.get("batch_size"):
        batch_server_p95 = result.get("batch_request_server_p95_ms")
        batch_server_text = "NA" if batch_server_p95 is None else f"{batch_server_p95:.1f}"
        print(
            f"  BATCH requests={result.get('batch_requests')} batch_size={result.get('batch_size')} "
            f"request_p95_ms={result.get('batch_request_p95_ms', 0.0):.1f} "
            f"request_server_p95_ms={batch_server_text} "
            f"per_item_p95_ms={result['p95_latency_ms']:.3f}"
        )
    for row, pred, error in result["misses"][:show_misses]:
        expected = "malicious" if row.label else "benign"
        suffix = f" error={error}" if error else ""
        print(f"  MISS expected={expected} pred={pred} url={row.url}{suffix}")
    for source, stats in sorted(result["source_stats"].items(), key=lambda item: (-item[1]["rows"], item[0])):
        print(
            f"  SOURCE {source}: rows={stats['rows']} malicious={stats['malicious']} benign={stats['benign']} "
            f"TP={stats['tp']} TN={stats['tn']} FP={stats['fp']} FN={stats['fn']} "
            f"unknown={stats['unknown']} errors={stats['errors']}"
        )


def json_safe_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "misses"}


def fail_if_below(name: str, actual: float, expected: float) -> bool:
    if actual >= expected:
        return False
    print(f"FAIL {name} {actual:.4f} < {expected:.4f}")
    return True


def fail_if_above(name: str, actual: float, expected: float) -> bool:
    if actual <= expected:
        return False
    print(f"FAIL {name} {actual:.4f} > {expected:.4f}")
    return True


def engine_threshold(default: float, specs: list[str], engine: str) -> float:
    value = default
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"invalid engine threshold {spec!r}; use engine=value")
        name, raw = spec.split("=", 1)
        if name.strip() == engine:
            value = float(raw)
    return value


def engine_int_threshold(default: int, specs: list[str], engine: str) -> int:
    value = default
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"invalid engine threshold {spec!r}; use engine=value")
        name, raw = spec.split("=", 1)
        if name.strip() == engine:
            value = int(raw)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--spawn-server", action="store_true", help="Start a temporary uvicorn server for this gate.")
    parser.add_argument("--port", type=int, default=0, help="Port for --spawn-server; 0 selects a free port.")
    parser.add_argument("--port-start", type=int, default=8060)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument("--holdout", action="append", required=True)
    parser.add_argument("--train", action="append", default=[])
    parser.add_argument("--engine", action="append", choices=sorted(ENDPOINTS), default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-label-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Evaluate batch-capable engines with /batch endpoints using this many URLs per request.",
    )
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N completed requests; 0 disables progress.")
    parser.add_argument("--show-misses", type=int, default=20)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-recall", type=float, default=0.0)
    parser.add_argument("--max-fp", type=int, default=-1)
    parser.add_argument("--max-fn", type=int, default=-1)
    parser.add_argument("--max-unknown-ratio", type=float, default=1.0)
    parser.add_argument("--max-p95-ms", type=float, default=0.0)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--max-fp-rate", type=float, default=1.0)
    parser.add_argument("--max-fn-rate", type=float, default=1.0)
    parser.add_argument("--engine-min-precision", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-min-recall", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-min-accuracy", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-max-fp", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-max-fn", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-max-fp-rate", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-max-fn-rate", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-max-unknown-ratio", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-max-p95-ms", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-min-accuracy-lower-bound", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-min-recall-lower-bound", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--engine-min-specificity-lower-bound", action="append", default=[], help="Override as engine=value.")
    parser.add_argument("--json-out", default="", help="Write machine-readable engine summaries to this JSON file.")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    proc: subprocess.Popen[str] | None = None
    log_dir: str | None = None
    stdout_handle = None
    stderr_handle = None
    success = False
    if args.spawn_server:
        port = args.port or find_free_port(args.port_start)
        args.api = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        if os.path.isdir(LOCAL_DEPS):
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = LOCAL_DEPS if not existing_pythonpath else f"{LOCAL_DEPS}{os.pathsep}{existing_pythonpath}"
        env.setdefault("LOG_REQUEST_TIMING", "0")
        env.setdefault("ANALYZE_VERBOSE_LOGS", "0")
        env.setdefault("STARTUP_SMOKE_GNN", "0")
        env.setdefault("XGBOOST_WORKERS", "2")
        env.setdefault("GNN_WORKERS", "2")
        env.setdefault("XG_DEBUG_LOGS", "0")
        log_dir = tempfile.mkdtemp(prefix="api_holdout_eval_")
        stdout_path = os.path.join(log_dir, "server.out.log")
        stderr_path = os.path.join(log_dir, "server.err.log")
        stdout_handle = open(stdout_path, "w", encoding="utf-8", errors="replace")
        stderr_handle = open(stderr_path, "w", encoding="utf-8", errors="replace")
        print(f"starting temp server on {args.api}", flush=True)
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=BASE,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        try:
            wait_ready(args.api, args.ready_timeout)
        except Exception:
            print("--- temp server stdout tail ---")
            print(tail_file(stdout_path))
            print("--- temp server stderr tail ---")
            print(tail_file(stderr_path))
            raise

    try:
        if args.require_ready:
            ready = get_ready(args.api, args.timeout)
            if not ready.get("ready"):
                print("FAIL /ready returned ready=false")
                print(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True))
                return 1

        rows = read_labeled(args.holdout, limit=args.limit, per_label_limit=args.per_label_limit)
        if not rows:
            print("FAIL no labeled holdout rows")
            return 1
        check_no_leakage(args.train, rows)
        positives = sum(row.label for row in rows)
        print(f"holdout_rows={len(rows)} malicious={positives} benign={len(rows) - positives}")

        engines = args.engine or ["urlml", "xgboost", "gnn", "kobert", "ensemble"]
        failed = False
        results: list[dict[str, Any]] = []
        for engine in engines:
            if args.batch_size > 1 and engine in BATCH_ENDPOINTS:
                result = evaluate_batch_endpoint(
                    engine,
                    BATCH_ENDPOINTS[engine],
                    args.api,
                    rows,
                    args.timeout,
                    args.workers,
                    args.progress_every,
                    args.batch_size,
                )
            else:
                result = evaluate_endpoint(
                    engine,
                    ENDPOINTS[engine],
                    args.api,
                    rows,
                    args.timeout,
                    args.workers,
                    args.progress_every,
                )
            results.append(json_safe_result(result))
            print_result(result, args.show_misses)
            min_precision = engine_threshold(args.min_precision, args.engine_min_precision, engine)
            min_recall = engine_threshold(args.min_recall, args.engine_min_recall, engine)
            min_accuracy = engine_threshold(args.min_accuracy, args.engine_min_accuracy, engine)
            max_unknown_ratio = engine_threshold(args.max_unknown_ratio, args.engine_max_unknown_ratio, engine)
            max_fp_rate = engine_threshold(args.max_fp_rate, args.engine_max_fp_rate, engine)
            max_fn_rate = engine_threshold(args.max_fn_rate, args.engine_max_fn_rate, engine)
            max_p95_ms = engine_threshold(args.max_p95_ms, args.engine_max_p95_ms, engine)
            min_accuracy_lower_bound = engine_threshold(0.0, args.engine_min_accuracy_lower_bound, engine)
            min_recall_lower_bound = engine_threshold(0.0, args.engine_min_recall_lower_bound, engine)
            min_specificity_lower_bound = engine_threshold(0.0, args.engine_min_specificity_lower_bound, engine)
            max_fp = engine_int_threshold(args.max_fp, args.engine_max_fp, engine)
            max_fn = engine_int_threshold(args.max_fn, args.engine_max_fn, engine)
            failed |= fail_if_below(f"{engine}.accuracy", result["accuracy"], min_accuracy)
            failed |= fail_if_below(
                f"{engine}.accuracy_lower_95",
                result["accuracy_lower_95"],
                min_accuracy_lower_bound,
            )
            failed |= fail_if_below(
                f"{engine}.recall_lower_95",
                result["recall_lower_95"],
                min_recall_lower_bound,
            )
            failed |= fail_if_below(
                f"{engine}.specificity_lower_95",
                result["specificity_lower_95"],
                min_specificity_lower_bound,
            )
            failed |= fail_if_below(f"{engine}.precision", result["precision"], min_precision)
            failed |= fail_if_below(f"{engine}.recall", result["recall"], min_recall)
            failed |= fail_if_above(f"{engine}.false_positive_rate", result["false_positive_rate"], max_fp_rate)
            failed |= fail_if_above(f"{engine}.false_negative_rate", result["false_negative_rate"], max_fn_rate)
            failed |= fail_if_above(f"{engine}.unknown_ratio", result["unknown_ratio"], max_unknown_ratio)
            if max_fp >= 0 and result["false_positive"] > max_fp:
                print(f"FAIL {engine}.false_positive {result['false_positive']} > {max_fp}")
                failed = True
            if max_fn >= 0 and result["false_negative"] > max_fn:
                print(f"FAIL {engine}.false_negative {result['false_negative']} > {max_fn}")
                failed = True
            if max_p95_ms and result["p95_latency_ms"] > max_p95_ms:
                print(f"FAIL {engine}.p95_latency_ms {result['p95_latency_ms']:.1f} > {max_p95_ms:.1f}")
                failed = True

        if failed:
            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump({"ok": False, "rows": len(rows), "results": results}, f, ensure_ascii=False, indent=2, sort_keys=True)
            return 1
        success = True
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump({"ok": True, "rows": len(rows), "results": results}, f, ensure_ascii=False, indent=2, sort_keys=True)
        print("PASS api holdout evaluation")
        return 0
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=8)
        if stdout_handle is not None:
            stdout_handle.close()
        if stderr_handle is not None:
            stderr_handle.close()
        if log_dir and not success:
            print("--- temp server stdout tail ---")
            print(tail_file(os.path.join(log_dir, "server.out.log")))
            print("--- temp server stderr tail ---")
            print(tail_file(os.path.join(log_dir, "server.err.log")))
        if log_dir:
            shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
