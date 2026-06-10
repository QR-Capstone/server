#!/usr/bin/env python3
"""Compare single-request API verdicts with batch API verdicts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from evaluate_api_holdout import (
    BATCH_ENDPOINTS,
    ENDPOINTS,
    LOCAL_DEPS,
    Row,
    _batch_item_payloads,
    _row_chunks,
    find_free_port,
    post_batch_json,
    post_json,
    read_labeled,
    risk_from_payload,
    tail_file,
    wait_ready,
    get_ready,
)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _compare_engine(
    *,
    engine: str,
    api: str,
    rows: list[Row],
    timeout: float,
    batch_size: int,
) -> list[str]:
    mismatches: list[str] = []
    single_risks: list[str] = []
    for row in rows:
        payload, _latency, _server_latency = post_json(api, ENDPOINTS[engine], row.url, timeout)
        single_risks.append(risk_from_payload(engine, payload))

    batch_risks: list[str] = []
    for chunk in _row_chunks(rows, batch_size):
        payload, _latency, _server_latency = post_batch_json(
            api,
            BATCH_ENDPOINTS[engine],
            [row.url for row in chunk],
            timeout,
        )
        raw_items = payload.get("results")
        if not isinstance(raw_items, list):
            mismatches.append(f"{engine}: batch response has no results list")
            continue
        if len(raw_items) != len(chunk):
            mismatches.append(f"{engine}: raw batch result length {len(raw_items)} != request length {len(chunk)}")
            continue
        expected_first_index_by_url: dict[str, int] = {}
        for position, (row, item) in enumerate(zip(chunk, raw_items)):
            if not isinstance(item, dict):
                mismatches.append(f"{engine}: batch item {position} is not an object")
                continue
            if item.get("index") != position:
                mismatches.append(f"{engine}: batch item {position} index={item.get('index')} expected={position}")
            if item.get("url") != row.url:
                mismatches.append(f"{engine}: batch item {position} url={item.get('url')!r} expected={row.url!r}")
            duplicate_of = item.get("duplicate_of")
            first_index = expected_first_index_by_url.setdefault(row.url, position)
            expected_duplicate_of = None if first_index == position else first_index
            if duplicate_of != expected_duplicate_of:
                mismatches.append(
                    f"{engine}: batch item {position} duplicate_of={duplicate_of} "
                    f"expected={expected_duplicate_of} url={row.url}"
                )
        item_payloads = _batch_item_payloads(payload)
        if len(item_payloads) != len(chunk):
            mismatches.append(f"{engine}: batch result length {len(item_payloads)} != request length {len(chunk)}")
            continue
        batch_risks.extend(risk_from_payload(engine, item_payload) for item_payload in item_payloads)

    if len(batch_risks) != len(single_risks):
        mismatches.append(f"{engine}: compared length mismatch single={len(single_risks)} batch={len(batch_risks)}")
        return mismatches

    for row, single_risk, batch_risk in zip(rows, single_risks, batch_risks):
        if single_risk != batch_risk:
            mismatches.append(f"{engine}: single={single_risk} batch={batch_risk} url={row.url}")
    return mismatches


def _with_duplicate_contract_rows(rows: list[Row]) -> list[Row]:
    if not rows:
        return rows
    duplicate = Row(
        url=rows[0].url,
        label=rows[0].label,
        key=f"{rows[0].key}#duplicate-contract",
        source=f"{rows[0].source}:duplicate_contract",
    )
    return [rows[0], duplicate, *rows[1:]]


def _spawn_server(port: int, ready_timeout: float):
    api = f"http://127.0.0.1:{port}"
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
    log_dir = tempfile.mkdtemp(prefix="api_batch_consistency_")
    stdout_path = os.path.join(log_dir, "server.out.log")
    stderr_path = os.path.join(log_dir, "server.err.log")
    stdout_handle = open(stdout_path, "w", encoding="utf-8", errors="replace")
    stderr_handle = open(stderr_path, "w", encoding="utf-8", errors="replace")
    print(f"starting temp server on {api}", flush=True)
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
        wait_ready(api, ready_timeout)
    except Exception:
        print("--- temp server stdout tail ---")
        print(tail_file(stdout_path))
        print("--- temp server stderr tail ---")
        print(tail_file(stderr_path))
        raise
    return api, proc, log_dir, stdout_handle, stderr_handle


def _cleanup(proc, log_dir: str | None, stdout_handle, stderr_handle, success: bool) -> None:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare single and batch API verdicts.")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--spawn-server", action="store_true")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-start", type=int, default=8060)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument("--holdout", action="append", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-label-limit", type=int, default=0)
    parser.add_argument("--engine", action="append", choices=sorted(BATCH_ENDPOINTS), default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--require-ready", action="store_true")
    parser.add_argument("--include-duplicate-contract", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args(argv)

    rows = read_labeled(args.holdout, limit=args.limit, per_label_limit=args.per_label_limit)
    if not rows:
        print("FAIL no labeled rows")
        return 1
    if args.include_duplicate_contract:
        rows = _with_duplicate_contract_rows(rows)

    proc = None
    log_dir = None
    stdout_handle = None
    stderr_handle = None
    success = False
    try:
        if args.spawn_server:
            port = args.port or find_free_port(args.port_start)
            args.api, proc, log_dir, stdout_handle, stderr_handle = _spawn_server(port, args.ready_timeout)
        if args.require_ready:
            ready = get_ready(args.api, args.timeout)
            if not ready.get("ready"):
                print("FAIL /ready returned ready=false")
                print(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True))
                return 1

        engines = args.engine or sorted(BATCH_ENDPOINTS)
        all_mismatches: list[str] = []
        for engine in engines:
            mismatches = _compare_engine(
                engine=engine,
                api=args.api,
                rows=rows,
                timeout=args.timeout,
                batch_size=max(1, args.batch_size),
            )
            if mismatches:
                all_mismatches.extend(mismatches)
                print(f"{engine}: FAIL mismatches={len(mismatches)}")
            else:
                print(f"{engine}: PASS rows={len(rows)} batch_size={args.batch_size}")
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(
                    {"ok": not all_mismatches, "rows": len(rows), "engines": engines, "mismatches": all_mismatches},
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
        if all_mismatches:
            for mismatch in all_mismatches[:50]:
                print("FAIL " + mismatch)
            return 1
        success = True
        print("PASS api batch consistency")
        return 0
    finally:
        _cleanup(proc, log_dir, stdout_handle, stderr_handle, success)


if __name__ == "__main__":
    raise SystemExit(main())
