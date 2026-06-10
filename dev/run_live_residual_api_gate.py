#!/usr/bin/env python3
"""Verify managed live-probe residual URLs through production API paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parents[1]
DEV = BASE / "dev"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from evaluate_api_batch_consistency import _cleanup, _spawn_server  # noqa: E402
from evaluate_api_holdout import BATCH_ENDPOINTS, ENDPOINTS, _batch_item_payloads, find_free_port, get_ready, post_batch_json, post_json, risk_from_payload  # noqa: E402


DEFAULT_REPORT = BASE / "dev" / "model_quality" / "live_probe_quality.json"
DEFAULT_JSON_OUT = BASE / "dev" / "model_quality" / "live_residual_api_quality.json"


def _read_residual_urls(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    urls: list[str] = []
    for miss in payload.get("misses") or []:
        if not isinstance(miss, dict):
            continue
        key = f"{miss.get('kind')}|{miss.get('source')}|{miss.get('url')}"
        if key in set(payload.get("allowed_residual_misses") or []):
            url = str(miss.get("url") or "").strip()
            if url:
                urls.append(url)
    return sorted(dict.fromkeys(urls))


def _evaluate(api: str, urls: list[str], timeout: float, batch_size: int) -> dict[str, Any]:
    engines = ["urlml", "ensemble"]
    results: dict[str, Any] = {}
    for engine in engines:
        single: list[dict[str, str]] = []
        for url in urls:
            payload, _latency, _server_latency = post_json(api, ENDPOINTS[engine], url, timeout)
            single.append({"url": url, "risk": risk_from_payload(engine, payload)})

        batch: list[dict[str, str]] = []
        endpoint = BATCH_ENDPOINTS[engine]
        for start in range(0, len(urls), max(1, batch_size)):
            chunk = urls[start : start + max(1, batch_size)]
            payload, _latency, _server_latency = post_batch_json(api, endpoint, chunk, timeout)
            item_payloads = _batch_item_payloads(payload)
            if len(item_payloads) != len(chunk):
                batch.append({"url": ",".join(chunk), "risk": "unknown"})
                continue
            for url, item in zip(chunk, item_payloads):
                batch.append({"url": url, "risk": risk_from_payload(engine, item)})
        results[engine] = {"single": single, "batch": batch}
    return {"urls": urls, "results": results}


def _failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    urls = report.get("urls") or []
    if not urls:
        failures.append("residual_url_count=0 (probe report lists no allowed residual misses)")
    for engine, result in (report.get("results") or {}).items():
        for mode in ("single", "batch"):
            rows = result.get(mode) or []
            if len(rows) != len(urls):
                failures.append(f"{engine}.{mode}.rows={len(rows)} != {len(urls)}")
            for row in rows:
                if row.get("risk") != "malicious":
                    failures.append(f"{engine}.{mode}.{row.get('url')}={row.get('risk')} != malicious")
    if set((report.get("results") or {}).keys()) != {"urlml", "ensemble"}:
        failures.append(f"engines={sorted((report.get('results') or {}).keys())}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-report", default=str(DEFAULT_REPORT))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON_OUT))
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--spawn-server", action="store_true", default=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-start", type=int, default=8130)
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)

    urls = _read_residual_urls(Path(args.probe_report))
    proc = None
    log_dir = None
    stdout_handle = None
    stderr_handle = None
    success = False
    try:
        if args.spawn_server:
            port = args.port or find_free_port(args.port_start)
            args.api, proc, log_dir, stdout_handle, stderr_handle = _spawn_server(port, args.ready_timeout)
        ready = get_ready(args.api, args.timeout)
        if not ready.get("ready"):
            print("FAIL /ready returned ready=false")
            print(json.dumps(ready, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
        report = _evaluate(args.api, urls, args.timeout, args.batch_size)
        failures = _failures(report)
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report["ok"] = not failures
        report["failures"] = failures
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        for engine, result in report["results"].items():
            single = ",".join(row["risk"] for row in result["single"])
            batch = ",".join(row["risk"] for row in result["batch"])
            print(f"{engine}: residual single=[{single}] batch=[{batch}]")
        if failures:
            for failure in failures:
                print(f"FAIL {failure}")
            return 1
        success = True
        print(f"PASS live residual API gate report={out.relative_to(BASE)}")
        return 0
    finally:
        _cleanup(proc, log_dir, stdout_handle, stderr_handle, success)


if __name__ == "__main__":
    raise SystemExit(main())
