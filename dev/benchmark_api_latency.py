#!/usr/bin/env python3
"""Measure API latency for phishing endpoints."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx


def post_json(api: str, endpoint: str, url: str, timeout: float) -> tuple[float, dict, float | None]:
    body = json.dumps({"url": url}).encode("utf-8")
    req = urllib.request.Request(
        f"{api.rstrip('/')}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read())
        server_time = response.headers.get("X-Process-Time")
    return time.perf_counter() - t0, payload, float(server_time) if server_time else None


async def post_json_keepalive(client: httpx.AsyncClient, endpoint: str, url: str) -> tuple[float, dict, float | None]:
    t0 = time.perf_counter()
    response = await client.post(endpoint, json={"url": url})
    response.raise_for_status()
    server_time = response.headers.get("X-Process-Time")
    return time.perf_counter() - t0, response.json(), float(server_time) if server_time else None


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[idx]


def normalize_risk_value(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "DANGEROUS" if value else "SAFE"
    if isinstance(value, (int, float)):
        if float(value) == 1.0:
            return "malicious"
        if float(value) == 0.0:
            return "benign"
    text = str(value)
    lowered = text.strip().lower()
    if lowered in {"1", "true", "malicious", "dangerous", "unnormal"}:
        return "malicious" if lowered in {"1", "malicious"} else "DANGEROUS"
    if lowered in {"0", "false", "benign", "safe", "normal"}:
        return "benign" if lowered in {"0", "benign"} else "SAFE"
    return text


def risk_from_payload(payload: dict) -> str:
    for key in ("riskLevel", "risklevel", "verdict", "judgment", "label"):
        value = payload.get(key)
        risk = normalize_risk_value(value)
        if risk:
            return risk
    for nested_key in ("url_ml", "xgboost", "gnn", "koBERT"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            for key in ("riskLevel", "risklevel", "verdict", "judgment", "label"):
                value = nested.get(key)
                risk = normalize_risk_value(value)
                if risk:
                    return risk
    return "unknown"


def get_json(api: str, path: str, timeout: float) -> dict:
    req = urllib.request.Request(f"{api.rstrip('/')}{path}", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def value_at_path(payload: dict, dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def parse_expected_value(text: str) -> Any:
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except ValueError:
        return text


def check_ready_expectations(ready: dict, specs: list[str], *, verbose: bool = True) -> bool:
    ok = True
    operators = ("<=", ">=", "!=", "=", "<", ">")
    for spec in specs:
        op = next((candidate for candidate in operators if candidate in spec), "")
        if not op:
            if verbose:
                print(f"FAIL invalid --expect-ready {spec!r}; use path=value, path<=value, etc.")
            ok = False
            continue
        path, expected_text = spec.split(op, 1)
        path = path.strip()
        expected = parse_expected_value(expected_text.strip())
        try:
            actual = value_at_path(ready, path)
        except KeyError:
            if verbose:
                print(f"FAIL ready path {path!r} missing")
            ok = False
            continue
        passed = False
        if op == "=":
            passed = actual == expected
        elif op == "!=":
            passed = actual != expected
        else:
            try:
                left = float(actual)
                right = float(expected)
            except (TypeError, ValueError):
                if verbose:
                    print(f"FAIL ready path {path!r} value {actual!r} is not numeric for {op}")
                ok = False
                continue
            if op == "<=":
                passed = left <= right
            elif op == ">=":
                passed = left >= right
            elif op == "<":
                passed = left < right
            elif op == ">":
                passed = left > right
        if not passed:
            if verbose:
                print(f"FAIL ready {path} actual={actual!r} {op} expected={expected!r}")
            ok = False
    return ok


def ready_diagnostic(ready: dict) -> str:
    snapshot = {
        "ready": ready.get("ready"),
        "issues": ready.get("issues"),
        "runtime": ready.get("runtime"),
        "cache": ready.get("cache"),
        "inflight": ready.get("inflight"),
    }
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True)


def self_test() -> int:
    ready = {
        "ready": True,
        "runtime": {
            "xgboost_workers": 2,
            "gnn": {
                "fetch_cache_hits": 1,
                "fetch_cache_misses": 1,
                "fetch_cache_currsize": 1,
            },
        },
        "cache": {
            "xgboost": {
                "hits": 4,
                "misses": 3,
            }
        },
    }
    checks = [
        pct([30.0, 10.0, 20.0], 0.50) == 20.0,
        pct([30.0, 10.0, 20.0], 0.95) == 30.0,
        parse_expected_value("true") is True,
        parse_expected_value("false") is False,
        parse_expected_value("null") is None,
        parse_expected_value("3") == 3,
        parse_expected_value("3.5") == 3.5,
        parse_expected_value("SAFE") == "SAFE",
        value_at_path(ready, "runtime.gnn.fetch_cache_hits") == 1,
        normalize_risk_value(1) == "malicious",
        normalize_risk_value(0) == "benign",
        normalize_risk_value("unnormal") == "DANGEROUS",
        normalize_risk_value("normal") == "SAFE",
        risk_from_payload({"riskLevel": "SAFE"}) == "SAFE",
        risk_from_payload({"gnn": {"verdict": "benign"}}) == "benign",
        risk_from_payload({"xgboost": {"label": "malicious"}}) == "malicious",
        risk_from_payload({"xgboost": {"label": 0}}) == "benign",
        check_ready_expectations(
            ready,
            [
                "ready=true",
                "runtime.xgboost_workers=2",
                "runtime.gnn.fetch_cache_hits>=1",
                "runtime.gnn.fetch_cache_misses<=1",
                "cache.xgboost.hits>3",
                "cache.xgboost.misses<4",
                "runtime.gnn.fetch_cache_currsize!=0",
            ],
        ),
        not check_ready_expectations(ready, ["runtime.xgboost_workers=3"], verbose=False),
        not check_ready_expectations(ready, ["runtime.gnn.missing=1"], verbose=False),
        not check_ready_expectations(ready, ["ready<0"], verbose=False),
        '"cache"' in ready_diagnostic(ready),
        '"runtime"' in ready_diagnostic(ready),
    ]
    failed = [str(index + 1) for index, ok in enumerate(checks) if not ok]
    if failed:
        print(f"FAIL self-test checks: {', '.join(failed)}")
        return 1
    print("PASS benchmark_api_latency self-test")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="Run parser/helper self-tests without calling an API.")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/analyze")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--keepalive", action="store_true", help="Use one async HTTP client with pooled keep-alive connections.")
    parser.add_argument("--max-p95-ms", type=float, default=0.0, help="Fail if client p95 exceeds this value. Disabled at 0.")
    parser.add_argument("--max-server-p95-ms", type=float, default=0.0, help="Fail if X-Process-Time p95 exceeds this value. Disabled at 0.")
    parser.add_argument("--ready-path", default="/ready")
    parser.add_argument("--print-ready", action="store_true", help="Print a compact /ready diagnostic snapshot after requests.")
    parser.add_argument(
        "--expect-ready",
        action="append",
        default=[],
        help="Expected /ready dotted path comparison, e.g. cache.xgboost.misses=1 or runtime.xgboost_workers>=2.",
    )
    parser.add_argument(
        "--expect-risk",
        action="append",
        default=[],
        help="Expected risk count as RISK=COUNT, e.g. SAFE=20. Repeatable.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
    )
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not args.url:
        args.url = [
            "https://www.naver.com",
            "https://www.apps-offiice-wps.com.cn/",
            "https://strip2.co",
        ]

    jobs = [url for _ in range(args.repeat) for url in args.url]
    t0 = time.perf_counter()
    if args.keepalive:
        latencies, server_latencies, risks = asyncio.run(run_keepalive(args, jobs))
    else:
        latencies = []
        server_latencies: list[float] = []
        risks: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(post_json, args.api, args.endpoint, url, args.timeout) for url in jobs]
            for future in as_completed(futures):
                latency, payload, server_latency = future.result()
                latencies.append(latency)
                if server_latency is not None:
                    server_latencies.append(server_latency)
                risk = risk_from_payload(payload)
                risks[risk] = risks.get(risk, 0) + 1

    elapsed = time.perf_counter() - t0
    print(
        f"requests={len(jobs)} workers={args.workers} elapsed_sec={elapsed:.3f} "
        f"mean_ms={statistics.mean(latencies) * 1000:.3f} "
        f"p50_ms={pct(latencies, 0.50) * 1000:.3f} "
        f"p95_ms={pct(latencies, 0.95) * 1000:.3f} "
        f"max_ms={max(latencies) * 1000:.3f}"
    )
    print(f"risks={risks}")
    if server_latencies:
        print(
            f"server_header_mean_ms={statistics.mean(server_latencies) * 1000:.3f} "
            f"server_header_p50_ms={pct(server_latencies, 0.50) * 1000:.3f} "
            f"server_header_p95_ms={pct(server_latencies, 0.95) * 1000:.3f} "
            f"server_header_max_ms={max(server_latencies) * 1000:.3f}"
        )
    p95_ms = pct(latencies, 0.95) * 1000.0
    if args.max_p95_ms and p95_ms > args.max_p95_ms:
        print(f"FAIL p95_ms {p95_ms:.3f} > {args.max_p95_ms:.3f}")
        return 1
    if args.max_server_p95_ms:
        if not server_latencies:
            print("FAIL server p95 requested but no X-Process-Time headers were observed")
            return 1
        server_p95_ms = pct(server_latencies, 0.95) * 1000.0
        if server_p95_ms > args.max_server_p95_ms:
            print(f"FAIL server_header_p95_ms {server_p95_ms:.3f} > {args.max_server_p95_ms:.3f}")
            return 1
    for spec in args.expect_risk:
        if "=" not in spec:
            print(f"FAIL invalid --expect-risk {spec!r}; use RISK=COUNT")
            return 1
        risk, count_text = spec.split("=", 1)
        expected = int(count_text)
        actual = int(risks.get(risk, 0))
        if actual != expected:
            print(f"FAIL risk {risk} count {actual} != {expected}")
            return 1
    if args.expect_ready:
        ready = get_json(args.api, args.ready_path, args.timeout)
        if args.print_ready:
            print("ready_diagnostic=" + ready_diagnostic(ready))
        if not check_ready_expectations(ready, args.expect_ready):
            print("ready_diagnostic=" + ready_diagnostic(ready))
            return 1
    elif args.print_ready:
        ready = get_json(args.api, args.ready_path, args.timeout)
        print("ready_diagnostic=" + ready_diagnostic(ready))
    return 0


async def run_keepalive(args: argparse.Namespace, jobs: list[str]) -> tuple[list[float], list[float], dict[str, int]]:
    limits = httpx.Limits(max_connections=args.workers, max_keepalive_connections=args.workers)
    timeout = httpx.Timeout(args.timeout)
    latencies: list[float] = []
    server_latencies: list[float] = []
    risks: dict[str, int] = {}
    semaphore = asyncio.Semaphore(args.workers)
    async with httpx.AsyncClient(base_url=args.api.rstrip("/"), limits=limits, timeout=timeout) as client:
        async def one(url: str) -> None:
            async with semaphore:
                latency, payload, server_latency = await post_json_keepalive(client, args.endpoint, url)
                latencies.append(latency)
                if server_latency is not None:
                    server_latencies.append(server_latency)
                risk = risk_from_payload(payload)
                risks[risk] = risks.get(risk, 0) + 1

        await asyncio.gather(*(one(url) for url in jobs))
    return latencies, server_latencies, risks


if __name__ == "__main__":
    raise SystemExit(main())
