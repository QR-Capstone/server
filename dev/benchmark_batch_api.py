#!/usr/bin/env python3
"""Measure and gate the /analyze/batch endpoint."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from typing import Any

from benchmark_api_latency import get_json, ready_diagnostic, check_ready_expectations, risk_from_payload


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[idx]


def post_payload(api: str, endpoint: str, payload_body: dict[str, Any], timeout: float) -> tuple[float, dict[str, Any], float | None]:
    body = json.dumps(payload_body).encode("utf-8")
    request = urllib.request.Request(
        f"{api.rstrip('/')}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
        server_time = response.headers.get("X-Process-Time")
    return time.perf_counter() - t0, payload, float(server_time) if server_time else None


def post_batch(api: str, endpoint: str, urls: list[str], timeout: float) -> tuple[float, dict[str, Any], float | None]:
    return post_payload(api, endpoint, {"urls": urls}, timeout)


def post_text(api: str, endpoint: str, text: str, timeout: float) -> tuple[float, dict[str, Any], float | None]:
    return post_payload(api, endpoint, {"text": text}, timeout)


def risk_counts(payload: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in payload.get("results") or []:
        result = item.get("result") if isinstance(item, dict) else None
        risk = risk_from_payload(result if isinstance(result, dict) else {})
        counts[risk] = counts.get(risk, 0) + 1
    return counts


def parse_expected_counts(specs: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"invalid --expect-risk {spec!r}; use RISK=COUNT")
        risk, raw_count = spec.split("=", 1)
        parsed[risk] = int(raw_count)
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/analyze/batch")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-server-p95-ms", type=float, default=0.0)
    parser.add_argument("--max-p95-ms", type=float, default=0.0)
    parser.add_argument("--expect-count", type=int, default=0)
    parser.add_argument("--expect-unique-count", type=int, default=0)
    parser.add_argument("--expect-risk", action="append", default=[])
    parser.add_argument("--expect-ready", action="append", default=[])
    parser.add_argument("--url", action="append", default=[])
    parser.add_argument("--text", default="")
    parser.add_argument("--defanged-text", action="store_true")
    args = parser.parse_args()

    urls = args.url or [
        "https://www.naver.com",
        "https://www.apps-offiice-wps.com.cn/",
        "https://www.dpdlocoew.shop/com",
        "https://www.naver.com",
        "https://www.dpdlocoew.shop/com",
        "https://www.apps-offiice-wps.com.cn/",
    ]
    default_text = (
        "정상 링크 https://www.naver.com 과 의심 링크 "
        "https://www.apps-offiice-wps.com.cn/ 를 확인하세요. "
        "중복: https://www.dpdlocoew.shop/com, https://www.naver.com, "
        "https://www.dpdlocoew.shop/com, https://www.apps-offiice-wps.com.cn/"
    )
    defanged_text = (
        "정상 링크 h t t p s : / / w w w . naver . com 과 의심 링크 "
        "hxxps 쌍점 슬래시 슬래시 www[.]apps-offiice-wps 점 com\u200b[.]cn/ 를 확인하세요. "
        "중복: \"=68=74=74=70=73=3A=2F=2F(www)\" + \"\\.dpdlocoew\\.shop/com\", "
        "aHR0cHM6Ly93d3cubmF2ZXIuY29t, "
        "h x x p s[:] [slash] [slash] ｗｗｗ[.]d p d l o c o e w 닷 ｓ ｈ ｏ ｐ / com, "
        "hxxps://www\uff0eapps-offiice-wps 쩜 com[.]cn/"
    )
    text = args.text or (defanged_text if args.defanged_text else default_text)
    expected_risks = parse_expected_counts(args.expect_risk)

    latencies: list[float] = []
    server_latencies: list[float] = []
    last_payload: dict[str, Any] = {}
    for _ in range(max(1, args.repeat)):
        if args.endpoint.endswith("/text"):
            latency, payload, server_latency = post_text(args.api, args.endpoint, text, args.timeout)
        else:
            latency, payload, server_latency = post_batch(args.api, args.endpoint, urls, args.timeout)
        latencies.append(latency)
        if server_latency is not None:
            server_latencies.append(server_latency)
        last_payload = payload

    count = int(last_payload.get("count", 0))
    unique_count = int(last_payload.get("unique_count", 0))
    counts = risk_counts(last_payload)
    print(
        f"batch_requests={len(latencies)} urls_per_batch={len(urls)} count={count} "
        f"unique_count={unique_count} deduplicated={last_payload.get('deduplicated')} "
        f"mean_ms={statistics.mean(latencies) * 1000:.3f} "
        f"p95_ms={pct(latencies, 0.95) * 1000:.3f} max_ms={max(latencies) * 1000:.3f}"
    )
    print(f"risks={counts}")
    if server_latencies:
        print(
            f"server_header_mean_ms={statistics.mean(server_latencies) * 1000:.3f} "
            f"server_header_p95_ms={pct(server_latencies, 0.95) * 1000:.3f} "
            f"server_header_max_ms={max(server_latencies) * 1000:.3f}"
        )

    if args.expect_count and count != args.expect_count:
        print(f"FAIL count {count} != {args.expect_count}")
        return 1
    if args.expect_unique_count and unique_count != args.expect_unique_count:
        print(f"FAIL unique_count {unique_count} != {args.expect_unique_count}")
        return 1
    for risk, expected in expected_risks.items():
        actual = int(counts.get(risk, 0))
        if actual != expected:
            print(f"FAIL risk {risk} count {actual} != {expected}")
            return 1
    if args.max_p95_ms and pct(latencies, 0.95) * 1000.0 > args.max_p95_ms:
        print(f"FAIL p95_ms {pct(latencies, 0.95) * 1000.0:.3f} > {args.max_p95_ms:.3f}")
        return 1
    if args.max_server_p95_ms:
        if not server_latencies:
            print("FAIL server p95 requested but no X-Process-Time headers were observed")
            return 1
        server_p95_ms = pct(server_latencies, 0.95) * 1000.0
        if server_p95_ms > args.max_server_p95_ms:
            print(f"FAIL server_header_p95_ms {server_p95_ms:.3f} > {args.max_server_p95_ms:.3f}")
            return 1
    if args.expect_ready:
        ready = get_json(args.api, "/ready", args.timeout)
        if not check_ready_expectations(ready, args.expect_ready):
            print("ready_diagnostic=" + ready_diagnostic(ready))
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
