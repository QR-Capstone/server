#!/usr/bin/env python3
"""Run API latency, cache, and runtime regression gates against a temp server."""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def wait_ready(api: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{api}/ready", timeout=2.0) as response:
                if response.status == 200 and b'"ready":true' in response.read().replace(b" ", b""):
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"server not ready after {timeout_sec:.1f}s: {last_error}")


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=BASE, check=True)


def bench(name: str, api: str, extra: list[str]) -> None:
    print(f"\n=== gate: {name} ===", flush=True)
    try:
        run([sys.executable, os.path.join(BASE, "dev", "benchmark_api_latency.py"), "--api", api, *extra])
    except subprocess.CalledProcessError as exc:
        print(f"FAIL gate {name}: exit_code={exc.returncode}", flush=True)
        raise


def run_benchmark_self_test() -> None:
    run([sys.executable, os.path.join(BASE, "dev", "benchmark_api_latency.py"), "--self-test"])


def tail_file(path: str, max_lines: int = 80) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return f"<could not read {path}: {exc}>"
    if not lines:
        return "<empty>"
    return "".join(lines[-max_lines:]).rstrip()


def dump_server_logs(stdout_path: str, stderr_path: str) -> None:
    print("\n--- temp server stdout tail ---", flush=True)
    print(tail_file(stdout_path), flush=True)
    print("\n--- temp server stderr tail ---", flush=True)
    print(tail_file(stderr_path), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0, help="0 selects a free port starting at --port-start.")
    parser.add_argument("--port-start", type=int, default=8050)
    parser.add_argument("--ready-timeout", type=float, default=75.0)
    parser.add_argument("--fast-p95-ms", type=float, default=20.0)
    parser.add_argument("--same-domain-p95-ms", type=float, default=20.0)
    parser.add_argument("--skip-gnn-fetch", action="store_true")
    args = parser.parse_args()

    port = args.port or find_free_port(args.port_start)
    api = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.setdefault("LOG_REQUEST_TIMING", "0")
    env.setdefault("ANALYZE_VERBOSE_LOGS", "0")
    env.setdefault("STARTUP_SMOKE_GNN", "0")
    env.setdefault("XGBOOST_WORKERS", "2")
    env.setdefault("GNN_WORKERS", "2")
    env.setdefault("XG_DEBUG_LOGS", "0")

    run_benchmark_self_test()

    print(f"starting temp server on {api}", flush=True)
    log_dir = tempfile.mkdtemp(prefix="api_perf_gate_")
    stdout_path = os.path.join(log_dir, "server.out.log")
    stderr_path = os.path.join(log_dir, "server.err.log")
    stdout_handle = open(stdout_path, "w", encoding="utf-8", errors="replace")
    stderr_handle = open(stderr_path, "w", encoding="utf-8", errors="replace")
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
    )
    success = False
    try:
        wait_ready(api, args.ready_timeout)

        bench(
            "full analyze mixed workload",
            api,
            [
                "--endpoint",
                "/analyze",
                "--repeat",
                "8",
                "--workers",
                "6",
                "--keepalive",
                "--timeout",
                "60",
                "--expect-risk",
                "SAFE=16",
                "--expect-risk",
                "DANGEROUS=8",
                "--max-server-p95-ms",
                str(args.fast_p95_ms),
                "--expect-ready",
                "runtime.xgboost_workers=2",
                "--expect-ready",
                "runtime.gnn_workers=2",
                "--expect-ready",
                "runtime.xgboost.rdap_max_attempts=1",
                "--expect-ready",
                "runtime.xgboost.lock_max>=1",
                "--expect-ready",
                "runtime.gnn.fetch_lock_max>=1",
                "--expect-ready",
                "cache.url_ml.misses=3",
                "--expect-ready",
                "cache.url_ml.hits>=1",
                "--expect-ready",
                "cache.url_rule.misses=3",
                "--expect-ready",
                "cache.url_rule.hits>=1",
                "--expect-ready",
                "cache.url_heuristic.misses=3",
                "--expect-ready",
                "cache.url_heuristic.hits>=1",
                "--expect-ready",
                "inflight.total=0",
            ],
        )

        bench(
            "trusted docs full analyze fast path",
            api,
            [
                "--endpoint",
                "/analyze",
                "--repeat",
                "1",
                "--workers",
                "4",
                "--keepalive",
                "--timeout",
                "60",
                "--url",
                "https://docs.python.org/3/library/urllib.parse.html",
                "--url",
                "https://docs.python.org/3/library/asyncio.html",
                "--url",
                "https://docs.python.org/3/library/http.html",
                "--url",
                "https://docs.python.org/3/library/json.html",
                "--expect-risk",
                "SAFE=4",
                "--max-server-p95-ms",
                str(args.same_domain_p95_ms),
            ],
        )

        bench(
            "trusted docs url-ml only fast path",
            api,
            [
                "--endpoint",
                "/analyze/url-ml",
                "--repeat",
                "1",
                "--workers",
                "4",
                "--keepalive",
                "--timeout",
                "30",
                "--url",
                "https://docs.python.org/3/library/urllib.parse.html",
                "--url",
                "https://docs.python.org/3/library/asyncio.html",
                "--url",
                "https://docs.python.org/3/library/http.html",
                "--url",
                "https://docs.python.org/3/library/json.html",
                "--expect-risk",
                "SAFE=4",
                "--max-server-p95-ms",
                str(args.same_domain_p95_ms),
            ],
        )

        bench(
            "idn confusable brand lure url-ml strong rule",
            api,
            [
                "--endpoint",
                "/analyze/url-ml",
                "--repeat",
                "1",
                "--workers",
                "4",
                "--keepalive",
                "--timeout",
                "30",
                "--url",
                "https://\u0440\u0430\u0443\u0440\u0430l.com/login",
                "--url",
                "https://\u0430\u0440\u0440\u04cf\u0435.com/verify",
                "--url",
                "https://www.g\u03bf\u03bfgle.com/signin",
                "--url",
                "https://\uccad\uc8fc\uacfc\uc678.com/login",
                "--expect-risk",
                "DANGEROUS=3",
                "--expect-risk",
                "SAFE=1",
                "--max-server-p95-ms",
                "10.0",
            ],
        )

        print("\n=== gate: batch analyze dedupe fast path ===", flush=True)
        run(
            [
                sys.executable,
                os.path.join(BASE, "dev", "benchmark_batch_api.py"),
                "--api",
                api,
                "--repeat",
                "5",
                "--timeout",
                "60",
                "--expect-count",
                "6",
                "--expect-unique-count",
                "3",
                "--expect-risk",
                "SAFE=2",
                "--expect-risk",
                "DANGEROUS=4",
                "--max-server-p95-ms",
                str(args.same_domain_p95_ms),
                "--expect-ready",
                "runtime.analyze_batch_workers>=1",
                "--expect-ready",
                "runtime.analyze_batch_max_urls>=6",
                "--expect-ready",
                "cache.url_ml.hits>=1",
                "--expect-ready",
                "inflight.total=0",
            ]
        )

        print("\n=== gate: urlml batch dedupe ultra-fast path ===", flush=True)
        run(
            [
                sys.executable,
                os.path.join(BASE, "dev", "benchmark_batch_api.py"),
                "--api",
                api,
                "--endpoint",
                "/analyze/url-ml/batch",
                "--repeat",
                "5",
                "--timeout",
                "30",
                "--expect-count",
                "6",
                "--expect-unique-count",
                "3",
                "--expect-risk",
                "SAFE=2",
                "--expect-risk",
                "DANGEROUS=4",
                "--max-server-p95-ms",
                "10.0",
                "--expect-ready",
                "cache.url_ml.hits>=1",
                "--expect-ready",
                "inflight.total=0",
            ]
        )

        print("\n=== gate: urlml text extraction ultra-fast path ===", flush=True)
        run(
            [
                sys.executable,
                os.path.join(BASE, "dev", "benchmark_batch_api.py"),
                "--api",
                api,
                "--endpoint",
                "/analyze/url-ml/text",
                "--repeat",
                "5",
                "--timeout",
                "30",
                "--expect-count",
                "6",
                "--expect-unique-count",
                "3",
                "--expect-risk",
                "SAFE=2",
                "--expect-risk",
                "DANGEROUS=4",
                "--max-server-p95-ms",
                "10.0",
                "--expect-ready",
                "runtime.analyze_text_max_chars>=1000",
                "--expect-ready",
                "cache.url_ml.hits>=1",
                "--expect-ready",
                "inflight.total=0",
            ]
        )

        print("\n=== gate: defanged urlml text extraction path ===", flush=True)
        run(
            [
                sys.executable,
                os.path.join(BASE, "dev", "benchmark_batch_api.py"),
                "--api",
                api,
                "--endpoint",
                "/analyze/url-ml/text",
                "--defanged-text",
                "--repeat",
                "5",
                "--timeout",
                "30",
                "--expect-count",
                "6",
                "--expect-unique-count",
                "3",
                "--expect-risk",
                "SAFE=2",
                "--expect-risk",
                "DANGEROUS=4",
                "--max-server-p95-ms",
                "10.0",
                "--expect-ready",
                "cache.url_ml.hits>=1",
                "--expect-ready",
                "inflight.total=0",
            ]
        )

        print("\n=== gate: full text extraction dedupe fast path ===", flush=True)
        run(
            [
                sys.executable,
                os.path.join(BASE, "dev", "benchmark_batch_api.py"),
                "--api",
                api,
                "--endpoint",
                "/analyze/text",
                "--repeat",
                "3",
                "--timeout",
                "60",
                "--expect-count",
                "6",
                "--expect-unique-count",
                "3",
                "--expect-risk",
                "SAFE=2",
                "--expect-risk",
                "DANGEROUS=4",
                "--max-server-p95-ms",
                str(args.same_domain_p95_ms),
                "--expect-ready",
                "cache.url_ml.hits>=1",
                "--expect-ready",
                "inflight.total=0",
            ]
        )

        bench(
            "xgboost urlml preflight fast path",
            api,
            [
                "--endpoint",
                "/analyze/xgboost",
                "--repeat",
                "8",
                "--workers",
                "6",
                "--keepalive",
                "--timeout",
                "60",
                "--url",
                "https://docs.python.org/3/library/urllib.parse.html",
                "--expect-risk",
                "SAFE=8",
                "--max-server-p95-ms",
                str(args.same_domain_p95_ms),
                "--expect-ready",
                "cache.xgboost.misses=0",
                "--expect-ready",
                "cache.xgboost.currsize=0",
                "--expect-ready",
                "cache.url_ml.hits>=1",
                "--expect-ready",
                "inflight.total=0",
            ],
        )

        bench(
            "gnn urlml preflight fast path",
            api,
            [
                "--endpoint",
                "/analyze/gnn",
                "--repeat",
                "6",
                "--workers",
                "4",
                "--keepalive",
                "--timeout",
                "90",
                "--url",
                "https://docs.python.org/3/library/urllib.parse.html",
                "--expect-risk",
                "SAFE=6",
                "--max-server-p95-ms",
                str(args.same_domain_p95_ms),
                "--expect-ready",
                "cache.gnn.misses=0",
                "--expect-ready",
                "cache.gnn.currsize=0",
                "--expect-ready",
                "cache.url_ml.hits>=1",
                "--expect-ready",
                "inflight.total=0",
            ],
        )

        if not args.skip_gnn_fetch:
            bench(
                "gnn fragment fetch cache",
                api,
                [
                    "--endpoint",
                    "/analyze/gnn",
                    "--repeat",
                    "1",
                    "--workers",
                    "1",
                    "--keepalive",
                    "--timeout",
                    "90",
                    "--url",
                    "https://news.ycombinator.com/item?id=123456#one",
                    "--url",
                    "https://news.ycombinator.com/item?id=123456#two",
                    "--expect-ready",
                    "runtime.gnn.fetch_cache_misses=1",
                    "--expect-ready",
                    "runtime.gnn.fetch_cache_hits=1",
                    "--expect-ready",
                    "runtime.gnn.fetch_cache_currsize=1",
                    "--expect-ready",
                    "runtime.gnn.fetch_lock_count=1",
                ],
            )

        print("PASS api performance gate")
        success = True
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=8)
        stdout_handle.close()
        stderr_handle.close()
        if not success:
            dump_server_logs(stdout_path, stderr_path)
        shutil.rmtree(log_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
