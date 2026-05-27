#!/usr/bin/env python3
"""Run URLML leakage, holdout quality, and micro-speed gates."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL_ML_DIR = os.path.join(BASE, "url_ml")
if BASE not in sys.path:
    sys.path.insert(0, BASE)
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)

from trusted_domains import strong_url_phishing_score, url_heuristic_phishing_score
from url_ml_engine import load_url_ml_model, predict_url_ml


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=BASE, env=env, check=True)


def micro_speed(model_path: str, max_ms: float) -> None:
    env = os.environ.copy()
    env["URL_ML_MODEL_PATH"] = model_path
    old = os.environ.get("URL_ML_MODEL_PATH")
    os.environ["URL_ML_MODEL_PATH"] = model_path
    try:
        model, status = load_url_ml_model(model_path)
        if model is None:
            raise RuntimeError(status.reason)
        urls = [
            "https://www.naver.com",
            "https://warasuto.com/a/b",
            "https://t-mobile.converselrqmc.help/pay/",
            "https://www.2345665432.vercel.app/",
            "https://godaddysites.com",
        ] * 1000
        t0 = time.perf_counter()
        for url in urls:
            strong_url_phishing_score(url)
            url_heuristic_phishing_score(url)
            predict_url_ml(model, url)
        elapsed = time.perf_counter() - t0
    finally:
        if old is None:
            os.environ.pop("URL_ML_MODEL_PATH", None)
        else:
            os.environ["URL_ML_MODEL_PATH"] = old
    per_url_ms = elapsed * 1000.0 / len(urls)
    print(f"micro_speed checks={len(urls)} elapsed_sec={elapsed:.6f} per_url_ms={per_url_ms:.4f}")
    if per_url_ms > max_ms:
        raise SystemExit(f"FAIL micro speed {per_url_ms:.4f}ms > {max_ms:.4f}ms")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="append", required=True)
    parser.add_argument("--holdout", action="append", required=True)
    parser.add_argument("--tmp-model", default=os.path.join(BASE, "dev", "url_ml_gate_tmp.joblib"))
    parser.add_argument("--min-coverage", type=float, default=0.996)
    parser.add_argument("--min-accuracy", type=float, default=0.0)
    parser.add_argument("--max-fp-rate", type=float, default=1.0)
    parser.add_argument("--max-fn-rate", type=float, default=1.0)
    parser.add_argument("--max-unknown-rate", type=float, default=1.0)
    parser.add_argument("--min-decisive-accuracy", type=float, default=1.0)
    parser.add_argument("--max-fp", type=int, default=0)
    parser.add_argument("--max-fn", type=int, default=0)
    parser.add_argument("--max-micro-ms", type=float, default=0.60)
    args = parser.parse_args()

    cmd = [sys.executable, os.path.join(URL_ML_DIR, "train_url_ml.py")]
    for path in args.train:
        cmd.extend(["--input", path])
    for path in args.holdout:
        cmd.extend(["--holdout", path])
    cmd.extend(
        [
            "--out",
            args.tmp_model,
            "--test-size",
            "0.2",
            "--random-state",
            "42",
            "--hard-benign-weight",
            "8",
            "--user-confirmed-weight",
            "8",
            "--naver-benign-weight",
            "4",
            "--synthetic-weight",
            "0.2",
            "--max-synthetic-malicious-ratio",
            "0.20",
            "--no-class-weight-balanced",
            "--C",
            "4.0",
        ]
    )
    run(cmd)

    env = os.environ.copy()
    env["URL_ML_MODEL_PATH"] = args.tmp_model
    gate = [
        sys.executable,
        os.path.join(BASE, "dev", "urlml_quality_gate.py"),
        "--min-coverage",
        str(args.min_coverage),
        "--min-accuracy",
        str(args.min_accuracy),
        "--max-fp-rate",
        str(args.max_fp_rate),
        "--max-fn-rate",
        str(args.max_fn_rate),
        "--max-unknown-rate",
        str(args.max_unknown_rate),
        "--min-decisive-accuracy",
        str(args.min_decisive_accuracy),
        "--max-fp",
        str(args.max_fp),
        "--max-fn",
        str(args.max_fn),
    ]
    for path in args.holdout:
        gate.extend(["--eval", path])
    run(gate, env=env)
    micro_speed(args.tmp_model, args.max_micro_ms)
    try:
        os.remove(args.tmp_model)
    except FileNotFoundError:
        pass
    print("PASS regression gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
