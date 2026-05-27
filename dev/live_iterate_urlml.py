#!/usr/bin/env python3
"""Iteratively collect live URLs, evaluate URLML, and retrain.

Each round:
1. Pull fresh malicious candidates from public feeds and benign candidates from
   Tranco.
2. Exclude every URL already present in known CSVs and prior rounds.
3. Keep only URLs that respond live.
4. Evaluate the current URLML model on that held-out live batch.
5. Add the batch to a cumulative training CSV and retrain URLML.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = os.path.join(BASE, "dev")
URL_ML_DIR = os.path.join(BASE, "url_ml")
for path in (BASE, DEV, URL_ML_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from collect_benign_tranco_fast import alive as benign_alive
from collect_naver_benign import (
    DEFAULT_QUERIES as NAVER_DEFAULT_QUERIES,
    collect_candidates as collect_naver_candidates,
    filter_alive as filter_naver_alive,
)
from collect_urls import (
    fetch_openphish,
    fetch_phishing_database_kr,
    fetch_phishstats_kr,
    fetch_tranco,
    fetch_urlhaus,
    filter_alive as filter_malicious_alive,
    save_csv,
)
from url_ml_engine import load_url_ml_model, predict_url_ml
from train_url_ml import canonical_url_key


def read_labeled_csv(path: str) -> list[dict[str, str]]:
    if not os.path.isfile(path):
        return []
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "url" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            return []
        for row in reader:
            url = (row.get("url") or "").strip()
            label = str(row.get("label", "")).strip()
            if url and label in {"0", "1"}:
                rows.append(
                    {
                        "url": url,
                        "label": label,
                        "source": (row.get("source") or "existing").strip() or "existing",
                        "is_korean": str(row.get("is_korean", "0")).strip() or "0",
                    }
                )
    return rows


def load_exclude(paths: list[str]) -> set[str]:
    urls: set[str] = set()
    for path in paths:
        for row in read_labeled_csv(path):
            key = canonical_url_key(row["url"])
            if key:
                urls.add(key)
    return urls


def write_rows(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "is_korean"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={path} rows={len(rows)}", flush=True)


def assert_disjoint(rows: list[dict[str, str]], existing_keys: set[str], name: str) -> None:
    overlap = []
    seen: set[str] = set()
    duplicates = []
    for row in rows:
        key = canonical_url_key(row["url"])
        if not key:
            continue
        if key in existing_keys:
            overlap.append(row["url"])
        if key in seen:
            duplicates.append(row["url"])
        seen.add(key)
    if overlap or duplicates:
        msg = (
            f"{name} failed disjoint audit: "
            f"overlap={len(overlap)} duplicates={len(duplicates)}"
        )
        for url in (overlap + duplicates)[:20]:
            msg += f"\n  {url}"
        raise RuntimeError(msg)


def merge_unique(paths: list[str], out: str) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    conflicts = 0
    for path in paths:
        for row in read_labeled_csv(path):
            key = canonical_url_key(row["url"])
            if not key:
                continue
            old = seen.get(key)
            if old and old["label"] != row["label"]:
                conflicts += 1
                continue
            seen.setdefault(key, row)
    rows = list(seen.values())
    write_rows(out, rows)
    counts = Counter(row["label"] for row in rows)
    print(f"merged unique={len(rows)} benign={counts['0']} malicious={counts['1']} conflicts={conflicts}", flush=True)
    return rows


def collect_malicious(
    target: int,
    exclude: set[str],
    workers: int,
    timeout: int,
    max_candidates: int,
) -> list[dict[str, str]]:
    if target <= 0:
        return []
    raw: list[str] = []
    for fetcher in (fetch_openphish, fetch_phishstats_kr, fetch_urlhaus, fetch_phishing_database_kr):
        raw.extend(fetcher())
        raw = [url for url in dict.fromkeys(raw) if canonical_url_key(url) not in exclude]
        if len(raw) >= target * 10:
            break
    raw = [url for url in dict.fromkeys(raw) if canonical_url_key(url) not in exclude]
    if max_candidates > 0:
        raw = raw[:max_candidates]
    print(f"malicious candidates={len(raw)} target={target}", flush=True)
    rows = filter_malicious_alive(
        raw,
        label=1,
        workers=workers,
        max_count=target,
        timeout=timeout,
        max_checked=max_candidates if max_candidates > 0 else None,
    )
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "url": row["url"],
                "label": "1",
                "source": "live_iter_malicious",
                "is_korean": "1" if row.get("is_korean") else "0",
            }
        )
    return out


def collect_benign(
    target: int,
    exclude: set[str],
    workers: int,
    timeout: float,
    tranco_n: int,
    max_candidates: int,
    tranco_offset: int,
    naver_fallback: bool,
    naver_queries: int,
) -> list[dict[str, str]]:
    if target <= 0:
        return []
    candidates = [url for url in fetch_tranco(tranco_n) if canonical_url_key(url) not in exclude]
    candidates = list(dict.fromkeys(candidates))
    if tranco_offset > 0:
        candidates = candidates[tranco_offset:]
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    print(f"benign candidates={len(candidates)} target={target}", flush=True)
    rows: list[dict[str, str]] = []
    with __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(benign_alive, url, timeout): url for url in candidates}
        for i, future in enumerate(__import__("concurrent.futures").futures.as_completed(futures), 1):
            url, ok = future.result()
            key = canonical_url_key(url)
            if ok and key and key not in exclude:
                rows.append({"url": url, "label": "0", "source": "live_iter_tranco", "is_korean": "0"})
                exclude.add(key)
                if len(rows) >= target:
                    break
            if i % 250 == 0 or len(rows) >= target:
                print(f"benign checked={i}/{len(candidates)} alive={len(rows)}", flush=True)
    if naver_fallback and len(rows) < target:
        remaining = target - len(rows)
        queries = NAVER_DEFAULT_QUERIES[: max(1, naver_queries)]
        print(f"benign naver fallback target={remaining} queries={len(queries)}", flush=True)
        naver_candidates = collect_naver_candidates(queries, delay=0.05)
        naver_candidates = [
            row
            for row in naver_candidates
            if canonical_url_key(row["url"]) not in exclude
        ]
        if max_candidates > 0:
            naver_candidates = naver_candidates[:max_candidates]
        for row in filter_naver_alive(naver_candidates, remaining, workers, int(max(1, timeout))):
            key = canonical_url_key(row["url"])
            if key and key not in exclude:
                rows.append(
                    {
                        "url": row["url"],
                        "label": "0",
                        "source": row.get("source", "live_iter_naver"),
                        "is_korean": row.get("is_korean", "1"),
                    }
                )
                exclude.add(key)
                if len(rows) >= target:
                    break
    return rows


def evaluate_urlml(rows: list[dict[str, str]]) -> dict[str, int | float]:
    model, status = load_url_ml_model()
    if model is None:
        raise RuntimeError(status.reason)
    tp = tn = fp = fn = unknown = 0
    misses: list[tuple[str, str, str]] = []
    for row in rows:
        out = predict_url_ml(model, row["url"])
        pred = str(out.get("verdict") or "unknown")
        label = row["label"]
        if label == "1" and pred == "malicious":
            tp += 1
        elif label == "0" and pred == "benign":
            tn += 1
        elif label == "0" and pred == "malicious":
            fp += 1
            misses.append((row["url"], label, pred))
        elif label == "1" and pred == "benign":
            fn += 1
            misses.append((row["url"], label, pred))
        else:
            unknown += 1
            misses.append((row["url"], label, pred))
    total = max(1, len(rows))
    valid = max(1, tp + tn + fp + fn)
    metrics = {
        "rows": len(rows),
        "accuracy_all": (tp + tn) / total,
        "accuracy_valid": (tp + tn) / valid,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "unknown": unknown,
        "fp_rate": fp / max(1, fp + tn),
        "fn_rate": fn / max(1, fn + tp),
    }
    print(
        "URLML "
        f"rows={metrics['rows']} acc_all={metrics['accuracy_all']:.4f} "
        f"TP={tp} TN={tn} FP={fp} FN={fn} unknown={unknown} "
        f"fp_rate={metrics['fp_rate']:.4f} fn_rate={metrics['fn_rate']:.4f}",
        flush=True,
    )
    for url, label, pred in misses[:20]:
        print(f"  MISS label={label} pred={pred} url={url}", flush=True)
    return metrics


def retrain(train_csv: str, holdout_csv: str | None = None) -> None:
    cmd = [
        sys.executable,
        os.path.join(URL_ML_DIR, "train_url_ml.py"),
        "--input",
        train_csv,
        "--out",
        os.path.join(URL_ML_DIR, "url_ml_model.joblib"),
        "--test-size",
        "0.2",
        "--random-state",
        "42",
    ]
    if holdout_csv:
        cmd.extend(["--holdout", holdout_csv])
    subprocess.run(cmd, cwd=BASE, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--target-malicious", type=int, default=80)
    parser.add_argument("--target-benign", type=int, default=80)
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--malicious-timeout", type=int, default=4)
    parser.add_argument("--max-malicious-candidates", type=int, default=1200)
    parser.add_argument("--max-benign-candidates", type=int, default=1200)
    parser.add_argument("--tranco-offset", type=int, default=0)
    parser.add_argument("--no-naver-fallback", action="store_true")
    parser.add_argument("--naver-queries", type=int, default=8)
    parser.add_argument("--tranco-n", type=int, default=8000)
    parser.add_argument("--seed", default=os.path.join(DEV, "urlml_retrain_all_unique_20260522.csv"))
    parser.add_argument("--prefix", default="live_iter_20260522")
    args = parser.parse_args()

    train_csv = os.path.join(DEV, f"{args.prefix}_cumulative.csv")
    all_paths = [args.seed]
    exclude = load_exclude([args.seed])
    summary_rows: list[dict[str, str]] = []
    t0 = time.time()

    for round_idx in range(1, args.rounds + 1):
        print(f"\n=== ROUND {round_idx}/{args.rounds} ===", flush=True)
        round_exclude = set(exclude)
        mal = collect_malicious(
            args.target_malicious,
            round_exclude,
            args.workers,
            args.malicious_timeout,
            args.max_malicious_candidates,
        )
        for row in mal:
            key = canonical_url_key(row["url"])
            if key:
                round_exclude.add(key)
        ben = collect_benign(
            args.target_benign,
            round_exclude,
            args.workers,
            args.timeout,
            args.tranco_n,
            args.max_benign_candidates,
            args.tranco_offset,
            not args.no_naver_fallback,
            args.naver_queries,
        )
        rows = mal + ben
        assert_disjoint(rows, exclude, f"round_{round_idx}")
        for row in rows:
            key = canonical_url_key(row["url"])
            if key:
                exclude.add(key)
        round_csv = os.path.join(DEV, f"{args.prefix}_r{round_idx}.csv")
        save_csv(rows, round_csv)
        metrics = evaluate_urlml(rows)
        all_paths.append(round_csv)
        merge_unique(all_paths, train_csv)
        retrain(train_csv)
        summary_rows.append(
            {
                "round": str(round_idx),
                "rows": str(metrics["rows"]),
                "tp": str(metrics["tp"]),
                "tn": str(metrics["tn"]),
                "fp": str(metrics["fp"]),
                "fn": str(metrics["fn"]),
                "unknown": str(metrics["unknown"]),
                "accuracy_all": f"{float(metrics['accuracy_all']):.6f}",
                "fp_rate": f"{float(metrics['fp_rate']):.6f}",
                "fn_rate": f"{float(metrics['fn_rate']):.6f}",
            }
        )

    summary_csv = os.path.join(DEV, f"{args.prefix}_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["round", "rows", "tp", "tn", "fp", "fn", "unknown", "accuracy_all", "fp_rate", "fn_rate"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nwrote={summary_csv} elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
