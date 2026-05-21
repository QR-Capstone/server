#!/usr/bin/env python3
"""
전체 학습 파이프라인: 중복 없이 2000+2000 수집 후 3개 모델 재학습
  배치 1~4: 각 500+500 수집, 이전 배치 제외
  병합 → XGBoost / GNN / KoBERT 순서로 학습

사용법:
  python dev/run_full_pipeline.py
  python dev/run_full_pipeline.py --batch-size 500 --batches 4 --skip-dom
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time

PYTHON = sys.executable
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # server root
DEV = os.path.join(BASE, "dev")

def run(cmd, desc=""):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    t0 = time.time()
    ret = subprocess.run(cmd, cwd=BASE)
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"\n[ERROR] 실패 (exit={ret.returncode}) — {desc} ({elapsed:.0f}초)")
        sys.exit(ret.returncode)
    print(f"\n[OK] {desc} 완료 ({elapsed:.0f}초)")
    return ret


def merge_csvs(paths, out_path):
    """여러 CSV를 중복 없이 병합"""
    seen = set()
    rows = []
    fieldnames = None
    for p in paths:
        if not os.path.isfile(p):
            print(f"[merge] 없음 — 건너뜀: {p}")
            continue
        with open(p, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            for row in reader:
                u = (row.get("url") or "").strip()
                if u and u not in seen:
                    seen.add(u)
                    rows.append(row)
    if not rows:
        print("[merge] 병합할 데이터 없음")
        return 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[merge] 저장: {out_path} ({len(rows)}행)")
    return len(rows)


def count_labels(path):
    mal = ben = 0
    if not os.path.isfile(path):
        return 0, 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lbl = str(row.get("label", "")).strip()
            if lbl == "1": mal += 1
            elif lbl == "0": ben += 1
    return mal, ben


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--skip-dom", action="store_true", default=True)
    parser.add_argument("--kobert-epochs", type=int, default=5)
    parser.add_argument("--skip-collect", action="store_true", help="수집 건너뜀 (기존 CSV 사용)")
    args = parser.parse_args()

    t_total = time.time()

    # ──────────────────────────────────────────
    # Phase 1: 배치 수집 (각 500+500, 중복 제외)
    # ──────────────────────────────────────────
    batch_csvs_all = []
    batch_csvs_kr = []

    if not args.skip_collect:
        exclude_path = ""
        collected_paths = []

        for i in range(1, args.batches + 1):
            suffix = f"_{i}"
            out_all = os.path.join(DEV, f"train_urls_all{suffix}.csv")
            out_kr  = os.path.join(DEV, f"train_urls_korean{suffix}.csv")

            # 이미 수집된 배치면 건너뜀
            if os.path.isfile(out_all):
                mal, ben = count_labels(out_all)
                print(f"\n[배치 {i}] 이미 존재 — 악성 {mal} / 정상 {ben} → 재사용")
                batch_csvs_all.append(out_all)
                batch_csvs_kr.append(out_kr)
                exclude_path = out_all if not collected_paths else os.path.join(DEV, "train_urls_all_merged_tmp.csv")
                # 누적 exclude를 위한 임시 병합
                if len(batch_csvs_all) > 0:
                    merge_csvs(batch_csvs_all, os.path.join(DEV, "train_urls_all_merged_tmp.csv"))
                    exclude_path = os.path.join(DEV, "train_urls_all_merged_tmp.csv")
                continue

            print(f"\n{'#'*60}")
            print(f"  배치 {i}/{args.batches} — {args.batch_size} 악성 + {args.batch_size} 정상")
            if exclude_path:
                print(f"  제외 CSV: {exclude_path}")
            print(f"{'#'*60}")

            cmd = [
                PYTHON, os.path.join(DEV, "collect_urls.py"),
                "--malicious", str(args.batch_size),
                "--benign", str(args.batch_size),
                "--workers", str(args.workers),
                "--out-dir", DEV,
                "--suffix", suffix,
            ]
            if exclude_path:
                cmd += ["--exclude-csv", exclude_path]

            run(cmd, f"배치 {i} URL 수집")

            batch_csvs_all.append(out_all)
            batch_csvs_kr.append(out_kr)

            # 다음 배치를 위한 exclude 병합
            tmp_path = os.path.join(DEV, "train_urls_all_merged_tmp.csv")
            merge_csvs(batch_csvs_all, tmp_path)
            exclude_path = tmp_path

            mal, ben = count_labels(out_all)
            print(f"[배치 {i} 완료] 악성 {mal} / 정상 {ben}")

    # ──────────────────────────────────────────
    # Phase 2: 전체 병합
    # ──────────────────────────────────────────
    # 수집 건너뛰기 옵션이면 기존 파일 사용
    if args.skip_collect:
        for i in range(1, args.batches + 1):
            p = os.path.join(DEV, f"train_urls_all_{i}.csv")
            pk = os.path.join(DEV, f"train_urls_korean_{i}.csv")
            if os.path.isfile(p):
                batch_csvs_all.append(p)
            if os.path.isfile(pk):
                batch_csvs_kr.append(pk)

    merged_all = os.path.join(DEV, "train_urls_all_merged.csv")
    merged_kr  = os.path.join(DEV, "train_urls_korean_merged.csv")

    print(f"\n{'='*60}")
    print(f"  전체 CSV 병합 ({len(batch_csvs_all)}개 배치)")
    print(f"{'='*60}")
    total = merge_csvs(batch_csvs_all, merged_all)
    total_kr = merge_csvs(batch_csvs_kr, merged_kr)

    mal_all, ben_all = count_labels(merged_all)
    mal_kr, ben_kr = count_labels(merged_kr)
    print(f"\n[병합 결과]")
    print(f"  전체:   {total}개 (악성 {mal_all} / 정상 {ben_all})")
    print(f"  한국어: {total_kr}개 (악성 {mal_kr} / 정상 {ben_kr}) → KoBERT용")

    if total == 0:
        print("[ERROR] 병합 데이터 없음 — 학습 중단")
        sys.exit(1)

    # ──────────────────────────────────────────
    # Phase 3: XGBoost 학습
    # ──────────────────────────────────────────
    xg_cmd = [
        PYTHON, os.path.join(BASE, "xgboost", "XG_train.py"),
        "--input", merged_all,
    ]
    if args.skip_dom:
        xg_cmd.append("--skip-dom")
    run(xg_cmd, "XGBoost 3-bundle 재학습")

    # ──────────────────────────────────────────
    # Phase 4: GNN 데이터 수집 + 학습
    # ──────────────────────────────────────────
    gnn_dataset = os.path.join(BASE, "gnn", "gnn_total_dataset.csv")

    run([
        PYTHON, os.path.join(BASE, "gnn", "collect_gnn_dataset.py"),
        "--input", merged_all,
    ], "GNN 데이터셋 구축")

    run([
        PYTHON, os.path.join(BASE, "gnn", "regenerate_gnn_model.py"),
        "--csv", gnn_dataset,
    ], "GNN 모델 재학습")

    # ──────────────────────────────────────────
    # Phase 5: KoBERT 학습
    # ──────────────────────────────────────────
    if total_kr >= 10:
        run([
            PYTHON, os.path.join(BASE, "KoBERT", "kobert_train.py"),
            "--input", merged_kr,
            "--epochs", str(args.kobert_epochs),
            "--workers", "8",
        ], f"KoBERT Fine-tuning (epochs={args.kobert_epochs})")
    else:
        print(f"\n[KoBERT] 한국어 데이터 부족 ({total_kr}개) — 건너뜀")

    print(f"\n{'='*60}")
    print(f"  전체 파이프라인 완료 — 총 소요: {(time.time()-t_total)/60:.1f}분")
    print(f"  학습된 모델:")
    print(f"    XGBoost: xgboost/url_xgb_paired_first.joblib")
    print(f"            xgboost/url_xgb_domain_age.joblib")
    print(f"    GNN:     gnn/gnn_model.pkl")
    print(f"    KoBERT:  KoBERT/kobert_phishing_model_weights.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
