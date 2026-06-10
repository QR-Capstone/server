#!/usr/bin/env python3
"""
전체 학습 파이프라인: 모든 모델을 1만 개 학습 입력 기준으로 재학습
  dev/dataset_splits/warehouse.csv에서 엔진별 10k 입력 생성
  XGBoost / GNN / KoBERT 순서로 최소 10k 조건을 걸어 학습

사용법:
  python dev/run_full_pipeline.py --skip-dom
  python dev/run_full_pipeline.py --target 10000 --kobert-epochs 3
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
ENGINE_INPUTS = os.path.join(DEV, "engine_training_inputs")

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


def count_labeled_rows(path):
    if not os.path.isfile(path):
        return 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return sum(
            1
            for row in csv.DictReader(f)
            if (row.get("url") or "").strip()
            and str(row.get("label", "")).strip() in {"0", "1"}
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--skip-dom", action="store_true", default=True)
    parser.add_argument("--with-dom", action="store_true", help="DOM 모델까지 학습합니다. 1만 URL 크롤링이라 오래 걸립니다.")
    parser.add_argument("--skip-domain", action="store_true", help="도메인 나이/SSL XGBoost 모델 학습을 건너뜁니다.")
    parser.add_argument("--kobert-epochs", type=int, default=5)
    parser.add_argument("--gnn-epochs", type=int, default=5)
    parser.add_argument("--gnn-min-holdout-accuracy", type=float, default=0.96)
    parser.add_argument("--skip-kobert-train", action="store_true", help="KoBERT 텍스트 10k 준비/검증만 수행합니다.")
    parser.add_argument("--skip-prepare", action="store_true", help="기존 dev/engine_training_inputs/*_10k.csv 파일을 재사용합니다.")
    parser.add_argument(
        "--reuse-kobert-text",
        action="store_true",
        help="기존 KoBERT 텍스트 CSV를 그대로 재사용합니다. 기본은 준비 단계에서 텍스트도 갱신합니다.",
    )
    args = parser.parse_args()

    t_total = time.time()
    xgb_csv = os.path.join(ENGINE_INPUTS, "xgboost_train_10k.csv")
    gnn_csv = os.path.join(BASE, "gnn", "gnn_total_dataset.csv")
    gnn_delta_csv = os.path.join(ENGINE_INPUTS, "gnn_collect_to_10k.csv")
    kobert_candidates = os.path.join(ENGINE_INPUTS, "kobert_candidates_10k.csv")
    kobert_text = os.path.join(ENGINE_INPUTS, "kobert_text_train_10k.csv")

    if not args.skip_prepare:
        prepare_cmd = [
            PYTHON, os.path.join(DEV, "prepare_engine_training_inputs.py"),
            "--target", str(args.target),
        ]
        expanded_source = os.path.join(ENGINE_INPUTS, "expanded_url_train_all.csv")
        if os.path.isfile(expanded_source):
            prepare_cmd.extend(["--extra", expanded_source])
        run(prepare_cmd, f"엔진별 {args.target}개 학습 입력 생성")

    gnn_delta_rows = count_labeled_rows(gnn_delta_csv)
    if gnn_delta_rows:
        gnn_collect_cmd = [
            PYTHON, os.path.join(BASE, "gnn", "collect_gnn_dataset.py"),
            "--input", gnn_delta_csv,
            "--workers", str(args.workers),
        ]
        for exclude_path in (
            os.path.join(DEV, "dataset_splits", "validation.csv"),
            os.path.join(DEV, "dataset_splits", "test_balanced.csv"),
            os.path.join(DEV, "dataset_splits", "test_operational.csv"),
        ):
            gnn_collect_cmd.extend(["--exclude-csv", exclude_path])
        run(gnn_collect_cmd, f"GNN 데이터셋 10k 보강 ({gnn_delta_rows}개 후보)")
    else:
        print("\n[SKIP] GNN 수집 후보가 없어 기존 gnn_total_dataset.csv를 재사용합니다.")

    kobert_text_cmd = [
        PYTHON, os.path.join(DEV, "collect_kobert_text_dataset.py"),
        "--input", kobert_candidates,
        "--out", kobert_text,
        "--workers", str(args.workers),
        "--fallback-url-text",
    ]
    if not args.skip_prepare and not args.reuse_kobert_text:
        kobert_text_cmd.append("--refresh-existing")
    run(kobert_text_cmd, "KoBERT 텍스트 10k 준비")

    # ──────────────────────────────────────────
    # Phase 3: XGBoost 학습
    # ──────────────────────────────────────────
    xg_cmd = [
        PYTHON, os.path.join(BASE, "xgboost", "XG_train.py"),
        "--input", xgb_csv,
        "--min-training-rows", str(args.target),
        "--min-test-accuracy", "0.99",
    ]
    if args.skip_domain:
        xg_cmd.append("--skip-domain")
    if args.skip_dom and not args.with_dom:
        xg_cmd.append("--skip-dom")
    run(xg_cmd, "XGBoost 3-bundle 재학습")

    # ──────────────────────────────────────────
    # Phase 4: GNN 데이터 수집 + 학습
    # ──────────────────────────────────────────
    run([
        PYTHON, os.path.join(BASE, "gnn", "regenerate_gnn_model.py"),
        "--csv", gnn_csv,
        "--min-training-rows", str(args.target),
        "--min-holdout-accuracy", str(args.gnn_min_holdout_accuracy),
        "--epochs", str(args.gnn_epochs),
    ], "GNN 모델 재학습")

    # ──────────────────────────────────────────
    # Phase 5: KoBERT 학습
    # ──────────────────────────────────────────
    kobert_cmd = [
        PYTHON, os.path.join(BASE, "KoBERT", "kobert_train.py"),
        "--input", kobert_text,
        "--epochs", str(args.kobert_epochs),
        "--workers", str(args.workers),
        "--skip-fetch",
        "--min-training-rows", str(args.target),
    ]
    if args.skip_kobert_train:
        kobert_cmd.append("--prepare-only")
    run(kobert_cmd, f"KoBERT Fine-tuning (epochs={args.kobert_epochs})")

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
