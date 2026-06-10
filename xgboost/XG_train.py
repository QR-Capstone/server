#!/usr/bin/env python3
"""
XGBoost 3-bundle 학습 스크립트
  - url_xgb_paired_first.joblib   (타이포스쿼팅 모델)
  - url_xgb_domain_age.joblib     (도메인 나이/평판 모델)
  - url_xgb_dom.joblib            (DOM 구조 모델)

사용법:
  python XG_train.py --input urls.csv [--out-dir .] [--skip-dom] [--dom-workers 4]

입력 CSV 컬럼: url, label  (label: 0=정상, 1=악성)
기존 모델 파일은 .bak으로 백업 후 덮어씁니다.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from XG_core import (
    DOM_MODEL_FEATURE_NAMES,
    FEATURE_NAMES,
    ModelBundle,
    _group_train_val_test_split,
    evaluate_binary_classifier,
    extract_features,
    extract_dom_feature_array,
    save_bundle,
    train_xgboost_classifier,
)


# ──────────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────────

def load_csv(path: str) -> Tuple[List[str], np.ndarray]:
    urls: List[str] = []
    labels: List[int] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u = (row.get("url") or "").strip()
            lb = str(row.get("label", "")).strip()
            if not u or lb not in ("0", "1"):
                continue
            urls.append(u)
            labels.append(int(lb))
    return urls, np.array(labels, dtype=np.int32)


# ──────────────────────────────────────────────
# DOM 특징 병렬 수집
# ──────────────────────────────────────────────

def collect_dom_features(urls: List[str], workers: int = 4, offline: bool = False) -> np.ndarray:
    feat_dim = len(DOM_MODEL_FEATURE_NAMES)
    if offline:
        print(f"  DOM 특징 수집: offline fallback features for {len(urls)} URL(s)")
        return np.zeros((len(urls), feat_dim), dtype=np.float32)
    results: Dict[int, np.ndarray] = {}
    print(f"  DOM 특징 수집: {len(urls)}개 URL, workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(extract_dom_feature_array, url): i for i, url in enumerate(urls)}
        done = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            done += 1
            try:
                arr = future.result()
                if arr.shape[0] < feat_dim:
                    arr = np.pad(arr, (0, feat_dim - arr.shape[0]))
                results[idx] = arr[:feat_dim]
            except Exception as e:
                results[idx] = np.zeros(feat_dim, dtype=np.float32)
            if done % 20 == 0 or done == len(urls):
                print(f"    [{done}/{len(urls)}]", flush=True)
    return np.vstack([results[i] for i in range(len(urls))])


def collect_xg_features(
    urls: List[str],
    *,
    enable_domain_age: bool,
    enable_ssl: bool,
    domain_only: bool,
    workers: int = 1,
) -> np.ndarray:
    if workers <= 1:
        return np.vstack(
            [
                extract_features(
                    url,
                    enable_domain_age=enable_domain_age,
                    enable_ssl=enable_ssl,
                    domain_only=domain_only,
                )
                for url in urls
            ]
        )

    print(f"  XG 특징 추출: {len(urls)}개 URL, workers={workers}")
    results: Dict[int, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                extract_features,
                url,
                enable_domain_age=enable_domain_age,
                enable_ssl=enable_ssl,
                domain_only=domain_only,
            ): i
            for i, url in enumerate(urls)
        }
        done = 0
        for future in as_completed(future_map):
            idx = future_map[future]
            done += 1
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"    feature_error idx={idx} url={urls[idx]} error={e}", flush=True)
                results[idx] = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
            if done % 250 == 0 or done == len(urls):
                print(f"    [{done}/{len(urls)}]", flush=True)
    return np.vstack([results[i] for i in range(len(urls))])


# ──────────────────────────────────────────────
# 그룹(도메인) 배열 생성
# ──────────────────────────────────────────────

def make_groups(urls: List[str]) -> np.ndarray:
    import tldextract
    groups = []
    for u in urls:
        ext = tldextract.extract(u)
        groups.append(f"{ext.domain}.{ext.suffix}")
    unique = list(dict.fromkeys(groups))
    idx_map = {g: i for i, g in enumerate(unique)}
    return np.array([idx_map[g] for g in groups], dtype=np.int32)


# ──────────────────────────────────────────────
# 백업
# ──────────────────────────────────────────────

def backup(path: str) -> None:
    if os.path.isfile(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"  백업: {bak}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ──────────────────────────────────────────────
# 공통 학습 루틴
# ──────────────────────────────────────────────

def train_and_save(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    meta: dict,
    out_path: str,
    model_name: str,
    min_test_accuracy: float = 0.0,
) -> None:
    print(f"\n[{model_name}] 학습 시작 — 샘플 {len(y)}개 (악성 {y.sum()} / 정상 {(y==0).sum()})")

    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _ = _group_train_val_test_split(
        X, y, groups, test_size=0.25
    )
    print(f"  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

    t0 = time.perf_counter()
    clf = train_xgboost_classifier(X_train, y_train, X_val, y_val)
    elapsed = time.perf_counter() - t0
    print(f"  학습 완료 ({elapsed:.1f}s)")

    metrics = evaluate_binary_classifier(clf, X_test, y_test, optimize_threshold=True)
    print(f"  [테스트 성능]")
    print(f"    Accuracy : {metrics['accuracy']:.4f}")
    print(f"    ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"    F1       : {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall   : {metrics['recall']:.4f}")
    print(f"    Threshold: {metrics['threshold']:.3f}")
    cm = metrics["confusion_matrix"]
    if cm:
        print(f"    Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    if min_test_accuracy and metrics["accuracy"] < min_test_accuracy:
        raise RuntimeError(
            f"{model_name} test accuracy {metrics['accuracy']:.4f} < min_test_accuracy {min_test_accuracy:.4f}; "
            "기존 모델을 덮어쓰지 않습니다."
        )

    backup(out_path)
    enriched_meta = {
        **meta,
        "training_rows": int(len(y)),
        "label_counts": {"0": int((y == 0).sum()), "1": int(y.sum())},
        "split_rows": {
            "train": int(len(y_train)),
            "validation": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "internal_test_metrics": _jsonable(metrics),
        "min_test_accuracy": float(min_test_accuracy),
        "training_elapsed_seconds": round(float(elapsed), 3),
    }
    bundle = ModelBundle(model=clf, feature_names=feature_names, meta=enriched_meta)
    save_bundle(bundle, out_path)
    print(f"  저장: {out_path}")


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="XGBoost 3-bundle 재학습")
    parser.add_argument("--input", required=True, help="학습 CSV (url, label 컬럼)")
    parser.add_argument("--out-dir", default=BASE_DIR, help="모델 저장 디렉토리")
    parser.add_argument("--skip-typo", action="store_true", help="타이포스쿼팅 URL 모델 학습 건너뜀")
    parser.add_argument("--skip-domain", action="store_true", help="도메인 나이/SSL 모델 학습 건너뜀")
    parser.add_argument("--skip-dom", action="store_true", help="DOM 모델 학습 건너뜀 (느린 크롤링 생략)")
    parser.add_argument(
        "--offline-domain-signals",
        action="store_true",
        help="도메인 나이/SSL 네트워크 조회 없이 동일 feature schema의 fallback 신호로 domain bundle을 학습합니다.",
    )
    parser.add_argument(
        "--offline-dom-signals",
        action="store_true",
        help="DOM 크롤링 없이 동일 feature schema의 zero fallback 신호로 DOM bundle을 학습합니다.",
    )
    parser.add_argument("--dom-workers", type=int, default=4, help="DOM 크롤링 병렬 수")
    parser.add_argument("--feature-workers", type=int, default=1, help="URL/domain 특징 추출 병렬 수")
    parser.add_argument("--min-training-rows", type=int, default=0, help="최소 학습 입력 row 수")
    parser.add_argument(
        "--min-test-accuracy",
        type=float,
        default=0.0,
        help="각 XGBoost bundle의 내부 테스트 정확도가 이 값보다 낮으면 저장하지 않습니다.",
    )
    args = parser.parse_args(argv)

    print(f"=== XGBoost 재학습 시작 ===")
    print(f"입력: {args.input}")

    urls, y = load_csv(args.input)
    if len(urls) == 0:
        print("오류: CSV에서 URL을 읽을 수 없습니다.")
        return 1
    if len(urls) < args.min_training_rows:
        print(f"오류: 학습 입력 {len(urls)}개 < 최소 {args.min_training_rows}개")
        return 1
    print(f"로드: {len(urls)}개 (악성 {int(y.sum())} / 정상 {int((y==0).sum())})")

    os.makedirs(args.out_dir, exist_ok=True)
    groups = make_groups(urls)

    # ── 1. 타이포스쿼팅 모델 ──
    typo_path = os.path.join(args.out_dir, "url_xgb_paired_first.joblib")
    if args.skip_typo:
        print("\n[타이포스쿼팅 모델] --skip-typo 옵션으로 건너뜀")
    else:
        print("\n특징 추출 중 (타이포/도메인 — URL만, 빠름)...")
        X_typo = collect_xg_features(
            urls,
            enable_domain_age=False,
            enable_ssl=False,
            domain_only=False,
            workers=args.feature_workers,
        )
        train_and_save(
            X_typo, y, groups,
            feature_names=FEATURE_NAMES,
            meta={"enable_domain_age": False, "enable_ssl": False, "domain_only": False},
            out_path=typo_path,
            model_name="타이포스쿼팅",
            min_test_accuracy=args.min_test_accuracy,
        )

    # ── 2. 도메인 나이/평판 모델 ──
    domain_path = os.path.join(args.out_dir, "url_xgb_domain_age.joblib")
    if args.skip_domain:
        print("\n[도메인 나이/평판 모델] --skip-domain 옵션으로 건너뜀")
    else:
        if args.offline_domain_signals:
            print("\n특징 추출 중 (도메인 나이 — offline fallback, RDAP/SSL 조회 없음)...")
        else:
            print("\n특징 추출 중 (도메인 나이 — RDAP/SSL 조회 포함, 다소 느림)...")
        X_domain = collect_xg_features(
            urls,
            enable_domain_age=not args.offline_domain_signals,
            enable_ssl=not args.offline_domain_signals,
            domain_only=True,
            workers=args.feature_workers,
        )
        train_and_save(
            X_domain, y, groups,
            feature_names=FEATURE_NAMES,
            meta={
                "enable_domain_age": not args.offline_domain_signals,
                "enable_ssl": not args.offline_domain_signals,
                "domain_only": True,
                "offline_domain_signals": bool(args.offline_domain_signals),
            },
            out_path=domain_path,
            model_name="도메인 나이/평판",
            min_test_accuracy=args.min_test_accuracy,
        )

    # ── 3. DOM 구조 모델 ──
    dom_path = os.path.join(args.out_dir, "url_xgb_dom.joblib")
    if args.skip_dom:
        print("\n[DOM 모델] --skip-dom 옵션으로 건너뜀")
    else:
        if args.offline_dom_signals:
            print("\nDOM 특징 수집 중 (offline zero fallback, 크롤링 없음)...")
        else:
            print("\nDOM 특징 수집 중 (실제 페이지 크롤링 — 느림)...")
        X_dom = collect_dom_features(urls, workers=args.dom_workers, offline=args.offline_dom_signals)
        train_and_save(
            X_dom, y, groups,
            feature_names=list(DOM_MODEL_FEATURE_NAMES),
            meta={"dom_model": True, "offline_dom_signals": bool(args.offline_dom_signals)},
            out_path=dom_path,
            model_name="DOM 구조",
            min_test_accuracy=args.min_test_accuracy,
        )

    print("\n=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
