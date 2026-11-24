"""
add_domain_features.py
======================
gnn_total_dataset.csv 의 모든 URL에 '도메인 나이' 피처를 붙입니다.
(페이지에 다시 접속하지 않고, 도메인 등록 정보(RDAP)만 조회)

추가되는 컬럼
  domain_young    : 등록 180일 미만이면 1
  domain_old      : 등록 2년 이상이면 1
  domain_age_days : 등록 후 일수 (분석용, 모르면 -1)

조회 결과는 domain_age_cache.csv 에 저장되므로 중간에 멈춰도
다시 실행하면 이어서 진행합니다.

[실행]
    python add_domain_features.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import gnn_engine  # noqa: E402
from gnn_engine import (  # noqa: E402
    DOMAIN_FEATURE_NAMES,
    FEATURE_NAMES,
    _registered_domain,
    domain_features_from_age,
)

DATASET_CSV = BASE_DIR / "gnn_total_dataset.csv"
CACHE_CSV = BASE_DIR / "domain_age_cache.csv"
WORKERS = int(os.getenv("WORKER_COUNT", "8"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def label_of(row):
    try:
        return int(float(row.get("label", "")))
    except Exception:
        return None


def load_cache() -> dict:
    cache = {}
    if CACHE_CSV.exists():
        with open(CACHE_CSV, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 2:
                    cache[row[0]] = None if row[1] == "" else float(row[1])
    return cache


def main() -> None:
    with open(DATASET_CSV, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader]
    print(f"데이터셋: {len(rows)}행")

    def dom_of(url: str) -> str:
        try:
            return _registered_domain(urlsplit(url.strip()).hostname or "")
        except Exception:
            return ""

    domains = sorted({d for d in (dom_of(r.get("url", "")) for r in rows) if d})
    cache = load_cache()
    todo = [d for d in domains if d not in cache]
    print(f"도메인 {len(domains)}개 (이미 조회됨 {len(domains) - len(todo)}개, 새로 조회 {len(todo)}개)")
    print("IANA RDAP 목록 불러오는 중...")
    print(f"  RDAP 지원 TLD: {len(gnn_engine._rdap_bootstrap())}개")

    start = time.time()
    done = 0
    step = max(1, len(todo) // 20)
    with open(CACHE_CSV, "a", newline="", encoding="utf-8") as cf, \
            ThreadPoolExecutor(max_workers=WORKERS) as ex:
        w = csv.writer(cf)
        futures = {ex.submit(gnn_engine.domain_age_days, d, 6.0): d for d in todo}
        for fut in as_completed(futures):
            d = futures[fut]
            try:
                age = fut.result()
            except Exception:
                age = None
            cache[d] = age
            w.writerow([d, "" if age is None else f"{age:.1f}"])
            done += 1
            if done % step == 0 or done == len(todo):
                cf.flush()
                el = time.time() - start
                eta = (len(todo) - done) / (done / el) / 60 if done else 0
                print(f"  조회 {done}/{len(todo)}  (ETA {eta:.1f}분)", flush=True)

    for col in DOMAIN_FEATURE_NAMES + ["domain_age_days"]:
        if col not in fieldnames:
            fieldnames.append(col)

    stats = {0: [0, 0, 0], 1: [0, 0, 0]}  # [전체, 나이 앎, 신규]
    for r in rows:
        age = cache.get(dom_of(r.get("url", "")))
        feats = domain_features_from_age(age)
        for k, v in feats.items():
            r[k] = repr(v)
        r["domain_age_days"] = "-1" if age is None else f"{age:.0f}"
        lab = label_of(r)
        if lab in stats:
            stats[lab][0] += 1
            if age is not None:
                stats[lab][1] += 1
                if feats["domain_young"] > 0:
                    stats[lab][2] += 1

    backup = BASE_DIR / f"gnn_total_dataset.bak_{STAMP}.csv"
    os.replace(DATASET_CSV, backup)
    with open(DATASET_CSV, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in fieldnames})

    def pct(a, b):
        return f"{a / max(1, b) * 100:.1f}%"

    lines = [
        "=" * 60,
        " 도메인 나이 피처 추가 결과",
        "=" * 60,
        f"  데이터셋 백업: {backup.name}",
        "",
        "                 피싱        정상",
        f"  나이 조회 성공  {pct(stats[1][1], stats[1][0]):>8}   {pct(stats[0][1], stats[0][0]):>8}",
        f"  신규(180일 미만) {pct(stats[1][2], stats[1][1]):>7}   {pct(stats[0][2], stats[0][1]):>8}   (조회 성공 중)",
        "",
        "  ※ '조회 성공' 비율이 피싱/정상 사이에 아주 크게 다르면 알려주세요.",
        "    (모델이 '정보 없으면 피싱'을 배울 위험 점검용)",
        f"\n  총 소요: {(time.time() - start) / 60:.1f}분",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (BASE_DIR / "domain_feature_report.txt").write_text(report, encoding="utf-8")
    print("\n다음: python clean_and_retrain.py  (도메인 피처 포함해서 재학습)")


if __name__ == "__main__":
    main()
