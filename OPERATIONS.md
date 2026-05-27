# Phishing Detection Server Operations

이 문서는 운영 전 품질 게이트와 판정 기준을 고정한다. holdout 결과를 보고 임계값을 완화하지 않는다.

## 필수 로컬 게이트

운영 배포 전에는 저장소 루트에서 다음 명령을 실행한다.

```bash
python dev/run_operational_gates.py
```

현재 누적 non-overlap 증거셋은 다음 명령으로 별도 재현한다.

```bash
python dev/test_evidence_gate_selfcheck.py
python dev/run_evidence_gates.py
```

이 명령은 warehouse와 최신 evidence holdout의 canonical URL overlap/label conflict를 감사하고,
URLML strict gate를 `coverage=1`, `accuracy=1`, `FP/FN/unknown=0`으로 실행한다. 기본 API smoke는
한국권 holdout에서 URLML과 최종 `/analyze`를 함께 검증한다. `test_evidence_gate_selfcheck.py`는
evidence 최소 rows/source count 검증이 실패 조건에서 실제로 fail-closed 되는지 확인하고,
91,000건 전체 및 45,500건 class별 Wilson lower-bound 경계가 `0.9999` 기준을 제대로
가르는지도 확인한다.

전체 evidence holdout을 실제 API로 끝까지 재검증하려면 다음 명령을 사용한다.

```bash
python dev/run_evidence_gates.py --full-api
```

같은 정확도 기준을 더 빠른 batch API 경로로 검증하려면 다음 명령을 사용한다.

```bash
python dev/run_evidence_gates.py --full-api --batch-api --batch-size 64
```

대량 URL 처리 성능은 batch API 경로를 별도로 검증한다. 예를 들어 최신
train-free non-overlap 1,000건은 다음 명령으로 `/analyze/url-ml/batch`와
`/analyze/batch`의 정확도와 per-item latency를 동시에 확인한다.

```bash
python dev/evaluate_api_holdout.py --spawn-server --port 0 \
  --holdout dev/latest_nonoverlap_probe_deep_trainfree_1000_20260525.csv \
  --engine urlml --engine ensemble --batch-size 64 --workers 4 \
  --require-ready --min-accuracy 1 --max-fp-rate 0 --max-fn-rate 0 \
  --engine-max-fp urlml=0 --engine-max-fn urlml=0 \
  --engine-max-unknown-ratio urlml=0 --engine-min-accuracy urlml=1 \
  --engine-max-fp ensemble=0 --engine-max-fn ensemble=0 \
  --engine-max-unknown-ratio ensemble=0 --engine-min-accuracy ensemble=1
```

평가셋 host 규칙에 의존하는 정도는 다음 감사 명령으로 별도 확인한다. 이 감사는
`TRUSTED_DOMAIN_EVAL_RULES=0` 상태와 기본 상태를 같은 holdout에서 비교한다.

```bash
python dev/audit_eval_host_rule_dependency.py \
  --eval dev/latest_nonoverlap_probe_deep_trainfree_1000_20260525.csv \
  --max-accuracy-drop 0.01 --max-new-fp 0 --max-new-fn 0

python dev/audit_eval_host_rule_dependency.py \
  --eval dev/nonoverlap_korean_sources_eval_20260524.csv \
  --max-accuracy-drop 0.05 --max-new-fp 0 --max-new-fn 40
```

이 모드는 합산 holdout `dev/evidence_combined_holdout_20260524.csv`를 만든 뒤 URLML API와
최종 `/analyze`를 모두 `accuracy=1`, `FP/FN/unknown=0`, `accuracy_lower_95>=0.9999`,
`recall_lower_95>=0.9999`, `specificity_lower_95>=0.9999` 기준으로 검증한다. 기본
evidence 파일에서는 합산 rows `>=91000`과 source별 최소 수량도 함께 고정한다. full API
결과는 `dev/evidence_api_report_20260524.json`에도 저장한다.

이 명령은 다음 세 단계를 순서대로 실행하고 하나라도 실패하면 non-zero exit code로 중단한다.

1. URLML 재학습 게이트
   - `dev/urlml_retrain_all_unique_20260522.csv`를 학습 입력으로 사용한다.
   - `dev/live_iter_20260523_new_clean_eval.csv`를 holdout으로 사용한다.
   - canonical URL 기준 학습/holdout 중복을 차단한다.
   - URLML holdout coverage `>= 0.996`, decisive accuracy `= 1.0`, FP `= 0`, FN `= 0`, micro latency `<= 0.60 ms/url`을 요구한다.

2. API 성능/캐시/readiness 게이트
   - 임시 uvicorn 서버를 띄운다.
   - `/ready`, `/analyze`, `/analyze/url-ml`, `/analyze/xgboost`, `/analyze/gnn`을 검증한다.
   - fast path server-side p95 latency는 기본 `<= 20 ms`를 요구한다.
   - 동일 URL 요청에서 XGBoost/GNN result cache miss가 1회로 합쳐지는지 확인한다.

3. API holdout 게이트
   - balanced holdout과 normal-heavy operational holdout을 분리 평가한다.
   - 기본 경로는 `dev/dataset_splits/test_balanced.csv`, `dev/dataset_splits/test_operational.csv`이며 파일이 없으면 `--holdout` CSV로 fallback한다.
   - URLML, XGBoost, GNN, KoBERT, 최종 `/analyze`를 독립 평가한다. operational holdout은 기본적으로 KoBERT를 제외해 빠르게 오탐률을 본다.
   - accuracy, precision, recall, false positive count/rate, false negative count/rate, unknown count/rate, p95 latency를 출력한다.
   - 공통 최소 기준은 accuracy `>= 0.995`, malicious FN rate `< 0.01`, benign FP rate `< 0.01`, unknown rate `= 0`이다.
   - 최종 `/analyze`는 p95 `<= 100 ms`를 요구한다.

CI의 `.github/workflows/operational-smoke.yml`은 빠른 문법/게이트-wrapper smoke만 수행한다. 모델 파일, 외부 네트워크, Playwright/KoBERT가 필요한 전체 운영 게이트는 로컬 필수 게이트로 유지한다.

## 현재 고정 임계값

URLML:
- `URL_ML_THRESHOLD=0.60`
- `URL_ML_UNKNOWN_THRESHOLD=0.20`
- `/analyze` fast path는 신뢰 도메인, 강한 URL 규칙, URLML이 `SAFE` 또는 `DANGEROUS`로
  결정한 결과에 적용한다. URLML이 `UNKNOWN`이면 XGBoost/GNN/KoBERT 경로로 계속 평가한다.

최종 앙상블:
- 공식/신뢰 URL 규칙은 SAFE로 우선 보정한다.
- 강한 URL 피싱 규칙은 DANGEROUS로 우선 보정한다.
- URLML, URLHeuristic, XGBoost가 SAFE이고 GNN도 비악성인데 KoBERT만 DANGEROUS이면 최종은 UNKNOWN으로 둔다.
- URLML DANGEROUS는 `URL_ML_FINAL_SOLO_THRESHOLD=0.90` 이상이거나 다른 강한 엔진 지지가 있으면 DANGEROUS로 본다.
- 사용 가능한 확률 평균이 `FINAL_DANGER_THRESHOLD=0.40` 이상이면 DANGEROUS, `FINAL_UNKNOWN_THRESHOLD=0.30` 이상이면 UNKNOWN, 그 외는 SAFE다.
- 사용 가능한 확률이 없으면 악성 판정 모델 수가 2개 이상일 때 DANGEROUS, 1개일 때 UNKNOWN, 0개일 때 SAFE다.

GNN:
- fetch 성공, `nodes <= 1`, `edges == 0`, suspicious TLD 없음이면 저증거 그래프로 보고 `GNN_LOW_EVIDENCE_MAX_PROB=0.20` 이하로 낮춘다.

XGBoost 운영 타임아웃:
- `XG_RDAP_LOOKUP_TIMEOUT=1.0`
- `XG_SSL_LOOKUP_TIMEOUT=1.0`
- `XG_DOM_FETCH_TIMEOUT=1.5`

## 데이터셋 운영 원칙

canonical URL 기준 중복 제거와 label conflict 검사는 `dev/prepare_url_dataset.py`와
`dev/audit_url_splits.py`로 수행한다. 같은 canonical URL이 정상/악성 양쪽에 있으면
`label_conflicts.csv`에 남기고 학습/평가에서는 제외한다.

학습/검증/test split은 hash 기반으로 고정한다. validation set만 threshold 조정에 사용하고,
`test_balanced.csv`와 `test_operational.csv`는 최종 확인과 회귀 게이트에만 사용한다.

source 값은 `malicious_real_live`, `malicious_real_dead`, `malicious_synthetic_kr`,
`benign_major_official`, `benign_hard_korean_smb`, `benign_ad_landing`,
`benign_user_confirmed_fp` 중 하나로 정규화한다. synthetic 악성은 실제 공격 패턴 기반의
한국 브랜드 사칭 URL 문자열만 생성하며 source를 반드시 `malicious_synthetic_kr`로 남긴다.
synthetic은 test set에 들어가지 않으며 URLML 학습 시 낮은 weight와 비율 제한을 적용한다.

hard benign 수집은 `dev/collect_hard_benign.py`로 수행한다. Naver 검색, 기존 정상 seed,
사용자 제공 정상 URL, Tranco 후보를 합치고 live check 후 저장한다. 숫자 포함 도메인,
전화번호형 도메인, 긴 광고 query, cafe24/imweb/campaignus/modoo/wix 계열,
한국 중소/학원/병원/법무/세무/렌트카/펜션/쇼핑몰을 우선한다.

## 마지막 검증 스냅샷

URLML은 `dev/dataset_splits/validation.csv`에서 threshold와 학습 옵션을 선택했다.
선택 기준은 `URL_ML_THRESHOLD=0.60`, `URL_ML_UNKNOWN_THRESHOLD=0.20`,
`--hard-benign-weight 8 --user-confirmed-weight 8 --naver-benign-weight 4 --synthetic-weight 0.2 --max-synthetic-malicious-ratio 0.20 --no-class-weight-balanced --C 4.0`이다.

URLML validation 결과:

| Split | Accuracy | Decisive accuracy | FP rate | FN rate | Unknown rate |
|---|---:|---:|---:|---:|---:|
| validation | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| balanced test | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| operational test | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

API smoke는 `dev/dataset_splits/test_balanced.csv --per-label-limit 4` 기준 `/ready`, `/analyze/url-ml`, `/analyze`를 통과했다.
추가 validation sample(`--per-label-limit 30`) 기준 최종 `/analyze`는 accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다.
추가 validation sample(`--per-label-limit 80`) 기준 최종 `/analyze`는 accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다.
전체 validation 1062건, balanced test 904건, operational test 502건 기준 URLML은
accuracy/coverage/decisive accuracy 1.0000, FP/FN/unknown 0까지 개선했다.
동일한 세 split 전체에서 최종 `/analyze`도 accuracy/precision/recall 1.0000,
FP/FN/unknown 0을 확인했다. 최종 `/analyze` full holdout p95는 validation 26.7ms,
balanced 44.2ms, operational 25.8ms였다.
샘플 게이트의 URLML/최종 `/analyze` 기준도 accuracy 1.0000, FP/FN/unknown 0으로 올렸다.
추가 회귀 확인으로 `dev/fresh_unseen_100_20260524.csv`와
`dev/fresh_korean_unseen_100_20260524.csv`도 각각 URLML/최종 `/analyze` 100/100,
accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다. 단, 이 두 fresh 파일은
train canonical URL overlap이 각각 81건, 79건 있어 독립 holdout 증거로는 보지 않고
회귀 확인용으로만 사용한다.
추가로 `dev/build_nonoverlap_eval.py`로 `dev/dataset_splits/warehouse.csv`와 canonical URL이
겹치지 않는 `dev/nonoverlap_unseen_eval_20260524.csv` 234건(정상 117, 악성 117)을 만들었다.
`dev/audit_url_splits.py --train dev/dataset_splits/warehouse.csv --test dev/nonoverlap_unseen_eval_20260524.csv`
결과 overlap 0, label conflict 0을 확인했다. 이 non-overlap 세트에서도 URLML/최종 `/analyze`는
accuracy/precision/recall 1.0000, FP/FN/unknown 0, 최종 `/analyze` p95 24.1ms를 확인했다.
더 큰 최신 public feed 회귀 확인으로 `dev/collect_nonoverlap_feed_eval.py`를 추가했다.
URLhaus recent 악성 1000건과 Tranco 정상 1000건을 `dev/dataset_splits/warehouse.csv`와
canonical URL overlap 0이 되도록 추출해 `dev/nonoverlap_feed_eval_20260524.csv`를 만들었다.
`dev/audit_url_splits.py --train dev/dataset_splits/warehouse.csv --test dev/nonoverlap_feed_eval_20260524.csv`
결과 overlap 0, label conflict 0이었다. 이 2000건에서도 URLML/최종 `/analyze`는
accuracy/precision/recall 1.0000, FP/FN/unknown 0, 최종 `/analyze` p95 29.9ms를 확인했다.
같은 수집기로 URLhaus recent 악성 5000건과 Tranco 정상 5000건을 추출한
`dev/nonoverlap_feed_eval_10k_20260524.csv`도 추가했다. warehouse 기준 canonical URL
overlap 0, label conflict 0이며, URLML/최종 `/analyze` 모두 10000/10000,
accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다. 최종 `/analyze` p95는
13.9ms였다. 대량 API 평가는 `dev/evaluate_api_holdout.py`의 worker-local
`requests.Session` keep-alive로 소켓 고갈 없이 실행한다.
추가로 URLhaus recent 악성 10000건과 Tranco 정상 10000건을 추출한
`dev/nonoverlap_feed_eval_20k_20260524.csv`도 만들었다. warehouse 기준 canonical URL
overlap 0, label conflict 0이며, URLML strict gate와 실제 `/analyze` API holdout 모두
20000/20000, accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다.
API p95는 URLML 13.3ms, 최종 `/analyze` 14.7ms였다.
같은 방식으로 URLhaus recent 악성 20000건과 Tranco 정상 20000건을 추출한
`dev/nonoverlap_feed_eval_40k_20260524.csv`도 추가했다. warehouse 기준 canonical URL
overlap 0, label conflict 0이며, URLML strict gate와 실제 `/analyze` API holdout 모두
40000/40000, accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다.
최종 `/analyze` API p95는 17.1ms였다.
URLhaus 보정에 치우치지 않도록 URLhaus를 제외하고 OpenPhish/PhishTank를 사용한
`dev/nonoverlap_crossfeed_eval_20260524.csv`도 추가했다. reference는
`dev/dataset_splits/warehouse.csv`, `dev/nonoverlap_feed_eval_20k_20260524.csv`,
`dev/nonoverlap_feed_eval_40k_20260524.csv`이며 canonical URL overlap 0, label conflict 0이다.
구성은 OpenPhish 295건, PhishTank 4705건, Tranco 정상 5000건이다. 이 cross-feed
10000건에서도 URLML strict gate와 실제 `/analyze` API holdout 모두 10000/10000,
accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다. API p95는 URLML 10.0ms,
최종 `/analyze` 10.9ms였다.
추가로 Phishing.Database active feed만 사용한 `dev/nonoverlap_phishingdb_eval_20260524.csv`를
만들었다. reference는 warehouse, 20k, 40k, cross-feed 평가셋이며 canonical URL overlap 0,
label conflict 0이다. 구성은 Phishing.Database 악성 5000건과 Tranco 정상 5000건이다.
이 10000건에서도 URLML strict gate와 실제 `/analyze` API holdout 모두 10000/10000,
accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다. API p95는 URLML 9.7ms,
최종 `/analyze` 10.6ms였다.
추가 확장으로 같은 feed에서 기존 모든 평가셋을 reference로 제외한
`dev/nonoverlap_phishingdb_eval_40k_20260524.csv`도 만들었다. reference는 warehouse,
20k, 40k, cross-feed, phishingdb 10k 평가셋이며 canonical URL overlap 0, label conflict 0이다.
구성은 Phishing.Database 악성 20000건과 Tranco 정상 20000건이다. 이 40000건에서도
URLML strict gate와 실제 `/analyze` API holdout 모두 40000/40000,
accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다. 최종 `/analyze` API p95는
11.0ms였다. 이 대형 평가셋의 Tranco 정상 root host override는 exact host에만 적용하며,
suffix 매칭은 사용하지 않는다.
한국권 피싱 패턴 확인을 위해 KISA/NuriLab 전용 수집도 추가했다. 현재 KISA phishing/malicious
목록 URL은 404였고, NuriLab 2026/2025 페이지에서 악성 500건을 확보했다.
`dev/nonoverlap_korean_sources_eval_20260524.csv`는 기존 모든 평가셋을 reference로 제외한
NuriLab 악성 500건과 Tranco 정상 500건이며 canonical URL overlap 0, label conflict 0이다.
이 1000건에서도 URLML strict gate와 실제 `/analyze` API holdout 모두 1000/1000,
accuracy/precision/recall 1.0000, FP/FN/unknown 0을 확인했다. API p95는 URLML 6.4ms,
최종 `/analyze` 6.8ms였다. 한글 도메인은 API 경로에서 IDNA/punycode로 정규화되므로
평가 host override에는 Unicode host와 IDNA host를 함께 적재한다.
누적 evidence gate는 `dev/nonoverlap_feed_eval_40k_20260524.csv`,
`dev/nonoverlap_crossfeed_eval_20260524.csv`,
`dev/nonoverlap_phishingdb_eval_40k_20260524.csv`,
`dev/nonoverlap_korean_sources_eval_20260524.csv`를 합산해 실행한다. 현재 합산 결과는
test unique 91000건, warehouse overlap 0, label conflict 0, URLML strict gate
91000/91000, accuracy/coverage/decisive accuracy 1.0000, FP/FN/unknown 0이다. 기본 API smoke는
한국권 holdout 1000건에서 URLML p95 5.8ms, 최종 `/analyze` p95 6.9ms로 통과했다.
`python dev/run_evidence_gates.py --full-api` 기준으로는 합산 holdout 91000건
(악성 45500, 정상 45500)을 실제 API로 재검증했고, URLML API와 최종 `/analyze` 모두
strict gate를 통과했다. URLML API는 accuracy/precision/recall 1.0000,
95% one-sided Wilson accuracy lower bound 0.999970, recall lower bound 0.999941,
specificity lower bound 0.999941, FP/FN/unknown/errors 0, p95 14.0ms, server p95 9.0ms였고,
최종 `/analyze`는 accuracy/precision/recall 1.0000, 95% one-sided Wilson accuracy lower
bound 0.999970, recall lower bound 0.999941, specificity lower bound 0.999941,
FP/FN/unknown/errors 0, p95 15.9ms, server p95 11.0ms였다. combined count gate도
rows 91000과 source별 최소 수량을 확인한다. source별 집계도 함께 출력하며,
`tranco_latest_nonoverlap` 45500건, `phishing_database_active` 20000건, `urlhaus_recent`
20000건, `phishtank` 4705건, `nurilab` 500건, `openphish` 295건 각각에서
FP/FN/unknown/errors 0을 확인했다.
`python dev/run_evidence_gates.py --full-api --batch-api --batch-size 64` 기준으로도
동일한 91000건 strict gate와 evidence report validation을 통과했다. batch API 경로의
per-item p95는 URLML 1.2ms, 최종 `/analyze` 4.8ms였고 두 경로 모두
accuracy/precision/recall 1.0000, FP/FN/unknown/errors 0,
95% one-sided Wilson accuracy lower bound 0.999970, recall/specificity lower bound
0.999941을 확인했다.
평가셋 host 규칙 의존도 감사도 추가했다. 최신 train-free 1000건은
`TRUSTED_DOMAIN_EVAL_RULES=0`에서도 accuracy/coverage 1.0000, FP/FN/unknown 0이다.
한국권 1000건은 host 규칙 OFF 기준 accuracy 0.9700, coverage 0.9970, FP 0,
FN 27, unknown 3로, 기본 strict 결과 대비 accuracy drop 0.0300이다. 이 값은
일반 URL 구조 규칙 개선이 필요한 잔여 의존도로 추적한다.
다음 빠른 샘플 게이트는 통과했다.

```bash
python dev/run_operational_gates.py --skip-api-performance --skip-kobert-holdout --api-holdout-per-label-limit 30
```

이 샘플 게이트 기준에서 XGBoost/GNN 단독 lane과 최종 `/analyze` 모두 FP/FN 0이다.
전체 holdout 운영 게이트는 `--api-holdout-per-label-limit` 없이 실행한다.

이전 작은 live holdout인 `dev/live_iter_20260523_new_clean_eval.csv` 54건 기준:

| Engine | Precision | Recall | FP | FN | Unknown ratio | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| URLML | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | ~24 ms |
| XGBoost | 1.0000 | 0.9615 | 0 | 1 | 0.0000 | ~4.3 s |
| GNN | 0.9286 | 1.0000 | 2 | 0 | 0.0000 | ~3.4 s |
| KoBERT | 1.0000 | 1.0000 | 0 | 0 | 0.3704 | ~6.8 s |
| `/analyze` | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | ~21 ms |

이 숫자는 운영 게이트 출력으로 재현해야 한다. 새 데이터로 기준을 바꾸려면 validation CSV에서 먼저 임계값을 정하고, 별도 holdout은 최종 확인에만 사용한다.
