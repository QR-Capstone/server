# Phishing Detection Server

피싱 URL을 탐지하는 멀티 모델 앙상블 기반 분석 서버입니다.  
URL ML을 먼저 실행해 명확한 정상 URL은 빠르게 종료하고, 추가 검증이 필요한 URL은
KoBERT, XGBoost, GNN을 병렬로 실행해 최종 판정을 도출합니다.

---

## 목차

- [아키텍처 개요](#아키텍처-개요)
- [엔진 상세](#엔진-상세)
  - [URL ML 엔진](#url-ml-엔진)
  - [KoBERT 엔진](#kobert-엔진)
  - [XGBoost 엔진](#xgboost-엔진)
  - [GNN 엔진](#gnn-엔진)
- [앙상블 판정 로직](#앙상블-판정-로직)
- [API 명세](#api-명세)
- [설치 및 실행](#설치-및-실행)
- [환경 변수](#환경-변수)
- [프로젝트 구조](#프로젝트-구조)

---

## 아키텍처 개요

```
POST /analyze
      │
      ├── URL ML + URL rule/heuristic
      │        │
      │        ├── decisive SAFE/DANGEROUS → fast path 응답
      │        │
      │        └── 추가 검증 필요
      │
      ├──(asyncio.gather)───────────────────────────────────┐
      │                                                      │
  KoBERT Engine           XGBoost Engine              GNN Engine
  (Playwright +           (3 Specialized              (GraphSAGE
   KoBERT Transformer)     XGB Classifiers)            Torch Model)
      │                          │                          │
      └──────────────────────────┴──────────────────────────┘
                                 │
                       점수 기반 앙상블 판정
                                 │
                          최종 판정 응답 반환
```

URL ML이 명확한 정상 URL로 판단하면 KoBERT/XGBoost/GNN을 건너뛰어 낮은 지연시간으로 응답합니다.  
fast path가 아니면 세 엔진은 `asyncio.gather()`로 **동시에** 실행됩니다.
각 엔진은 서로 독립적이며, 하나가 실패해도 나머지 결과와 URL 휴리스틱으로 판정을 계속합니다.

---

## 엔진 상세

### URL ML 엔진

**파일**: `url_ml/url_ml_engine.py`, `trusted_domains.py`  
**모델**: `url_ml/url_ml_model.joblib`

URL 문자열 특징과 신뢰 도메인/강한 피싱 휴리스틱을 결합해 빠른 1차 판정을 수행합니다.
`/analyze`에서는 가장 먼저 실행되며, 명확한 정상 URL은 fast path로 종료합니다.

#### 출력 필드

| 필드 | 설명 |
|------|------|
| `verdict` | `"benign"` / `"malicious"` / `"unknown"` |
| `riskLevel` | `"SAFE"` / `"DANGEROUS"` / `"UNKNOWN"` |
| `probability` | 최종 악성 확률 (0.0~1.0) |
| `ml_probability` | URL ML 모델 악성 확률 |
| `heuristic_probability` | URL 휴리스틱 악성 점수 |

---

### KoBERT 엔진

**파일**: `KoBERT/koBERT.py`  
**모델**: `KoBERT/kobert_phishing_model_weights.pt` (monologg/kobert 기반 파인튜닝)

실제 브라우저로 페이지를 렌더링한 뒤, 한국어 BERT 모델로 텍스트를 분류합니다.

#### 동작 흐름

1. **Playwright** (iPhone 14 UA 에뮬레이션)로 JavaScript 렌더링
2. **curl-cffi** (Chrome 116 fingerprint)로 봇 탐지 우회 병행
3. BeautifulSoup으로 HTML 파싱 → 제목, 버튼, 입력 폼, 텍스트 추출
4. 로그인/결제 딥링크 최대 2단계까지 재귀 추적
5. 팝업·얼럿 메시지 수집
6. PII 자동 난독화 (전화번호, 이메일, 긴 숫자열)
7. KoBERT 토크나이저로 인코딩 → 피싱 분류 모델 추론

#### 출력 필드

| 필드 | 설명 |
|------|------|
| `judgment` | `"normal"` / `"unnormal"` |
| `riskLevel` | `"SAFE"` / `"DANGEROUS"` / `"UNKNOWN"` |
| `threatType` | 탐지된 위협 유형 (금융사기, 개인정보탈취 등) |
| `evidence` | 판단 근거 텍스트 목록 |

---

### XGBoost 엔진

**파일**: `xgboost/XG_core.py`, `xgboost/XG_infer.py`  
**모델**: `.joblib` 파일 3개 (앙상블)

URL을 정형 피처 벡터로 변환한 뒤 세 개의 전문화된 XGBoost 분류기를 앙상블합니다.

#### 3가지 분류기

| 분류기 | 모델 파일 | 역할 | 가중치 |
|--------|-----------|------|--------|
| 타이포스쿼팅 탐지 | `url_xgb_paired_first.joblib` | 도메인 문자 치환, 브랜드명 편집거리, 동형이의 문자 탐지 | 50% |
| 도메인 평판 | `url_xgb_domain_age.joblib` | RDAP 등록일, SSL 인증서 유효성, 도메인 나이 | 35% |
| DOM 구조 분석 | `url_xgb_dom.joblib` | HTML form·input·iframe·script 패턴 수 | 15% |

#### 판정 임계값

- 타이포스쿼팅 확률 ≥ 70%
- 도메인 평판 확률 ≥ 80%
- DOM 구조 확률 ≥ 80%
- 가중 합산 최종 확률 ≥ 55%

위 조건 중 하나라도 충족하면 해당 분류기 단독으로 악성 판정을 출력합니다.

#### 출력 필드

| 필드 | 설명 |
|------|------|
| `probability` | 최종 가중 합산 악성 확률 (0.0~1.0) |
| `label` | `"malicious"` / `"benign"` |
| `feature_details` | 피처별 기여 점수 상세 |

---

### GNN 엔진

**파일**: `gnn/gnn_engine.py`  
**모델**: `gnn/gnn_model.pkl` (Torch GraphSAGE), `gnn/gnn_model_features.pkl` (피처 메타데이터)

웹페이지의 구성 요소를 **그래프**로 모델링하여 GNN으로 피싱 여부를 분류합니다.

#### 그래프 구성

**노드 종류**

| 노드 타입 | 설명 |
|-----------|------|
| Domain | 페이지 도메인 |
| FormInput | 입력 폼 필드 |
| Button | 클릭 버튼 |
| Script | 외부 스크립트 소스 |
| Iframe | 삽입된 iframe |
| Image | 이미지 요소 |
| Brand | 감지된 브랜드 명칭 |

**노드 피처** (수치 벡터)

- URL 전체 길이, 호스트명 길이
- 의심 TLD 여부 (`.xyz`, `.top`, `.tk` 등)
- URL 내 숫자 비율
- 알려진 브랜드 키워드 포함 여부

**엣지**: 포함 관계 (도메인 → 폼, 도메인 → 스크립트 등)

#### 추론 과정

1. Playwright로 단일 페이지 렌더링 (링크 추적 없음)
2. HTML 파싱 → 노드/엣지 추출 → PyTorch 그래프 구성
3. GraphSAGE 2-hop 메시지 패싱
4. 루트 도메인 노드 임베딩으로 분류

#### 출력 필드

| 필드 | 설명 |
|------|------|
| `verdict` | `"malicious"` / `"benign"` |
| `probability` | 악성 확률 (0.0~1.0) |
| `explanation` | 탐지된 그래프 패턴 설명 |

---

## 앙상블 판정 로직

최종 판정은 URL ML, URL 휴리스틱, KoBERT, XGBoost, GNN의 세부 신호를 함께 사용합니다.
공식/신뢰 URL 규칙은 SAFE로 우선 보정하고, 강한 URL 피싱 규칙은 DANGEROUS로 우선 보정합니다.
그 외에는 URL ML의 강한 단독 신호 또는 다른 엔진의 지지 신호를 먼저 반영한 뒤,
사용 가능한 확률 평균으로 `SAFE` / `UNKNOWN` / `DANGEROUS`를 결정합니다.

```
공식/신뢰 URL 규칙                 → SAFE
강한 URL 피싱 규칙                 → DANGEROUS
URLML SAFE + URLHeuristic SAFE
  + XGBoost SAFE + GNN 비악성
  + KoBERT 단독 DANGEROUS          → UNKNOWN
URL ML DANGEROUS + 강한 확률/지지  → DANGEROUS
사용 가능한 확률 평균 >= 0.40      → DANGEROUS
사용 가능한 확률 평균 >= 0.30      → UNKNOWN
그 외                              → SAFE
```

`FINAL_DANGER_THRESHOLD`, `FINAL_UNKNOWN_THRESHOLD`,
`URL_ML_FINAL_SOLO_THRESHOLD`, `URL_ML_FINAL_SUPPORT_THRESHOLD`로 기준값을 조정할 수 있습니다.
URLML 단독 판정은 기본적으로 `URL_ML_THRESHOLD=0.60`, `URL_ML_UNKNOWN_THRESHOLD=0.20`을 사용합니다.
검증 CSV 기준 coverage/오판정 균형을 맞춘 값이며, test/holdout을 보고 재튜닝하지 않습니다.
GNN은 fetch가 성공했지만 그래프가 사실상 비어 있고 강한 URL 위험 신호도 없으면
`GNN_LOW_EVIDENCE_MAX_PROB` 이하로 확률을 낮춰 빈 그래프 오탐을 줄입니다.

---

## API 명세

### `POST /analyze`

URL ML fast path를 먼저 확인하고, 필요하면 세 엔진을 병렬 실행하여 최종 앙상블 결과를 반환합니다.

**요청**

```json
{
  "url": "https://example.com"
}
```

**응답**

```json
{
  "url": "https://example.com",
  "judgment": "normal",
  "riskLevel": "SAFE",
  "conclusion": "정상",
  "reasons": [
    "최종 판단 - 정상 사이트로 판단됩니다.",
    "KoBERT : 정상",
    "XGBoost : 정상 (확률: 0.12)",
    "GNN : 정상 (확률: 0.08)"
  ],
  "model_details": [...],
  "koBERT": {
    "judgment": "normal",
    "riskLevel": "SAFE",
    "threatType": null,
    "evidence": []
  },
  "xgboost": {
    "probability": 0.12,
    "label": "benign",
    "feature_details": {...}
  },
  "gnn": {
    "verdict": "benign",
    "probability": 0.08,
    "explanation": "..."
  },
  "timing": {
    "koBERT_sec": 3.21,
    "xgboost_sec": 1.45,
    "gnn_sec": 2.10,
    "total_wall_sec": 3.25
  }
}
```

---

### `POST /analyze/engine`

KoBERT 엔진만 단독 실행합니다.

**요청**: `{"url": "https://example.com"}`  
**응답**: KoBERT 결과 단독 반환

---

### `POST /analyze/xgboost`

XGBoost 엔진만 단독 실행합니다.

**요청**: `{"url": "https://example.com"}`  
**응답**: XGBoost 결과 단독 반환

---

### `POST /analyze/gnn`

GNN 엔진만 단독 실행합니다.

**요청**: `{"url": "https://example.com"}`  
**응답**: GNN 결과 단독 반환

---

### `POST /analyze/url-ml`

URL ML 엔진만 단독 실행합니다.

**요청**: `{"url": "https://example.com"}`  
**응답**: URL ML 결과 단독 반환

---

### `GET /health`

서버 생존 여부 확인 (liveness probe).

**응답**: `200 OK`

---

### `GET /ready`

서버 준비 상태 확인 (readiness probe). KoBERT 워밍업, 엔진 import 상태, 런타임 worker 설정, 캐시 통계,
동일 URL 요청 병합 상태를 함께 반환합니다.

**응답 예시**

```json
{
  "http_ok": true,
  "ready": true,
  "kobert_import_ok": true,
  "warmup_done": true,
  "runtime": {
    "xgboost_workers": 2,
    "gnn_workers": 2,
    "kobert_cache_size": 512,
    "xgboost_cache_size": 512,
    "gnn_cache_size": 512,
    "url_ml_cache_size": 1024,
    "gnn": {
      "fetch_cache_hits": 0,
      "fetch_cache_misses": 0,
      "fetch_cache_currsize": 0,
      "fetch_lock_count": 0
    },
    "xgboost": {
      "rdap_max_attempts": 1,
      "lock_max": 4096,
      "rdap_lock_count": 0,
      "ssl_lock_count": 0,
      "dom_lock_count": 0
    }
  },
  "cache": {
    "url_ml": {"hits": 0, "misses": 0, "currsize": 0},
    "kobert": {"hits": 0, "misses": 0, "currsize": 0},
    "xgboost": {"hits": 0, "misses": 0, "currsize": 0},
    "gnn": {"hits": 0, "misses": 0, "currsize": 0}
  },
  "inflight": {"total": 0},
  "issues": []
}
```

---

### `POST /warmup`

Playwright 브라우저를 수동으로 워밍업합니다. 서버 시작 직후 호출을 권장합니다.

**응답**: `{"status": "warmed up"}`

---

## 설치 및 실행

### 요구사항

- Python 3.10 이상
- pip

### 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# Playwright 브라우저 설치
playwright install chromium
```

### 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

서버 시작 후 워밍업을 먼저 실행하면 첫 요청 응답 시간이 단축됩니다.

```bash
curl -X POST http://localhost:8000/warmup
```

---

## 성능 검증

운영 절차와 고정 판정 기준은 [OPERATIONS.md](OPERATIONS.md)를 기준으로 관리합니다.

운영 전 로컬 회귀 게이트를 한 번에 실행하려면 다음 명령을 사용합니다.

```bash
python dev/run_operational_gates.py
```

이 통합 게이트는 URLML 재학습/누수/holdout 품질, API 성능/캐시/readiness, 엔진별 API holdout 평가를 순서대로 실행합니다.
실패하면 해당 단계에서 즉시 non-zero exit code로 중단되므로 CI의 필수 job으로 그대로 사용할 수 있습니다.

성능 회귀를 확인하려면 임시 uvicorn 서버를 띄우는 게이트를 실행합니다.

```bash
python dev/run_api_performance_gate.py
```

이 스크립트는 `/ready`가 준비될 때까지 기다린 뒤 다음 항목을 검증합니다.
각 검증은 `=== gate: ... ===` 이름으로 출력되어 실패한 시나리오를 바로 구분할 수 있습니다.

- 기본 `/analyze` 반복 요청의 위험도 분포와 server-side p95 지연시간
- 기본 `/analyze` 반복 요청의 URLML/rule/heuristic 캐시 miss/hit
- 공식 개발자 문서 도메인 fast path의 `SAFE` 판정과 p95 지연시간
- `/analyze/url-ml` 단독 fast path의 `SAFE` 판정과 p95 지연시간
- `/analyze/xgboost` 동일 URL 동시 요청의 캐시 miss/hit와 p95 지연시간
- `/analyze/gnn` 동일 URL 동시 요청의 캐시 miss/hit와 p95 지연시간
- GNN fetch 캐시가 fragment만 다른 URL을 중복 fetch하지 않는지 여부
- `/ready`의 worker, cache, lock 런타임 설정

실행 중 실패하면 임시 서버 stdout/stderr 마지막 로그를 출력합니다. p95 기준을 조정하려면 다음 옵션을 사용합니다.

```bash
python dev/run_api_performance_gate.py --fast-p95-ms 30 --same-domain-p95-ms 30
```

이미 실행 중인 서버를 대상으로 직접 측정하려면 벤치마크 도구를 사용합니다.

```bash
python dev/benchmark_api_latency.py --api http://127.0.0.1:8000 --endpoint /analyze --repeat 8 --workers 6 --keepalive
```

`--expect-risk SAFE=16`, `--max-server-p95-ms 20`, `--expect-ready runtime.xgboost_workers=2`처럼
위험도 분포, 응답 헤더 지연시간, `/ready` 필드 값을 게이트 조건으로 걸 수 있습니다.
`--expect-ready`가 실패하면 핵심 `/ready` 진단 스냅샷을 출력하며, 성공 시에도 보고 싶으면 `--print-ready`를 추가합니다.

벤치마크 도구의 파서와 검증 로직만 빠르게 확인하려면 서버 없이 self-test를 실행합니다.

```bash
python dev/benchmark_api_latency.py --self-test
```

## Holdout 평가

실제 holdout CSV 기준으로 엔진별 탐지 품질을 검증하려면 서버를 먼저 띄운 뒤 API 평가 게이트를 실행합니다.
CSV는 `url,label` 컬럼을 포함해야 하며 `label=0`은 정상, `label=1`은 악성입니다.

### 데이터셋 분리 원칙

학습 창고는 불균형이어도 되지만, URLML 학습 배치는 기본적으로 1:1 balanced sampling을 사용합니다.
canonical URL 기준으로 중복을 제거하고, 같은 URL이 정상/악성 양쪽에 있는 label conflict는 분리 파일로 남긴 뒤 학습/평가에서 제외합니다.
split은 `dev/prepare_url_dataset.py`의 hash 기반 고정 split으로 생성하며, test set은 임계값 튜닝에 사용하지 않습니다.

지원하는 `source` 값은 다음과 같습니다.

| source | 용도 |
|---|---|
| `malicious_real_live` | live 악성 URL |
| `malicious_real_dead` | 죽은 악성 URL. URLML/XGBoost URL lane 학습에는 사용 가능 |
| `malicious_synthetic_kr` | 한국 브랜드 사칭 synthetic 악성. test set에는 넣지 않음 |
| `benign_major_official` | 주요 공식 정상 |
| `benign_hard_korean_smb` | 한국 중소/병원/학원/법무/세무/펜션/쇼핑 hard benign |
| `benign_ad_landing` | 광고/랜딩/긴 query 정상 |
| `benign_user_confirmed_fp` | 사용자 확인 오탐 정상 |

데이터셋 생성 예시:

```bash
python dev/generate_synthetic_korean_phish.py \
  --out dev/synthetic_korean_phish_20260524.csv \
  --count 1200

python dev/collect_hard_benign.py \
  --out dev/hard_benign_live.csv \
  --target 1000 \
  --exclude dev/dataset_splits/warehouse.csv

python dev/prepare_url_dataset.py \
  --input dev/urlml_retrain_all_unique_20260522.csv \
  --input dev/hard_benign_live.csv \
  --input dev/synthetic_korean_phish_20260524.csv \
  --out-dir dev/dataset_splits
```

위 명령은 `warehouse.csv`, `train.csv`, `validation.csv`, `test_balanced.csv`,
`test_operational.csv`, `label_conflicts.csv`, `summary.json`을 생성합니다.
`test_balanced.csv`는 악성/정상 1:1 모델 비교용이고,
`test_operational.csv`는 정상 URL이 훨씬 많은 운영형 오탐률 확인용입니다.

URLML 재학습은 validation/test URL을 명시적으로 제외하고, hard benign과 사용자 확인 오탐 정상에는 더 높은 sample weight를 부여합니다.
synthetic 악성은 기본 weight가 낮고 malicious 학습 비율도 제한됩니다.

```bash
python url_ml/train_url_ml.py \
  --input dev/dataset_splits/train.csv \
  --holdout dev/dataset_splits/validation.csv \
  --holdout dev/dataset_splits/test_balanced.csv \
  --holdout dev/dataset_splits/test_operational.csv \
  --hard-benign-weight 8 \
  --user-confirmed-weight 8 \
  --naver-benign-weight 4 \
  --synthetic-weight 0.2 \
  --max-synthetic-malicious-ratio 0.20 \
  --no-class-weight-balanced \
  --C 4.0 \
  --out url_ml/url_ml_model.joblib
```

```bash
python dev/evaluate_api_holdout.py \
  --spawn-server \
  --train dev/urlml_retrain_all_unique_20260522.csv \
  --holdout dev/live_iter_20260523_new_clean_eval.csv \
  --require-ready \
  --max-unknown-ratio 0.0 \
  --engine-min-precision urlml=1.0 \
  --engine-min-recall urlml=1.0 \
  --engine-min-precision xgboost=1.0 \
  --engine-min-recall xgboost=0.95 \
  --engine-min-precision gnn=0.90 \
  --engine-min-recall gnn=1.0 \
  --engine-min-precision kobert=1.0 \
  --engine-min-recall kobert=1.0 \
  --engine-max-unknown-ratio kobert=0.40 \
  --engine-min-precision ensemble=1.0 \
  --engine-min-recall ensemble=1.0 \
  --engine-max-fp ensemble=0 \
  --engine-max-fn ensemble=0
```

이 게이트는 학습 CSV와 holdout CSV의 canonical URL 중복을 먼저 차단한 뒤
`/analyze/url-ml`, `/analyze/xgboost`, `/analyze/gnn`, `/analyze/engine`, `/analyze`를 같은 holdout으로 호출합니다.
각 엔진마다 accuracy, precision, recall, false positive count/rate, false negative count/rate, unknown count/rate, client p95 latency,
가능하면 `X-Process-Time` 기반 server p95 latency를 출력하고 기준 위반 시 실패합니다.
`--spawn-server`를 쓰면 임시 uvicorn 서버를 띄워 평가 후 자동 종료합니다.

엔진 하나만 볼 때는 `--engine urlml`, `--engine xgboost`, `--engine gnn`, `--engine kobert`,
`--engine ensemble`을 사용합니다. 빠른 smoke 평가에는 `--limit 20`을 추가할 수 있지만,
운영 기준 성능 보장은 전체 holdout으로 실행한 결과만 사용합니다.
`--min-accuracy 0.995`, `--max-fp-rate 0.01`, `--max-fn-rate 0.01`,
`--engine-min-precision xgboost=0.95`, `--engine-max-p95-ms ensemble=100`처럼 엔진별 기준을
명시할 수 있으며, 기준은 검증 세트로 고정하고 holdout 결과를 보고 완화하지 않습니다.
샘플 평가를 할 때는 `--per-label-limit 20`처럼 label별 동일 개수 제한을 걸어 앞쪽 행 편향을 피합니다.

---

## 환경 변수

`.env.example`을 복사하여 `.env`로 저장 후 값을 입력합니다.

```bash
cp .env.example .env
```

| 변수 | 설명 | 필수 |
|------|------|------|
| `PHISH_API_BASE` | 외부 피싱 DB API 베이스 URL | 선택 |
| `PHISH_API_PORT` | 외부 피싱 DB API 포트 | 선택 |
| `LOG_REQUEST_TIMING` | 요청별 처리시간 로그 출력 여부 (`0`/`1`) | 선택 |
| `ANALYZE_VERBOSE_LOGS` | `/analyze` 상세 로그 출력 여부 (`0`/`1`) | 선택 |
| `STARTUP_SMOKE_GNN` | 서버 시작 시 GNN smoke test 실행 여부 (`0`/`1`) | 선택 |
| `URL_ML_CACHE_SIZE` | URL ML 결과 LRU 캐시 크기 | 선택 |
| `KOBERT_CACHE_SIZE` | KoBERT 결과 LRU 캐시 크기 | 선택 |
| `XGBOOST_CACHE_SIZE` | XGBoost 결과 LRU 캐시 크기 | 선택 |
| `GNN_CACHE_SIZE` | GNN 결과 LRU 캐시 크기 | 선택 |
| `URL_RULE_CACHE_SIZE` | URL 규칙/도메인 판정 LRU 캐시 크기 | 선택 |
| `XGBOOST_WORKERS` | XGBoost 추론 thread pool worker 수 | 선택 |
| `GNN_WORKERS` | GNN 추론 thread pool worker 수 | 선택 |
| `XG_RDAP_LOOKUP_TIMEOUT` | XGBoost RDAP 조회 타임아웃(초, 기본 1.0) | 선택 |
| `XG_RDAP_MAX_ATTEMPTS` | XGBoost RDAP 조회 재시도 횟수 | 선택 |
| `XG_SSL_LOOKUP_TIMEOUT` | XGBoost SSL 조회 타임아웃(초, 기본 1.0) | 선택 |
| `XG_DOM_FETCH_TIMEOUT` | XGBoost DOM fetch 타임아웃(초, 기본 1.5) | 선택 |
| `XG_LOCK_MAX` | XGBoost RDAP/SSL/DOM per-key lock 최대 보관 수 | 선택 |
| `GNN_FETCH_CACHE_SIZE` | GNN HTML fetch LRU 캐시 크기 | 선택 |
| `GNN_FETCH_LOCK_MAX` | GNN fetch per-key lock 최대 보관 수 | 선택 |
| `SUPPRESS_KOBERT_DEBUG_LOGS` | KoBERT 추론 내부 debug stdout 억제 여부 (`0`/`1`) | 선택 |
| `SUPPRESS_XGBOOST_DEBUG_LOGS` | XGBoost 추론 내부 debug stdout 억제 여부 (`0`/`1`) | 선택 |

---

## 프로젝트 구조

```
.
├── main.py                            # FastAPI 서버 진입점, 앙상블 오케스트레이터
├── requirements.txt                   # Python 의존성 목록
├── .env.example                       # 환경 변수 템플릿
│
├── KoBERT/
│   ├── koBERT.py                      # Playwright 크롤링 + KoBERT 추론 로직
│   └── kobert_phishing_model_weights.pt  # 파인튜닝된 KoBERT 가중치
│
├── xgboost/
│   ├── XG_core.py                     # 피처 추출 + 앙상블 추론 (3947줄)
│   ├── XG_infer.py                    # CLI 추론 인터페이스
│   ├── XG_router.py                   # 학습/추론 라우터
│   ├── url_xgb_paired_first.joblib    # 타이포스쿼팅 탐지 모델
│   ├── url_xgb_domain_age.joblib      # 도메인 평판 모델
│   └── url_xgb_dom.joblib             # DOM 구조 분석 모델
│
└── gnn/
    ├── gnn_engine.py                  # 그래프 구성 + GraphSAGE 추론 (1758줄)
    ├── gnn_model.pkl                  # 학습된 Torch GraphSAGE 모델
    ├── gnn_model_features.pkl         # 피처 컬럼 메타데이터
    ├── gnn_total_dataset.csv          # GNN 학습 데이터셋
    ├── collect_gnn_dataset.py         # 데이터셋 수집 스크립트
    ├── regenerate_gnn_model.py        # 모델 재학습 스크립트
    └── reexport_gnn_model.py          # 모델 재익스포트 스크립트
```
