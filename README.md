# Phishing Detection Server

피싱 URL을 탐지하는 멀티 모델 앙상블 기반 분석 서버입니다.  
KoBERT, XGBoost, GNN 세 가지 독립 엔진을 병렬로 실행하여 최종 판정을 도출합니다.

---

## 목차

- [아키텍처 개요](#아키텍처-개요)
- [엔진 상세](#엔진-상세)
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
      ├──(asyncio.gather)──────────────────────────────────┐
      │                                                     │
  KoBERT Engine          XGBoost Engine              GNN Engine
  (Playwright +          (3 Specialized              (GraphSAGE
   KoBERT Transformer)    XGB Classifiers)            Torch Model)
      │                         │                          │
      └─────────────────────────┴──────────────────────────┘
                                │
                    앙상블 투표 (2/3 이상 악성 → 위험)
                                │
                         최종 판정 응답 반환
```

세 엔진은 `asyncio.gather()`로 **동시에** 실행됩니다.  
각 엔진은 서로 독립적이며, 하나가 실패해도 나머지 결과로 판정을 계속합니다.

---

## 엔진 상세

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

세 엔진의 결과를 단순 투표로 합산합니다.

```
악성 판정 엔진 수 >= 2  →  최종 판정: 악성 (dangerous)
악성 판정 엔진 수 <= 1  →  최종 판정: 정상 (safe)
```

모든 모델이 정상 / 알 수 없음으로 반환하면 최종 판정은 정상입니다.

---

## API 명세

### `POST /analyze`

세 엔진을 병렬 실행하여 최종 앙상블 결과를 반환합니다.

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

### `GET /health`

서버 생존 여부 확인 (liveness probe).

**응답**: `200 OK`

---

### `GET /ready`

서버 준비 상태 확인 (readiness probe). KoBERT 워밍업 완료 여부를 포함합니다.

**응답**: `{"status": "ready", "kobert_warmed_up": true}`

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

## 환경 변수

`.env.example`을 복사하여 `.env`로 저장 후 값을 입력합니다.

```bash
cp .env.example .env
```

| 변수 | 설명 | 필수 |
|------|------|------|
| `PHISH_API_BASE` | 외부 피싱 DB API 베이스 URL | 선택 |
| `PHISH_API_PORT` | 외부 피싱 DB API 포트 | 선택 |

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
