"""
GNN(서비스 명칭): gnn-ready.py로 학습한 URL 어휘 특징 + RandomForest(opqr_model.pkl) 추론.
PyG 그래프 신경망 체크포인트가 아니라 동일 스크립트의 RF 모델을 로드한다.
Python 3.10+ 권장(3.9에서 pickle 로드 실패할 수 있음).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

# gnn-ready.py 의 dict 기반 특징 (순서 고정 — 학습 시 DataFrame 컬럼 순서와 일치해야 함)
FEATURE_ORDER: List[str] = [
    "url_len",
    "dot_count",
    "hyphen_count",
    "slash_count",
    "is_https",
    "digit_ratio",
    "keyword_match",
]

SUSPICIOUS_WORDS = ("login", "verify", "bank", "update", "free", "account")


def extract_lexical_features(url: str) -> Dict[str, float]:
    """단일 URL에서 gnn-ready.py 와 동일한 특징 dict."""
    u = (url or "").strip()
    features: Dict[str, float] = {}
    features["url_len"] = float(len(u))
    features["dot_count"] = float(u.count("."))
    features["hyphen_count"] = float(u.count("-"))
    features["slash_count"] = float(u.count("/"))
    features["is_https"] = 1.0 if u.startswith("https") else 0.0
    digits = re.findall(r"\d", u)
    features["digit_ratio"] = float(len(digits) / len(u)) if len(u) > 0 else 0.0
    ul = u.lower()
    features["keyword_match"] = (
        1.0 if any(word in ul for word in SUSPICIOUS_WORDS) else 0.0
    )
    return features


def _feature_vector_row(url: str, column_order: List[str]) -> pd.DataFrame:
    feat = extract_lexical_features(url)
    row = {k: feat.get(k, 0.0) for k in column_order}
    return pd.DataFrame([row])


def load_opqr_model(
    model_path: str,
    feature_columns_path: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    """모델과 예측에 사용할 컬럼 순서를 반환."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(
            "opqr_model.pkl 로드 실패. 파일이 손상됐거나, "
            "더 새 Python(예: 3.11+)으로 저장된 pickle이라 구버전 Python에서 열 수 없을 수 있습니다. "
            "서버를 Python 3.11+로 맞추거나, 학습 환경에서 "
            "`joblib.dump(model, 'opqr_model.pkl', compress=3)`(또는 protocol=4)로 다시 저장해 보세요."
        ) from e
    cols: Optional[List[str]] = None
    if feature_columns_path and os.path.isfile(feature_columns_path):
        loaded = joblib.load(feature_columns_path)
        if isinstance(loaded, list):
            cols = [str(c) for c in loaded]
    if cols is None:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
        else:
            cols = list(FEATURE_ORDER)
    return model, cols


def predict_opqr(
    model: Any,
    column_order: List[str],
    raw_url: str,
) -> Dict[str, Any]:
    """피싱 확률·라벨·판정. 실패 시 예외를 호출자에게 전달."""
    u = (raw_url or "").strip()
    if not u:
        raise ValueError("empty url")

    if not u.startswith(("http://", "https://")):
        u = "https://" + u

    X = _feature_vector_row(u, column_order)
    prob_mal = float(model.predict_proba(X)[0][1])
    label = int(model.predict(X)[0])
    return {
        "url": u,
        "probability": round(prob_mal, 6),
        "label": label,
        "verdict": "malicious" if label == 1 else "benign",
    }
