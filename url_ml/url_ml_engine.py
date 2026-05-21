"""Fast URL lexical ML lane.

This model is deliberately lightweight: it classifies the URL string itself with
character n-grams, then lets trusted-domain and strong URL rules override it.
It complements KoBERT/GNN/XGBoost when page fetching is slow or unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("URL_ML_MODEL_PATH", os.path.join(BASE_DIR, "url_ml_model.joblib"))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in os.sys.path:
    os.sys.path.insert(0, PARENT_DIR)

from trusted_domains import is_trusted_official_url, strong_url_phishing_score, url_heuristic_phishing_score


@dataclass
class URLMLStatus:
    enabled: bool
    reason: str = ""
    model_path: str = MODEL_PATH


def load_url_ml_model(path: str = MODEL_PATH) -> tuple[Any | None, URLMLStatus]:
    if joblib is None:
        return None, URLMLStatus(False, "joblib_not_available", path)
    if not os.path.isfile(path):
        return None, URLMLStatus(False, f"missing_model:{path}", path)
    try:
        return joblib.load(path), URLMLStatus(True, "", path)
    except Exception as e:
        return None, URLMLStatus(False, f"load_error:{e}", path)


def predict_url_ml(model: Any, raw_url: str) -> dict[str, Any]:
    url = (raw_url or "").strip()
    if is_trusted_official_url(url):
        return {
            "verdict": "benign",
            "riskLevel": "SAFE",
            "probability": 0.0,
            "adjusted_by_rule": True,
            "adjustment_reason": "공식/신뢰 도메인 URLML 통과",
        }

    strong = float(strong_url_phishing_score(url))
    if strong >= 0.66:
        return {
            "verdict": "malicious",
            "riskLevel": "DANGEROUS",
            "probability": round(strong, 6),
            "adjusted_by_rule": True,
            "adjustment_reason": "강한 URL 피싱 패턴 URLML 사전 감지",
        }

    heuristic = float(url_heuristic_phishing_score(url))
    try:
        proba = float(model.predict_proba([url])[0][1])
    except Exception as e:
        return {
            "verdict": "unknown",
            "riskLevel": "UNKNOWN",
            "probability": round(heuristic, 6) if heuristic else None,
            "error": str(e),
        }

    final_prob = max(proba, heuristic)
    threshold = float(os.getenv("URL_ML_THRESHOLD", "0.50"))
    unknown_threshold = float(os.getenv("URL_ML_UNKNOWN_THRESHOLD", "0.30"))
    if final_prob >= threshold:
        risk_level = "DANGEROUS"
        verdict = "malicious"
        ensemble_probability: float | None = final_prob
    elif final_prob >= unknown_threshold:
        risk_level = "UNKNOWN"
        verdict = "unknown"
        ensemble_probability = None
    else:
        risk_level = "SAFE"
        verdict = "benign"
        ensemble_probability = None

    return {
        "verdict": verdict,
        "riskLevel": risk_level,
        "probability": round(ensemble_probability, 6) if ensemble_probability is not None else None,
        "raw_probability": round(final_prob, 6),
        "ml_probability": round(proba, 6),
        "heuristic_probability": round(heuristic, 6),
        "threshold": threshold,
    }
