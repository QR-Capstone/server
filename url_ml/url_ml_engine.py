"""Fast URL lexical ML lane.

This model is deliberately lightweight: it classifies the URL string itself with
character n-grams, then lets trusted-domain and strong URL rules override it.
It complements KoBERT/GNN/XGBoost when page fetching is slow or unavailable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("URL_ML_MODEL_PATH", os.path.join(BASE_DIR, "url_ml_model.joblib"))
PARENT_DIR = os.path.dirname(BASE_DIR)
if PARENT_DIR not in os.sys.path:
    os.sys.path.insert(0, PARENT_DIR)

from trusted_domains import (
    is_low_risk_hosted_platform_url,
    is_trusted_official_url,
    strong_url_phishing_score,
    url_heuristic_phishing_score,
)


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


def _is_low_confidence_root_benign(raw_url: str, final_prob: float, heuristic: float) -> bool:
    if final_prob >= 0.46 or heuristic > 0.33:
        return False
    candidate = raw_url if "://" in raw_url else f"//{raw_url}"
    try:
        parsed = urlsplit(candidate)
    except Exception:
        return False
    return (parsed.path or "") in {"", "/"} and not parsed.query


def predict_url_ml(model: Any, raw_url: str) -> dict[str, Any]:
    url = (raw_url or "").strip()
    if is_trusted_official_url(url) or is_low_risk_hosted_platform_url(url):
        return {
            "verdict": "benign",
            "riskLevel": "SAFE",
            "probability": 0.0,
            "adjusted_by_rule": True,
            "adjustment_reason": "공식/저위험 호스팅 플랫폼 URLML 통과",
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

    threshold = float(os.getenv("URL_ML_THRESHOLD", "0.47"))
    unknown_threshold = float(os.getenv("URL_ML_UNKNOWN_THRESHOLD", "0.20"))
    if _is_low_confidence_root_benign(url, max(proba, heuristic), heuristic):
        return {
            "verdict": "benign",
            "riskLevel": "SAFE",
            "probability": None,
            "raw_probability": round(max(proba, heuristic), 6),
            "ml_probability": round(proba, 6),
            "heuristic_probability": round(heuristic, 6),
            "threshold": threshold,
            "adjusted_by_rule": True,
            "adjustment_reason": "낮은 신뢰도의 루트 도메인 URLML 정상 처리",
        }
    return _url_ml_result_from_scores(proba, heuristic, threshold, unknown_threshold)


def _url_ml_result_from_scores(proba: float, heuristic: float, threshold: float, unknown_threshold: float) -> dict[str, Any]:
    final_prob = max(float(proba), float(heuristic))
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
        "ml_probability": round(float(proba), 6),
        "heuristic_probability": round(float(heuristic), 6),
        "threshold": threshold,
    }


def predict_url_ml_batch(model: Any, raw_urls: list[str]) -> list[dict[str, Any]]:
    """Vectorized URLML inference for API batch endpoints.

    Rule-only short circuits stay identical to predict_url_ml; unresolved URLs
    share one model.predict_proba call to avoid per-URL sklearn overhead.
    """
    threshold = float(os.getenv("URL_ML_THRESHOLD", "0.47"))
    unknown_threshold = float(os.getenv("URL_ML_UNKNOWN_THRESHOLD", "0.20"))
    results: list[dict[str, Any] | None] = [None] * len(raw_urls)
    pending_indices: list[int] = []
    pending_urls: list[str] = []
    pending_heuristics: list[float] = []

    for index, raw_url in enumerate(raw_urls):
        url = (raw_url or "").strip()
        if is_trusted_official_url(url) or is_low_risk_hosted_platform_url(url):
            results[index] = {
                "verdict": "benign",
                "riskLevel": "SAFE",
                "probability": 0.0,
                "adjusted_by_rule": True,
                "adjustment_reason": "공식/저위험 호스팅 플랫폼 URLML 통과",
            }
            continue

        strong = float(strong_url_phishing_score(url))
        if strong >= 0.66:
            results[index] = {
                "verdict": "malicious",
                "riskLevel": "DANGEROUS",
                "probability": round(strong, 6),
                "adjusted_by_rule": True,
                "adjustment_reason": "강한 URL 피싱 패턴 URLML 사전 감지",
            }
            continue

        pending_indices.append(index)
        pending_urls.append(url)
        pending_heuristics.append(float(url_heuristic_phishing_score(url)))

    if pending_urls:
        try:
            probabilities = model.predict_proba(pending_urls)[:, 1]
        except Exception as e:
            for index, heuristic in zip(pending_indices, pending_heuristics):
                results[index] = {
                    "verdict": "unknown",
                    "riskLevel": "UNKNOWN",
                    "probability": round(heuristic, 6) if heuristic else None,
                    "error": str(e),
                }
        else:
            for index, url, proba, heuristic in zip(pending_indices, pending_urls, probabilities, pending_heuristics):
                if _is_low_confidence_root_benign(url, max(float(proba), heuristic), heuristic):
                    results[index] = {
                        "verdict": "benign",
                        "riskLevel": "SAFE",
                        "probability": None,
                        "raw_probability": round(max(float(proba), heuristic), 6),
                        "ml_probability": round(float(proba), 6),
                        "heuristic_probability": round(float(heuristic), 6),
                        "threshold": threshold,
                        "adjusted_by_rule": True,
                        "adjustment_reason": "낮은 신뢰도의 루트 도메인 URLML 정상 처리",
                    }
                else:
                    results[index] = _url_ml_result_from_scores(float(proba), heuristic, threshold, unknown_threshold)

    return [result if result is not None else {"verdict": "unknown", "riskLevel": "UNKNOWN"} for result in results]
