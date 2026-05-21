from __future__ import annotations

import asyncio
import functools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlsplit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIRS = {
    "gnn": os.path.join(BASE_DIR, "gnn"),
    "xgboost": os.path.join(BASE_DIR, "xgboost"),
    "KoBERT": os.path.join(BASE_DIR, "KoBERT"),
    "url_ml": os.path.join(BASE_DIR, "url_ml"),
}
for _model_dir in MODEL_DIRS.values():
    if _model_dir not in sys.path:
        sys.path.insert(0, _model_dir)

# Tuning (same env vars as engine module): MAX_SEQ_LEN, USE_PLAYWRIGHT_IN_ANALYZE, etc.
# os.environ.setdefault("MAX_SEQ_LEN", "128")
# os.environ.setdefault("USE_PLAYWRIGHT_IN_ANALYZE", "0")
# os.environ.setdefault("TORCH_NUM_THREADS", "4")

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from XG_core import (
    build_all_explanations,
    load_bundle,
    predict_url,
    predict_url_dom,
    xgboost_weighted_ensemble_verdict,
)
from gnn_engine import GNN_Engine, predict_gnn
from url_ml_engine import load_url_ml_model, predict_url_ml

try:
    from trusted_domains import (
        is_trusted_official_url,
        strong_url_phishing_score,
        url_heuristic_phishing_score,
    )
except Exception:  # pragma: no cover
    def is_trusted_official_url(raw_url: str) -> bool:
        return False

    def strong_url_phishing_score(raw_url: str) -> float:
        return 0.0

    def url_heuristic_phishing_score(raw_url: str) -> float:
        return 0.0

app = FastAPI(title="Phishing Detection API")


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Wall time for the full request (logged next to HTTP OK)."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dur = time.perf_counter() - t0
    response.headers["X-Process-Time"] = f"{dur:.3f}"
    print(
        f"--- [request done] {request.method} {request.url.path} "
        f"{response.status_code} OK ({dur:.3f}s)"
    )
    return response

# Warm up Playwright on startup (default off = faster boot)
STARTUP_WARMUP_PLAYWRIGHT = os.getenv("STARTUP_WARMUP_PLAYWRIGHT", "0") == "1"

# Playwright sync API is bound to one thread; engine runs on a single worker.
_engine_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phish_engine")
# XGBoost on its own thread pool (runs in parallel with KoBERT).
_xgboost_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xgboost_infer")
# Web-structure GNN (gnn_model.pkl)
_gnn_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gnn_webgraph")


async def _run_engine(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_engine_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_engine_executor, functools.partial(fn, *args))


async def _run_xgboost(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_xgboost_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_xgboost_executor, functools.partial(fn, *args))


async def _run_gnn(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_gnn_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_gnn_executor, functools.partial(fn, *args))


async def _run_url_ml(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_xgboost_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_xgboost_executor, functools.partial(fn, *args))


def _normalize_url_for_xgboost(url: str) -> str:
    """Normalize URL for XGBoost; prepend https if no scheme."""
    candidate = (url or "").strip()
    if not candidate:
        return candidate
    parsed = urlsplit(candidate)
    if parsed.scheme and parsed.netloc:
        return candidate
    return f"https://{candidate}"


def _run_xgboost_inference(raw_url: str):
    """Run XGBoost bundles (typo, domain-age, DOM); returns dict or None if no models."""
    typo_bundle = getattr(app.state, "xg_typo_bundle", None)
    domain_bundle = getattr(app.state, "xg_domain_bundle", None)
    dom_bundle = getattr(app.state, "xg_dom_bundle", None)
    if typo_bundle is None and domain_bundle is None and dom_bundle is None:
        return None

    url = _normalize_url_for_xgboost(raw_url)
    output = {"url": url}

    if typo_bundle is not None:
        typo_label, typo_prob, typo_feature_map = predict_url(
            typo_bundle,
            url,
            enable_domain_age=False,
            enable_ssl=bool(typo_bundle.meta.get("enable_ssl", False)),
            domain_only=False,
        )
        output["typo_probability"] = round(float(typo_prob), 6)
        output["typo_label"] = int(typo_label)
    else:
        typo_feature_map = {}

    if domain_bundle is not None:
        domain_label, domain_prob, domain_feature_map = predict_url(
            domain_bundle,
            url,
            enable_domain_age=True,
            enable_ssl=bool(domain_bundle.meta.get("enable_ssl", False)),
            domain_only=True,
        )
        output["domain_probability"] = round(float(domain_prob), 6)
        output["domain_label"] = int(domain_label)
    else:
        domain_feature_map = {}

    if dom_bundle is not None:
        dom_label, dom_prob, dom_feature_map = predict_url_dom(dom_bundle, url)
        output["dom_probability"] = round(float(dom_prob), 6)
        output["dom_label"] = int(dom_label)
        output["dom_features"] = {
            k: round(float(v), 6) for k, v in dom_feature_map.items()
        }
    else:
        dom_feature_map = {}

    typo_prob = float(output.get("typo_probability", 0.0))
    domain_prob = float(output.get("domain_probability", 0.0))
    dom_prob = float(output.get("dom_probability", 0.0))
    # CLI(XG_infer)와 동일한 가중치·게이트 (XG_core.xgboost_weighted_ensemble_verdict)
    final_prob, verdict_label = xgboost_weighted_ensemble_verdict(
        typo_prob, domain_prob, dom_prob
    )
    output["final_probability"] = round(float(final_prob), 6)
    output["label"] = int(verdict_label)
    output["verdict"] = "malicious" if verdict_label == 1 else "benign"
    explain_threshold = float(os.getenv("XG_EXPLAIN_THRESHOLD", "0.5"))
    output["explanations"] = build_all_explanations(
        url=url,
        typo_feat_map=typo_feature_map,
        typo_probability=typo_prob if typo_bundle is not None else 0.0,
        domain_feat_map=domain_feature_map,
        domain_probability=domain_prob if domain_bundle is not None else 0.0,
        dom_feature_map=dom_feature_map,
        dom_probability=dom_prob if dom_bundle is not None else 0.0,
        verdict_label=int(verdict_label),
        threshold=explain_threshold,
    )
    return output


def _run_gnn_inference(raw_url: str):
    """Web-structure GNN (gnn_model.pkl); None if model not loaded."""
    model = getattr(app.state, "gnn_model", None)
    cols = getattr(app.state, "gnn_columns", None)
    if model is None or not cols:
        return None
    try:
        return predict_gnn(model, cols, raw_url)
    except Exception as e:
        return {"error": str(e), "verdict": "unknown", "enabled": True}


def _run_url_ml_inference(raw_url: str):
    model = getattr(app.state, "url_ml_model", None)
    if model is None:
        return None
    return predict_url_ml(model, raw_url)


def _log_line_kobert(result: dict) -> str:
    j = result.get("judgment", "?")
    rl = result.get("riskLevel", result.get("risklevel", "?"))
    if result.get("engine_disabled"):
        return f"KoBERT: disabled ({result.get('engine_reason', '')})"
    return f"KoBERT: judgment={j} riskLevel={rl}"


def _log_line_xgboost(xg: object) -> str:
    if xg is None:
        return "XGBoost: skipped (no model)"
    parts = [f"verdict={xg.get('verdict')}", f"final_p={xg.get('final_probability')}"]
    if xg.get("dom_probability") is not None:
        parts.append(f"dom_p={xg.get('dom_probability')}")
    return "XGBoost: " + " ".join(parts)


def _log_line_gnn(gnn: object) -> str:
    if gnn is None:
        st = getattr(app.state, "gnn_status", {}) or {}
        reason = st.get("reason") or "no_model_loaded"
        return f"GNN(web graph): skipped — {reason}"
    if isinstance(gnn, dict) and gnn.get("error"):
        return f"GNN(web graph): error {gnn.get('error', '')[:80]}"
    return f"GNN(web graph): verdict={gnn.get('verdict')} p={gnn.get('probability')}"


def _log_line_url_ml(url_ml: object) -> str:
    if url_ml is None:
        st = getattr(app.state, "url_ml_status", {}) or {}
        reason = st.get("reason") or "no_model_loaded"
        return f"URLML: skipped — {reason}"
    if isinstance(url_ml, dict) and url_ml.get("error"):
        return f"URLML: error {url_ml.get('error', '')[:80]}"
    return f"URLML: verdict={url_ml.get('verdict')} p={url_ml.get('probability')}"


def _risk_from_model_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"normal", "safe", "low", "benign", "clean", "allow", "allowed", "0"}:
        return "SAFE"
    if text in {
        "unnormal",
        "abnormal",
        "high",
        "danger",
        "dangerous",
        "malicious",
        "phishing",
        "scam",
        "block",
        "blocked",
        "1",
    }:
        return "DANGEROUS"
    return "UNKNOWN"


def _verdict_from_risk(risk_level: str) -> str:
    if risk_level == "SAFE":
        return "benign"
    if risk_level == "DANGEROUS":
        return "malicious"
    return "unknown"


def _judgment_from_risk(risk_level: str) -> str:
    if risk_level == "SAFE":
        return "normal"
    if risk_level == "DANGEROUS":
        return "unnormal"
    return "unknown"


def _korean_risk_text(risk_level: str) -> str:
    if risk_level == "SAFE":
        return "정상"
    if risk_level == "DANGEROUS":
        return "악성"
    return "의심"


def _extract_model_risk(result: Any) -> str:
    if not isinstance(result, dict):
        return "UNKNOWN"
    for key in ("judgment", "verdict", "riskLevel", "risklevel", "risk_level", "label"):
        if key in result:
            risk = _risk_from_model_text(result.get(key))
            if risk != "UNKNOWN":
                return risk
    return "UNKNOWN"


def _url_rule_adjustment(url: str) -> dict[str, Any] | None:
    """Conservative URL-only override shared by all API lanes."""
    if is_trusted_official_url(url):
        return {
            "riskLevel": "SAFE",
            "judgment": "normal",
            "verdict": "benign",
            "probability": 0.0,
            "reason": "공식/신뢰 도메인 사전 통과",
        }
    score = float(strong_url_phishing_score(url))
    if score >= 0.66:
        return {
            "riskLevel": "DANGEROUS",
            "judgment": "unnormal",
            "verdict": "malicious",
            "probability": score,
            "reason": "강한 URL 피싱 패턴 사전 감지",
        }
    return None


def _model_probability(model: str, result: Any, risk_level: str | None = None) -> float | None:
    if not isinstance(result, dict):
        return None
    if result.get("probability") is not None and model in {"URLML", "URLHeuristic"}:
        try:
            return max(0.0, min(1.0, float(result.get("probability"))))
        except (TypeError, ValueError):
            return None
    if model == "XGBoost":
        for key in ("final_probability", "probability", "typo_probability", "domain_probability", "dom_probability"):
            if result.get(key) is not None:
                try:
                    return max(0.0, min(1.0, float(result.get(key))))
                except (TypeError, ValueError):
                    continue
    elif model == "GNN":
        if result.get("probability") is not None:
            try:
                return max(0.0, min(1.0, float(result.get("probability"))))
            except (TypeError, ValueError):
                return None
    elif model == "KoBERT":
        if result.get("threat_score") is not None:
            try:
                score = float(result.get("threat_score"))
                return max(0.0, min(1.0, score / 100.0 if score > 1.0 else score))
            except (TypeError, ValueError):
                pass

    risk = risk_level or _extract_model_risk(result)
    if risk == "DANGEROUS":
        return 0.75
    if risk == "SAFE":
        return 0.10
    return None


def _apply_url_rule_adjustment(model: str, url: str, result: Any) -> Any:
    adjustment = _url_rule_adjustment(url)
    if adjustment is None:
        return result

    out = dict(result) if isinstance(result, dict) else {}
    out.update(
        {
            "riskLevel": adjustment["riskLevel"],
            "risklevel": adjustment["riskLevel"],
            "judgment": adjustment["judgment"],
            "verdict": adjustment["verdict"],
            "adjusted_by_rule": True,
            "adjustment_reason": adjustment["reason"],
        }
    )
    prob = float(adjustment["probability"])
    if model == "XGBoost":
        out["final_probability"] = round(prob, 6)
        out["label"] = 1 if adjustment["riskLevel"] == "DANGEROUS" else 0
        out.setdefault("explanations", [adjustment["reason"]])
    elif model == "GNN":
        out["probability"] = round(prob, 6)
        out["label"] = 1 if adjustment["riskLevel"] == "DANGEROUS" else 0
        out.setdefault("explanation", [adjustment["reason"]])
    elif model == "KoBERT":
        out["threat_score"] = round(prob * 100.0, 1)
        out.setdefault("threat_type", "URL 구조 기반 판정" if prob else "안전(공식/신뢰 도메인)")
        out.setdefault(
            "evidence",
            {
                "heuristic_evidence": {"detected_actions": [], "rule_trigger": adjustment["reason"]},
                "ai_semantic_evidence": {
                    "suspect_sentence": url,
                    "ai_inference_logic": adjustment["reason"],
                },
            },
        )
    return out


def _url_heuristic_result(url: str) -> dict[str, Any]:
    score = float(url_heuristic_phishing_score(url))
    ensemble_probability: float | None = score
    if is_trusted_official_url(url):
        risk_level = "SAFE"
        reason = "공식/신뢰 도메인 URL 휴리스틱 통과"
        ensemble_probability = 0.0
    elif score >= 0.66:
        risk_level = "DANGEROUS"
        reason = "강한 URL 휴리스틱 악성 패턴"
    elif score >= 0.35:
        risk_level = "UNKNOWN"
        reason = "중간 강도 URL 휴리스틱 의심 패턴"
    else:
        risk_level = "SAFE"
        reason = "URL 휴리스틱 특이사항 낮음"
        ensemble_probability = None
    return {
        "model": "URLHeuristic",
        "available": True,
        "riskLevel": risk_level,
        "judgment": _judgment_from_risk(risk_level),
        "verdict": _verdict_from_risk(risk_level),
        "summary": f"URLHeuristic: {_korean_risk_text(risk_level)}",
        "probability": round(ensemble_probability, 6) if ensemble_probability is not None else None,
        "raw_probability": round(score, 6),
        "evidence_reasons": [reason],
    }


def _model_detail(
    model: str,
    result: Any,
    status: dict | None = None,
) -> dict[str, Any]:
    if result is None:
        reason = (status or {}).get("reason") or "model_not_loaded"
        return {
            "model": model,
            "available": False,
            "riskLevel": "UNKNOWN",
            "judgment": "unknown",
            "verdict": "unknown",
            "summary": f"{model} 모델은 실행되지 않았습니다: {reason}",
        }

    if not isinstance(result, dict):
        return {
            "model": model,
            "available": True,
            "riskLevel": "UNKNOWN",
            "judgment": "unknown",
            "verdict": "unknown",
            "summary": f"{model} 모델 결과 형식이 예상과 다릅니다.",
            "raw": result,
        }

    risk_level = _extract_model_risk(result)
    judgment = str(result.get("judgment") or _judgment_from_risk(risk_level))
    verdict = str(result.get("verdict") or _verdict_from_risk(risk_level))
    probability = _model_probability(model, result, risk_level)
    summary = f"{model}: {_korean_risk_text(risk_level)}"
    if result.get("error"):
        summary = f"{model}: 의심"
    elif result.get("engine_disabled"):
        summary = f"{model}: 의심"

    detail = {
        "model": model,
        "available": not (result.get("engine_disabled") or result.get("status") == "unavailable"),
        "riskLevel": risk_level,
        "judgment": judgment,
        "verdict": verdict,
        "summary": summary,
        "probability": round(probability, 6) if probability is not None else None,
        "adjusted_by_rule": bool(result.get("adjusted_by_rule")),
        "adjustment_reason": result.get("adjustment_reason"),
    }

    if model == "KoBERT":
        evidence = result.get("evidence") or {}
        semantic = evidence.get("ai_semantic_evidence") or {}
        evidence_reasons = [
            f"문맥 분석 결과 - {semantic.get('ai_inference_logic')}"
            if semantic.get("ai_inference_logic")
            else None,
        ]
        detail.update(
            {
                "threat_type": result.get("threat_type"),
                "site_category": result.get("site_category"),
                "evidence_reasons": [r for r in evidence_reasons if r],
                "ai_inference_logic": semantic.get("ai_inference_logic"),
            },
        )
    elif model == "XGBoost":
        explanations = _clean_xgboost_explanations(result.get("explanations") or [])
        detail.update(
            {
                "evidence_reasons": explanations,
                "typo_label": result.get("typo_label"),
                "domain_label": result.get("domain_label"),
                "dom_label": result.get("dom_label"),
            },
        )
    elif model == "GNN":
        probability = result.get("probability")
        evidence_reasons = _clean_gnn_explanations(result.get("explanation"))
        detail.update(
            {
                "evidence_reasons": evidence_reasons,
                "model_type": result.get("model_type"),
                "probability": probability,
            },
        )
    elif model == "URLML":
        evidence_reasons = []
        if result.get("adjustment_reason"):
            evidence_reasons.append(str(result.get("adjustment_reason")))
        elif result.get("ml_probability") is not None:
            evidence_reasons.append(
                f"URL 문자열 ML 확률 {float(result.get('ml_probability')):.3f}, "
                f"휴리스틱 확률 {float(result.get('heuristic_probability', 0.0)):.3f}"
            )
        detail.update({"evidence_reasons": evidence_reasons})

    return {k: v for k, v in detail.items() if v not in (None, "", [])}


def _clean_xgboost_explanations(explanations: list[Any]) -> list[str]:
    cleaned = []
    for item in explanations:
        text = str(item).strip()
        if not text or text.startswith("["):
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        if text:
            cleaned.append(text)
    return cleaned


def _clean_gnn_explanations(explanation: Any) -> list[str]:
    if not explanation:
        return []
    # build_explanation 이 List[str] 을 반환하므로 그대로 정리
    if isinstance(explanation, list):
        items = [str(x) for x in explanation]
    else:
        # 과거 호환: 단일 문자열이면 줄 단위 분할
        items = str(explanation).splitlines()

    cleaned = []
    for raw in items:
        text = str(raw).strip()
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        # 과거 emoji 접두 제거 (안전망)
        for emoji in ("🚨 ", "⚠️ ", "🔍 ", "✅ ", "🟡 ", "🔥 ", "🎯 "):
            if text.startswith(emoji):
                text = text[len(emoji):]
        if text:
            cleaned.append(text)
    return cleaned


def _xgboost_dominant_signal(result: dict[str, Any]) -> str:
    signals = [
        ("타이포스쿼팅", result.get("typo_probability"), result.get("typo_label")),
        ("도메인 평판", result.get("domain_probability"), result.get("domain_label")),
        ("DOM 구조", result.get("dom_probability"), result.get("dom_label")),
    ]
    present = [
        (name, float(prob), label)
        for name, prob, label in signals
        if prob is not None
    ]
    if not present:
        return "사용 가능한 세부 신호 없음"
    name, prob, label = max(present, key=lambda item: item[1])
    return f"{name} 신호가 가장 강함(label={label}, probability={prob:.3f})"


def _decide_final_risk(details: list[dict[str, Any]]) -> str:
    official_override = any(
        detail.get("adjusted_by_rule")
        and detail.get("riskLevel") == "SAFE"
        and "공식/신뢰" in str(detail.get("adjustment_reason") or "")
        for detail in details
    )
    if official_override:
        return "SAFE"

    strong_url_override = any(
        detail.get("adjusted_by_rule")
        and detail.get("riskLevel") == "DANGEROUS"
        and "강한 URL" in str(detail.get("adjustment_reason") or "")
        for detail in details
    )
    if strong_url_override:
        return "DANGEROUS"

    by_model = {str(detail.get("model")): detail for detail in details}
    url_ml = by_model.get("URLML") or {}
    url_heuristic = by_model.get("URLHeuristic") or {}
    kobert = by_model.get("KoBERT") or {}
    if (
        url_ml.get("available")
        and url_ml.get("riskLevel") == "SAFE"
        and url_heuristic.get("available")
        and url_heuristic.get("riskLevel") == "SAFE"
        and kobert.get("riskLevel") != "DANGEROUS"
    ):
        return "SAFE"

    if url_ml.get("available") and url_ml.get("riskLevel") == "DANGEROUS":
        try:
            url_ml_prob = float(url_ml.get("probability") or 0.0)
        except Exception:
            url_ml_prob = 0.0
        supporting_danger = sum(
            1
            for name in ("KoBERT", "XGBoost", "GNN", "URLHeuristic")
            if (by_model.get(name) or {}).get("riskLevel") == "DANGEROUS"
        )
        if url_ml_prob >= 0.60 or supporting_danger >= 1:
            return "DANGEROUS"

    usable_probs = [
        float(detail["probability"])
        for detail in details
        if detail.get("available") and detail.get("probability") is not None
    ]
    if usable_probs:
        avg_prob = sum(usable_probs) / len(usable_probs)
        danger_threshold = float(os.getenv("FINAL_DANGER_THRESHOLD", "0.40"))
        unknown_threshold = float(os.getenv("FINAL_UNKNOWN_THRESHOLD", "0.30"))
        if avg_prob >= danger_threshold:
            return "DANGEROUS"
        if avg_prob >= unknown_threshold:
            return "UNKNOWN"
        return "SAFE"

    malicious_count = sum(
        1 for detail in details
        if detail.get("available") and detail.get("riskLevel") == "DANGEROUS"
    )
    if malicious_count >= 2:
        return "DANGEROUS"
    if malicious_count == 1:
        return "UNKNOWN"
    return "SAFE"


def _detail_reason_lines(details: list[dict[str, Any]]) -> list[str]:
    lines = []
    for detail in details:
        evidence_reasons = detail.get("evidence_reasons") or []
        for reason in evidence_reasons:
            lines.append(f"{detail['model']} : {reason}")
    return lines


def _build_final_response(
    target_url: str,
    kobert_result: dict,
    xg_result: object,
    gnn_result: object,
    dur_wall: float,
    t_kobert: float,
    t_xg: float,
    t_gnn: float,
    url_ml_result: object = None,
    t_url_ml: float = 0.0,
) -> dict:
    kobert_result = _apply_url_rule_adjustment("KoBERT", target_url, kobert_result)
    xg_result = _apply_url_rule_adjustment("XGBoost", target_url, xg_result)
    gnn_result = _apply_url_rule_adjustment("GNN", target_url, gnn_result)
    url_ml_result = _apply_url_rule_adjustment("URLML", target_url, url_ml_result)
    details = [
        _model_detail("KoBERT", kobert_result, getattr(app.state, "eng_status", {})),
        _model_detail("XGBoost", xg_result, getattr(app.state, "xg_status", {})),
        _model_detail("GNN", gnn_result, getattr(app.state, "gnn_status", {})),
        _model_detail("URLML", url_ml_result, getattr(app.state, "url_ml_status", {})),
        _url_heuristic_result(target_url),
    ]
    risk_level = _decide_final_risk(details)
    judgment = _judgment_from_risk(risk_level)
    malicious_count = sum(
        1 for detail in details
        if detail.get("available") and detail.get("riskLevel") == "DANGEROUS"
    )
    reasons = [
        f"최종 판단 - {_korean_risk_text(risk_level)} (악성 판정 모델 {malicious_count}개)",
        *_detail_reason_lines(details),
    ]
    return {
        "url": target_url,
        "judgment": judgment,
        "riskLevel": risk_level,
        "conclusion": _korean_risk_text(risk_level),
        "decision_method": "score_weighted_ensemble",
        "reasons": reasons,
        "model_details": details,
        "koBERT": kobert_result,
        "xgboost": xg_result,
        "gnn": gnn_result,
        "url_ml": url_ml_result,
        "engine_status": getattr(app.state, "eng_status", {"enabled": False}),
        "xgboost_status": getattr(app.state, "xg_status", {"enabled": False}),
        "gnn_status": getattr(app.state, "gnn_status", {"enabled": False}),
        "url_ml_status": getattr(app.state, "url_ml_status", {"enabled": False}),
        "duration_sec": round(dur_wall, 3),
        "timing": {
            "koBERT_sec": round(t_kobert, 6),
            "xgboost_sec": round(t_xg, 6),
            "gnn_sec": round(t_gnn, 6),
            "url_ml_sec": round(t_url_ml, 6),
            "total_wall_sec": round(dur_wall, 6),
        },
    }


class URLRequest(BaseModel):
    url: str


@app.on_event("startup")
async def startup_event():
    print("--- [1/2] Loading models (import only) ---")
    # Same contract as koBERT.warmup_engine / predict_phishing_result
    app.state.eng = None
    app.state.eng_status = {"enabled": False, "reason": "not_loaded"}
    try:
        import koBERT as eng
        app.state.eng = eng
        app.state.eng_status = {"enabled": True}
    except Exception as e:
        app.state.eng_status = {"enabled": False, "reason": str(e)}
        print(f"[warn] KoBERT engine load failed: {e}")

    app.state.warmup_done = False
    app.state.warmup_info = None
    app.state.xg_typo_bundle = None
    app.state.xg_domain_bundle = None
    app.state.xg_dom_bundle = None
    app.state.xg_status = {"enabled": False, "reason": "not_loaded"}
    app.state.gnn_model = None
    app.state.gnn_columns = None
    app.state.gnn_status = {"enabled": False, "reason": "not_loaded"}
    app.state.gnn_engine = None
    app.state.url_ml_model = None
    app.state.url_ml_status = {"enabled": False, "reason": "not_loaded"}

    try:
        url_ml_model, url_ml_status = load_url_ml_model()
        app.state.url_ml_model = url_ml_model
        app.state.url_ml_status = url_ml_status.__dict__
        if url_ml_model is not None:
            sm = predict_url_ml(url_ml_model, "https://example.com")
            print(f"  [URLML] smoke OK — verdict={sm.get('verdict')} p={sm.get('probability')}")
        else:
            print(f"  [URLML] skipped — {url_ml_status.reason}")
    except Exception as e:
        app.state.url_ml_status = {"enabled": False, "reason": str(e)}

    # Optional: web-structure GNN (gnn_engine: gnn_model.pkl + gnn_model_features.pkl)
    try:
        eng_gnn = GNN_Engine()
        app.state.gnn_engine = eng_gnn
        if eng_gnn.ok:
            app.state.gnn_model = eng_gnn.model
            app.state.gnn_columns = eng_gnn.columns
            app.state.gnn_status = {
                "enabled": True,
                "model_path": eng_gnn.model_path,
                "feature_columns_path": eng_gnn.feature_columns_path
                if os.path.isfile(eng_gnn.feature_columns_path)
                else None,
            }
        else:
            reason = eng_gnn._load_error or "model_not_loaded"
            if reason.startswith("missing_model:"):
                app.state.gnn_status = {"enabled": False, "reason": reason}
            else:
                app.state.gnn_status = {"enabled": False, "reason": reason}
    except Exception as e:
        app.state.gnn_status = {"enabled": False, "reason": str(e)}

    # Smoke predict after GNN load
    if app.state.gnn_model is not None and app.state.gnn_columns is not None:
        try:
            sm = predict_gnn(
                app.state.gnn_model,
                app.state.gnn_columns,
                "https://example.com",
            )
            print(
                f"  [GNN] smoke OK — verdict={sm.get('verdict')} "
                f"p={sm.get('probability')}"
            )
        except Exception as e:
            print(f"  [GNN] smoke failed (requests may fail too): {e}")

    # Optional XGBoost bundles (typo, domain-age, DOM — same as XG_infer.py)
    xg_typo_path = os.getenv(
        "XG_MODEL_TYPO",
        os.path.join(MODEL_DIRS["xgboost"], "url_xgb_paired_first.joblib"),
    )
    xg_domain_path = os.getenv(
        "XG_MODEL_DOMAIN",
        os.path.join(MODEL_DIRS["xgboost"], "url_xgb_domain_age.joblib"),
    )
    xg_dom_path = os.getenv(
        "XG_MODEL_DOM",
        os.path.join(MODEL_DIRS["xgboost"], "url_xgb_dom.joblib"),
    )
    xg_errors = []
    try:
        if os.path.isfile(xg_typo_path):
            app.state.xg_typo_bundle = load_bundle(xg_typo_path)
        else:
            xg_errors.append(f"missing_typo_model:{xg_typo_path}")
    except Exception as e:
        xg_errors.append(f"typo_load_error:{e}")

    try:
        if os.path.isfile(xg_domain_path):
            app.state.xg_domain_bundle = load_bundle(xg_domain_path)
        else:
            xg_errors.append(f"missing_domain_model:{xg_domain_path}")
    except Exception as e:
        xg_errors.append(f"domain_load_error:{e}")

    try:
        if os.path.isfile(xg_dom_path):
            app.state.xg_dom_bundle = load_bundle(xg_dom_path)
        else:
            xg_errors.append(f"missing_dom_model:{xg_dom_path}")
    except Exception as e:
        xg_errors.append(f"dom_load_error:{e}")

    if (
        app.state.xg_typo_bundle is not None
        or app.state.xg_domain_bundle is not None
        or app.state.xg_dom_bundle is not None
    ):
        app.state.xg_status = {
            "enabled": True,
            "typo_loaded": app.state.xg_typo_bundle is not None,
            "domain_loaded": app.state.xg_domain_bundle is not None,
            "dom_loaded": app.state.xg_dom_bundle is not None,
            "warnings": xg_errors,
        }
    else:
        app.state.xg_status = {
            "enabled": False,
            "warnings": xg_errors,
        }

    print("--- [2/2] Warmup (warmup_engine) ---")
    if app.state.eng is not None:
        try:
            info = await _run_engine(app.state.eng.warmup_engine, STARTUP_WARMUP_PLAYWRIGHT)
            app.state.warmup_info = info
            app.state.warmup_done = True
        except Exception as e:
            print(f"[warmup failed] {e}")
            app.state.warmup_done = False
            # Server still starts; /analyze may retry Playwright as needed.
            app.state.warmup_info = {"error": str(e)}
    else:
        app.state.warmup_done = False
        app.state.warmup_info = {"skipped": "engine_not_loaded"}

    print("--- [system] Ready. /analyze runs full checks. ---")
    eng = getattr(app.state, "eng", None)
    if eng is None:
        r = getattr(app.state, "eng_status", {}).get("reason", "")
        print(
            "  [hint] KoBERT engine did not load (import failed). "
            "/ready will show ready=false until fixed. "
            "Install deps with the SAME Python that runs uvicorn, e.g.: "
            "python -m pip install -r requirements.txt"
        )
        if r:
            print(f"  [hint] Import error was: {r}")


@app.get("/health")
async def health():
    """Liveness: process is up (use for K8s livenessProbe). Does not check ML deps."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: KoBERT warmup finished. False if import failed or warmup errored."""
    eng_ok = getattr(app.state, "eng", None) is not None
    warmup_done = bool(getattr(app.state, "warmup_done", False))
    eng_status = getattr(app.state, "eng_status", {"enabled": False})
    issues = []
    if not eng_ok:
        issues.append(
            f"koBERT import failed: {eng_status.get('reason', 'unknown')}. "
            "Use the same interpreter for pip and uvicorn (python -m pip install -r requirements.txt)."
        )
    elif not warmup_done:
        wi = getattr(app.state, "warmup_info", None)
        issues.append(f"warmup incomplete: {wi}")

    return {
        "http_ok": True,
        "ready": warmup_done and eng_ok,
        "kobert_import_ok": eng_ok,
        "warmup_done": warmup_done,
        "warmup": getattr(app.state, "warmup_info", None),
        "engine": eng_status,
        "xgboost": getattr(app.state, "xg_status", {"enabled": False}),
        "gnn": getattr(app.state, "gnn_status", {"enabled": False}),
        "url_ml": getattr(app.state, "url_ml_status", {"enabled": False}),
        "issues": issues,
    }


@app.post("/warmup")
async def warmup_manual():
    """Manual warmup if Playwright was skipped at boot."""
    eng = getattr(app.state, "eng", None)
    if eng is None:
        raise HTTPException(
            status_code=503,
            detail=f"Engine not loaded: {getattr(app.state, 'eng_status', {}).get('reason', 'unknown')}",
        )
    include_pw = os.getenv("WARMUP_PLAYWRIGHT", "1") == "1"
    t0 = time.perf_counter()
    try:
        info = await _run_engine(eng.warmup_engine, include_pw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    app.state.warmup_info = info
    app.state.warmup_done = True
    dur = time.perf_counter() - t0
    return {"ok": True, "warmup": info, "duration_sec": round(dur, 3)}


@app.post("/analyze")
async def analyze_url(request: URLRequest):
    """Parallel koBERT, XGBoost, GNN via asyncio.gather."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")

    print(f"--- [analyze] URL: {target_url} ---")

    t_wall0 = time.perf_counter()
    try:

        # Three branches in parallel (not sequential)
        async def _kobert_timed():
            t0 = time.perf_counter()
            eng = getattr(app.state, "eng", None)
            if eng is not None:
                r = await _run_engine(eng.predict_phishing_result, target_url)
            else:
                r = {
                    "judgment": "unknown",
                    "riskLevel": "UNKNOWN",
                    "risklevel": "UNKNOWN",
                    "engine_disabled": True,
                    "engine_reason": getattr(app.state, "eng_status", {}).get("reason", "not_loaded"),
                }
            return r, time.perf_counter() - t0

        async def _xg_timed():
            t0 = time.perf_counter()
            r = await _run_xgboost(_run_xgboost_inference, target_url)
            return r, time.perf_counter() - t0

        async def _gnn_timed():
            t0 = time.perf_counter()
            r = await _run_gnn(_run_gnn_inference, target_url)
            return r, time.perf_counter() - t0

        async def _url_ml_timed():
            t0 = time.perf_counter()
            r = await _run_url_ml(_run_url_ml_inference, target_url)
            return r, time.perf_counter() - t0

        (
            (kobert_result, t_kobert),
            (xg_result, t_xg),
            (gnn_result, t_gnn),
            (url_ml_result, t_url_ml),
        ) = await asyncio.gather(
            _kobert_timed(),
            _xg_timed(),
            _gnn_timed(),
            _url_ml_timed(),
        )
    except Exception as e:
        print(f"[error] {e}")
        raise HTTPException(status_code=500, detail=str(e))

    dur_wall = time.perf_counter() - t_wall0
    print(
        f"--- [analyze summary] URL: {target_url}  (koBERT | xgboost | gnn parallel)\n"
        f"    {_log_line_kobert(kobert_result)}  ({t_kobert:.3f}s)\n"
        f"    {_log_line_xgboost(xg_result)}  ({t_xg:.3f}s)\n"
        f"    {_log_line_gnn(gnn_result)}  ({t_gnn:.3f}s)\n"
        f"    {_log_line_url_ml(url_ml_result)}  ({t_url_ml:.3f}s)\n"
        f"    wall time (parallel): {dur_wall:.3f}s"
    )
    return _build_final_response(
        target_url=target_url,
        kobert_result=kobert_result,
        xg_result=xg_result,
        gnn_result=gnn_result,
        dur_wall=dur_wall,
        t_kobert=t_kobert,
        t_xg=t_xg,
        t_gnn=t_gnn,
        url_ml_result=url_ml_result,
        t_url_ml=t_url_ml,
    )


@app.post("/analyze/engine")
async def analyze_engine_only(request: URLRequest):
    """KoBERT only (e.g. step 1 in UI)."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    eng = getattr(app.state, "eng", None)
    if eng is not None:
        result = await _run_engine(eng.predict_phishing_result, target_url)
    else:
        result = {
            "judgment": "unknown",
            "riskLevel": "UNKNOWN",
            "risklevel": "UNKNOWN",
            "engine_disabled": True,
            "engine_reason": getattr(app.state, "eng_status", {}).get("reason", "not_loaded"),
        }
    result = _apply_url_rule_adjustment("KoBERT", target_url, result)
    return {
        **result,
        "engine_status": getattr(app.state, "eng_status", {"enabled": False}),
    }


@app.post("/analyze/xgboost")
async def analyze_xgboost_only(request: URLRequest):
    """XGBoost only; use /analyze/gnn for GNN separately."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    xg_result = await _run_xgboost(_run_xgboost_inference, target_url)
    xg_result = _apply_url_rule_adjustment("XGBoost", target_url, xg_result)
    return {
        "xgboost": xg_result,
        "xgboost_status": getattr(app.state, "xg_status", {"enabled": False}),
    }


@app.post("/analyze/gnn")
async def analyze_gnn_only(request: URLRequest):
    """Web-structure GNN only."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    gnn_result = await _run_gnn(_run_gnn_inference, target_url)
    gnn_result = _apply_url_rule_adjustment("GNN", target_url, gnn_result)
    return {
        "gnn": gnn_result,
        "gnn_status": getattr(app.state, "gnn_status", {"enabled": False}),
    }


@app.post("/analyze/url-ml")
async def analyze_url_ml_only(request: URLRequest):
    """Fast URL lexical ML only."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    url_ml_result = await _run_url_ml(_run_url_ml_inference, target_url)
    url_ml_result = _apply_url_rule_adjustment("URLML", target_url, url_ml_result)
    return {
        "url_ml": url_ml_result,
        "url_ml_status": getattr(app.state, "url_ml_status", {"enabled": False}),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
