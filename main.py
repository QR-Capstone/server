import asyncio
import functools
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

# 추론·응답 더 줄이려면(환경변수, 엔진 모듈 상단과 동일):
# os.environ.setdefault("MAX_SEQ_LEN", "128")
# os.environ.setdefault("USE_PLAYWRIGHT_IN_ANALYZE", "0")
# os.environ.setdefault("TORCH_NUM_THREADS", "4")

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from XG_core import load_bundle, predict_url
from gnn_infer import load_opqr_model, predict_opqr

app = FastAPI(title="Phishing Detection API")


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """요청 수신부터 응답 완료까지(전체) 걸린 시간 — 터미널 OK 옆에 보이게 출력."""
    t0 = time.perf_counter()
    response = await call_next(request)
    dur = time.perf_counter() - t0
    response.headers["X-Process-Time"] = f"{dur:.3f}"
    print(
        f"--- [요청 완료] {request.method} {request.url.path} "
        f"{response.status_code} OK ({dur:.3f}s)"
    )
    return response

# 기동 시 Playwright까지 예열할지 (기본은 끄는 것이 훨씬 빠릅니다)
STARTUP_WARMUP_PLAYWRIGHT = os.getenv("STARTUP_WARMUP_PLAYWRIGHT", "0") == "1"

# Playwright(sync)는 greenlet 컨텍스트가 스레드에 묶이므로, 엔진 호출은 항상 동일 스레드에서만 실행해야 함.
_engine_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phish_engine")
# KoBERT와 겹쳐 돌리기 위해 XGBoost는 별도 풀(동시에 서로 다른 스레드에서 실행).
_xgboost_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="xgboost_infer")
# OPQR(RandomForest, gnn-ready 학습) 추론용 풀
_gnn_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="opqr_gnn")


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


def _normalize_url_for_xgboost(url: str) -> str:
    """XGBoost 입력 URL 정규화: 스킴이 없으면 https 추가."""
    candidate = (url or "").strip()
    if not candidate:
        return candidate
    parsed = urlsplit(candidate)
    if parsed.scheme and parsed.netloc:
        return candidate
    return f"https://{candidate}"


def _run_xgboost_inference(raw_url: str):
    """로드된 번들로 XGBoost 추론 수행. 반환: dict 또는 None."""
    typo_bundle = getattr(app.state, "xg_typo_bundle", None)
    domain_bundle = getattr(app.state, "xg_domain_bundle", None)
    if typo_bundle is None and domain_bundle is None:
        return None

    url = _normalize_url_for_xgboost(raw_url)
    output = {"url": url}

    if typo_bundle is not None:
        typo_label, typo_prob, _ = predict_url(
            typo_bundle,
            url,
            enable_domain_age=False,
            domain_only=False,
        )
        output["typo_probability"] = round(float(typo_prob), 6)
        output["typo_label"] = int(typo_label)

    if domain_bundle is not None:
        domain_label, domain_prob, _ = predict_url(
            domain_bundle,
            url,
            enable_domain_age=True,
            domain_only=True,
        )
        output["domain_probability"] = round(float(domain_prob), 6)
        output["domain_label"] = int(domain_label)

    typo_prob = float(output.get("typo_probability", 0.0))
    domain_prob = float(output.get("domain_probability", 0.0))
    final_prob = max(typo_prob, domain_prob)
    output["final_probability"] = round(final_prob, 6)
    output["label"] = int(1 if final_prob >= 0.5 else 0)
    output["verdict"] = "malicious" if output["label"] == 1 else "benign"
    return output


def _run_gnn_inference(raw_url: str):
    """gnn-ready 학습 RF(opqr_model.pkl) 추론. 모델 없으면 None."""
    model = getattr(app.state, "gnn_model", None)
    cols = getattr(app.state, "gnn_columns", None)
    if model is None or not cols:
        return None
    try:
        return predict_opqr(model, cols, raw_url)
    except Exception as e:
        return {"error": str(e), "verdict": "unknown", "enabled": True}


def _log_line_kobert(result: dict) -> str:
    j = result.get("judgment", "?")
    rl = result.get("riskLevel", result.get("risklevel", "?"))
    if result.get("engine_disabled"):
        return f"KoBERT: 비활성 ({result.get('engine_reason', '')})"
    return f"KoBERT: judgment={j} riskLevel={rl}"


def _log_line_xgboost(xg: object) -> str:
    if xg is None:
        return "XGBoost: 스킵(모델 없음)"
    return (
        f"XGBoost: verdict={xg.get('verdict')} "
        f"final_p={xg.get('final_probability')}"
    )


def _log_line_gnn(gnn: object) -> str:
    if gnn is None:
        return "GNN(lexical RF): 스킵(모델 없음)"
    if isinstance(gnn, dict) and gnn.get("error"):
        return f"GNN(lexical RF): 오류 {gnn.get('error', '')[:80]}"
    return f"GNN(lexical RF): verdict={gnn.get('verdict')} p={gnn.get('probability')}"


class URLRequest(BaseModel):
    url: str


@app.on_event("startup")
async def startup_event():
    print("--- [1/2] 모델 로드 (import, 검증 아님) ---")
    # 예열·판별 API는 koBERT.warmup_engine / predict_phishing_result 와 동일 계약
    app.state.eng = None
    app.state.eng_status = {"enabled": False, "reason": "not_loaded"}
    try:
        import koBERT as eng
        app.state.eng = eng
        app.state.eng_status = {"enabled": True}
    except Exception as e:
        app.state.eng_status = {"enabled": False, "reason": str(e)}
        print(f"[경고] 코발트 엔진 로드 실패: {e}")

    app.state.warmup_done = False
    app.state.warmup_info = None
    app.state.xg_typo_bundle = None
    app.state.xg_domain_bundle = None
    app.state.xg_status = {"enabled": False, "reason": "not_loaded"}
    app.state.gnn_model = None
    app.state.gnn_columns = None
    app.state.gnn_status = {"enabled": False, "reason": "not_loaded"}

    # OPQR / gnn-ready RandomForest (선택)
    opqr_path = os.getenv("OPQR_MODEL_PATH", "opqr_model.pkl")
    opqr_cols_path = os.getenv("OPQR_FEATURES_PATH", "model_features.pkl")
    try:
        if os.path.isfile(opqr_path):
            m, cols = load_opqr_model(opqr_path, opqr_cols_path)
            app.state.gnn_model = m
            app.state.gnn_columns = cols
            app.state.gnn_status = {
                "enabled": True,
                "model_path": opqr_path,
                "feature_columns_path": opqr_cols_path if os.path.isfile(opqr_cols_path) else None,
            }
        else:
            app.state.gnn_status = {
                "enabled": False,
                "reason": f"missing_model:{opqr_path}",
            }
    except Exception as e:
        app.state.gnn_status = {"enabled": False, "reason": str(e)}

    # GNN(opqr_model.pkl) 로드 후 1회 스모크 추론 — 런타임에서 predict 경로 확인
    if app.state.gnn_model is not None and app.state.gnn_columns is not None:
        try:
            sm = predict_opqr(
                app.state.gnn_model,
                app.state.gnn_columns,
                "https://example.com",
            )
            print(
                f"  [GNN] 스모크 추론 OK — verdict={sm.get('verdict')} "
                f"p={sm.get('probability')}"
            )
        except Exception as e:
            print(f"  [GNN] 스모크 추론 실패(요청 시에도 동일할 수 있음): {e}")

    # XGBoost 번들은 선택 로딩(없어도 서버 동작)
    xg_typo_path = os.getenv("XG_MODEL_TYPO", "url_xgb_paired_first.joblib")
    xg_domain_path = os.getenv("XG_MODEL_DOMAIN", "url_xgb_domain_age.joblib")
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

    if app.state.xg_typo_bundle is not None or app.state.xg_domain_bundle is not None:
        app.state.xg_status = {
            "enabled": True,
            "typo_loaded": app.state.xg_typo_bundle is not None,
            "domain_loaded": app.state.xg_domain_bundle is not None,
            "warnings": xg_errors,
        }
    else:
        app.state.xg_status = {
            "enabled": False,
            "warnings": xg_errors,
        }

    print("--- [2/2] 예열 (warmup_engine: 검증과 분리) ---")
    if app.state.eng is not None:
        try:
            info = await _run_engine(app.state.eng.warmup_engine, STARTUP_WARMUP_PLAYWRIGHT)
            app.state.warmup_info = info
            app.state.warmup_done = True
        except Exception as e:
            print(f"[예열 실패] {e}")
            app.state.warmup_done = False
            # Playwright 의존성이 없거나 느려도 서버는 먼저 떠야 합니다.
            # 실제 분석은 /analyze 호출 시 필요하면 그때 처리합니다.
            app.state.warmup_info = {"error": str(e)}
    else:
        app.state.warmup_done = False
        app.state.warmup_info = {"skipped": "engine_not_loaded"}

    print("--- [시스템] 서버 준비 완료. /analyze 는 검증만 수행합니다. ---")


@app.get("/ready")
async def ready():
    """예열 완료 여부 (로드밸런서/헬스체크용)."""
    return {
        "ready": getattr(app.state, "warmup_done", False),
        "warmup": getattr(app.state, "warmup_info", None),
        "engine": getattr(app.state, "eng_status", {"enabled": False}),
        "xgboost": getattr(app.state, "xg_status", {"enabled": False}),
        "gnn": getattr(app.state, "gnn_status", {"enabled": False}),
    }


@app.post("/warmup")
async def warmup_manual():
    """기동 시 Playwright 생략했으면 나중에 여기서만 예열."""
    eng = getattr(app.state, "eng", None)
    if eng is None:
        raise HTTPException(
            status_code=503,
            detail=f"엔진 미로드: {getattr(app.state, 'eng_status', {}).get('reason', 'unknown')}",
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
    """koBERT(koBERT.py) · XGBoost · GNN(lexical RF) 세 분기를 asyncio.gather 로 병렬 실행."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL이 비어있습니다.")

    print(f"--- [검증] URL: {target_url} ---")

    t_wall0 = time.perf_counter()
    try:

        # 세 분기 koBERT · XGBoost · GNN 은 asyncio.gather 로 동시에 실행 (순차 아님)
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

        (kobert_result, t_kobert), (xg_result, t_xg), (gnn_result, t_gnn) = await asyncio.gather(
            _kobert_timed(),
            _xg_timed(),
            _gnn_timed(),
        )
    except Exception as e:
        print(f"[오류] {e}")
        raise HTTPException(status_code=500, detail=str(e))

    dur_wall = time.perf_counter() - t_wall0
    print(
        f"--- [검증 요약] URL: {target_url}  (koBERT / xgboost / gnn 병렬)\n"
        f"    {_log_line_kobert(kobert_result)}  ({t_kobert:.3f}s)\n"
        f"    {_log_line_xgboost(xg_result)}  ({t_xg:.3f}s)\n"
        f"    {_log_line_gnn(gnn_result)}  ({t_gnn:.3f}s)\n"
        f"    병렬 전체(벽시계): {dur_wall:.3f}s"
    )
    return {
        "url": target_url,
        "koBERT": kobert_result,
        "xgboost": xg_result,
        "gnn": gnn_result,
        "engine_status": getattr(app.state, "eng_status", {"enabled": False}),
        "xgboost_status": getattr(app.state, "xg_status", {"enabled": False}),
        "gnn_status": getattr(app.state, "gnn_status", {"enabled": False}),
        "duration_sec": round(dur_wall, 3),
        "timing": {
            "koBERT_sec": round(t_kobert, 6),
            "xgboost_sec": round(t_xg, 6),
            "gnn_sec": round(t_gnn, 6),
            "total_wall_sec": round(dur_wall, 6),
        },
    }


@app.post("/analyze/engine")
async def analyze_engine_only(request: URLRequest):
    """KoBERT(텍스트) 엔진만 실행 — 클라이언트에서 1단계 진행률용."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL이 비어있습니다.")
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
    return {
        **result,
        "engine_status": getattr(app.state, "eng_status", {"enabled": False}),
    }


@app.post("/analyze/xgboost")
async def analyze_xgboost_only(request: URLRequest):
    """XGBoost(타이포 + 도메인 연령)만 실행 — GNN(`/analyze/gnn`)과는 별도 엔드포인트."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL이 비어있습니다.")
    xg_result = await _run_xgboost(_run_xgboost_inference, target_url)
    return {
        "xgboost": xg_result,
        "xgboost_status": getattr(app.state, "xg_status", {"enabled": False}),
    }


@app.post("/analyze/gnn")
async def analyze_gnn_only(request: URLRequest):
    """OPQR RandomForest(gnn-ready 학습)만 실행."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL이 비어있습니다.")
    gnn_result = await _run_gnn(_run_gnn_inference, target_url)
    return {
        "gnn": gnn_result,
        "gnn_status": getattr(app.state, "gnn_status", {"enabled": False}),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
