import asyncio
import functools
import os
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

# Tuning (same env vars as engine module): MAX_SEQ_LEN, USE_PLAYWRIGHT_IN_ANALYZE, etc.
# os.environ.setdefault("MAX_SEQ_LEN", "128")
# os.environ.setdefault("USE_PLAYWRIGHT_IN_ANALYZE", "0")
# os.environ.setdefault("TORCH_NUM_THREADS", "4")

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from XG_core import load_bundle, predict_url
from gnn_engine import GNN_Engine, predict_gnn

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
# GNN lexical RF (gnn_model.pkl)
_gnn_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gnn_lexical")


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
    """Normalize URL for XGBoost; prepend https if no scheme."""
    candidate = (url or "").strip()
    if not candidate:
        return candidate
    parsed = urlsplit(candidate)
    if parsed.scheme and parsed.netloc:
        return candidate
    return f"https://{candidate}"


def _run_xgboost_inference(raw_url: str):
    """Run XGBoost bundles; returns dict or None if no models."""
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
            enable_ssl=bool(typo_bundle.meta.get("enable_ssl", False)),
            domain_only=False,
        )
        output["typo_probability"] = round(float(typo_prob), 6)
        output["typo_label"] = int(typo_label)

    if domain_bundle is not None:
        domain_label, domain_prob, _ = predict_url(
            domain_bundle,
            url,
            enable_domain_age=True,
            enable_ssl=bool(domain_bundle.meta.get("enable_ssl", False)),
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
    """GNN lexical RF (gnn_model.pkl); None if model not loaded."""
    model = getattr(app.state, "gnn_model", None)
    cols = getattr(app.state, "gnn_columns", None)
    if model is None or not cols:
        return None
    try:
        return predict_gnn(model, cols, raw_url)
    except Exception as e:
        return {"error": str(e), "verdict": "unknown", "enabled": True}


def _log_line_kobert(result: dict) -> str:
    j = result.get("judgment", "?")
    rl = result.get("riskLevel", result.get("risklevel", "?"))
    if result.get("engine_disabled"):
        return f"KoBERT: disabled ({result.get('engine_reason', '')})"
    return f"KoBERT: judgment={j} riskLevel={rl}"


def _log_line_xgboost(xg: object) -> str:
    if xg is None:
        return "XGBoost: skipped (no model)"
    return (
        f"XGBoost: verdict={xg.get('verdict')} "
        f"final_p={xg.get('final_probability')}"
    )


def _log_line_gnn(gnn: object) -> str:
    if gnn is None:
        st = getattr(app.state, "gnn_status", {}) or {}
        reason = st.get("reason") or "no_model_loaded"
        # Typical: missing file, or pickle/Python mismatch (see startup logs)
        return f"GNN(lexical RF): skipped — {reason}"
    if isinstance(gnn, dict) and gnn.get("error"):
        return f"GNN(lexical RF): error {gnn.get('error', '')[:80]}"
    return f"GNN(lexical RF): verdict={gnn.get('verdict')} p={gnn.get('probability')}"


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
    app.state.xg_status = {"enabled": False, "reason": "not_loaded"}
    app.state.gnn_model = None
    app.state.gnn_columns = None
    app.state.gnn_status = {"enabled": False, "reason": "not_loaded"}
    app.state.gnn_engine = None

    # Optional: GNN lexical RF (gnn_engine: gnn_model.pkl + gnn_model_features.pkl)
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

    # Optional XGBoost bundles
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

        (kobert_result, t_kobert), (xg_result, t_xg), (gnn_result, t_gnn) = await asyncio.gather(
            _kobert_timed(),
            _xg_timed(),
            _gnn_timed(),
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
        f"    wall time (parallel): {dur_wall:.3f}s"
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
    return {
        "xgboost": xg_result,
        "xgboost_status": getattr(app.state, "xg_status", {"enabled": False}),
    }


@app.post("/analyze/gnn")
async def analyze_gnn_only(request: URLRequest):
    """GNN lexical RF only."""
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL is empty.")
    gnn_result = await _run_gnn(_run_gnn_inference, target_url)
    return {
        "gnn": gnn_result,
        "gnn_status": getattr(app.state, "gnn_status", {"enabled": False}),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
