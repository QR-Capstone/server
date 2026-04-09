import asyncio
import functools
import os
from concurrent.futures import ThreadPoolExecutor

# 추론·응답 더 줄이려면(환경변수, 엔진 모듈 상단과 동일):
# os.environ.setdefault("MAX_SEQ_LEN", "128")
# os.environ.setdefault("USE_PLAYWRIGHT_IN_ANALYZE", "0")
# os.environ.setdefault("TORCH_NUM_THREADS", "4")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Phishing Detection API")

# 기동 시 Playwright까지 예열할지 (기본은 끄는 것이 훨씬 빠릅니다)
STARTUP_WARMUP_PLAYWRIGHT = os.getenv("STARTUP_WARMUP_PLAYWRIGHT", "0") == "1"

# Playwright(sync)는 greenlet 컨텍스트가 스레드에 묶이므로, 엔진 호출은 항상 동일 스레드에서만 실행해야 함.
_engine_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="phish_engine")


async def _run_engine(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_engine_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_engine_executor, functools.partial(fn, *args))


class URLRequest(BaseModel):
    url: str


@app.on_event("startup")
async def startup_event():
    print("--- [1/2] 모델 로드 (import, 검증 아님) ---")
    # 예열·판별 API는 TEST_27_server.warmup_engine / predict_phishing_result 와 동일 계약
    import TEST_27_server as eng

    app.state.eng = eng
    app.state.warmup_done = False
    app.state.warmup_info = None

    print("--- [2/2] 예열 (warmup_engine: 검증과 분리) ---")
    try:
        info = await _run_engine(eng.warmup_engine, STARTUP_WARMUP_PLAYWRIGHT)
        app.state.warmup_info = info
        app.state.warmup_done = True
    except Exception as e:
        print(f"[예열 실패] {e}")
        app.state.warmup_done = False
        # Playwright 의존성이 없거나 느려도 서버는 먼저 떠야 합니다.
        # 실제 분석은 /analyze 호출 시 필요하면 그때 처리합니다.
        app.state.warmup_info = {"error": str(e)}

    print("--- [시스템] 서버 준비 완료. /analyze 는 검증만 수행합니다. ---")


@app.get("/ready")
async def ready():
    """예열 완료 여부 (로드밸런서/헬스체크용)."""
    return {
        "ready": getattr(app.state, "warmup_done", False),
        "warmup": getattr(app.state, "warmup_info", None),
    }


@app.post("/warmup")
async def warmup_manual():
    """기동 시 Playwright 생략했으면 나중에 여기서만 예열."""
    eng = getattr(app.state, "eng", None)
    if eng is None:
        raise HTTPException(status_code=503, detail="엔진 미로드")
    include_pw = os.getenv("WARMUP_PLAYWRIGHT", "1") == "1"
    try:
        info = await _run_engine(eng.warmup_engine, include_pw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    app.state.warmup_info = info
    app.state.warmup_done = True
    return {"ok": True, "warmup": info}


@app.post("/analyze")
async def analyze_url(request: URLRequest):
    target_url = (request.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="URL이 비어있습니다.")

    print(f"--- [검증] URL: {target_url} ---")

    try:
        eng = app.state.eng
        result = await _run_engine(eng.predict_phishing_result, target_url)
    except Exception as e:
        print(f"[오류] {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)