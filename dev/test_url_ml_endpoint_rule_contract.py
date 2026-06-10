#!/usr/bin/env python3
"""Self-check /analyze/url-ml applies URL rule adjustment without a model."""

from __future__ import annotations

import asyncio
import os
import sys


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import main  # noqa: E402


async def _run() -> None:
    original_model = getattr(main.app.state, "url_ml_model", None)
    original_status = getattr(main.app.state, "url_ml_status", None)
    main.app.state.url_ml_model = None
    main.app.state.url_ml_status = {"enabled": False, "reason": "selfcheck_no_model"}
    try:
        redirector = (
            "https://redirector.test/click?"
            "target=https%253A%252F%252Fwww.apps-offiice-wps.com.cn%252Flogin"
        )
        payload = await main.analyze_url_ml_only(main.URLRequest(url=redirector))
        result = payload["url_ml"]
        assert result["riskLevel"] == "DANGEROUS", payload
        assert result["verdict"] == "malicious", payload
        assert result["adjusted_by_rule"] is True, payload
        assert "redirect" in result["adjustment_reason"], payload

        trusted = await main.analyze_url_ml_only(main.URLRequest(url="https://trustwallet.com/"))
        assert trusted["url_ml"]["riskLevel"] == "SAFE", trusted
    finally:
        main.app.state.url_ml_model = original_model
        main.app.state.url_ml_status = original_status


def main_check() -> int:
    asyncio.run(_run())
    print("URLML endpoint rule contract selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())
