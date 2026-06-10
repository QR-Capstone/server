#!/usr/bin/env python3
"""Self-check text analysis endpoints preserve extraction and batch contracts."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

import main  # noqa: E402


def _fake_batch_response(raw_urls: list[str], result_key: str) -> dict[str, Any]:
    normalized, keys, unique_urls, first_index = main._prepare_batch_urls(raw_urls)
    return {
        "count": len(raw_urls),
        "unique_count": len(unique_urls),
        "deduplicated": len(unique_urls) < len(raw_urls),
        "duration_sec": 0.0,
        "results": [
            {
                "index": index,
                "url": normalized[index],
                "duplicate_of": first_index[key] if first_index[key] != index else None,
                "result": {result_key: {"riskLevel": "UNKNOWN"}},
            }
            for index, key in enumerate(keys)
        ],
    }


async def _fake_analyze_url_batch(request: main.URLBatchRequest) -> dict[str, Any]:
    return _fake_batch_response(list(request.urls or []), "ensemble")


async def _fake_analyze_url_ml_batch(request: main.URLBatchRequest) -> dict[str, Any]:
    return _fake_batch_response(list(request.urls or []), "url_ml")


async def _run_endpoint_checks() -> None:
    original_batch = main.analyze_url_batch
    original_urlml_batch = main.analyze_url_ml_batch
    main.analyze_url_batch = _fake_analyze_url_batch
    main.analyze_url_ml_batch = _fake_analyze_url_ml_batch
    try:
        text = " ".join(
            [
                "first hxxps[:]//www[.]apps-offiice-wps[.]com[.]cn/login",
                "again https://www.apps-offiice-wps.com.cn/login",
                r"escaped \u{68}\u{74}\u{74}\u{70}\u{73}\u{3a}\u{2f}\u{2f}evil-checkout.test/pay",
                "double https%253A%252F%252Fevil-checkout.test%252Fpay",
                "redirect https://redirector.test/click?target=https%253A%252F%252Fevil-checkout.test%252Fpay",
                "b64redirect https://redirector.test/click?"
                "url=aHR0cHM6Ly9ldmlsLWNoZWNrb3V0LnRlc3QvcGF5",
                "charcode String.fromCharCode("
                "104,116,116,112,115,58,47,47,101,118,105,108,45,99,104,101,99,107,111,117,116,46,"
                "116,101,115,116,47,112,97,121)",
                "wrapped (https://evil-checkout.test/pay_(1))",
                """joined ['hxxps[:]//www','[.]apps-offiice-wps','[.]com','[.]cn/login'].join('')""",
            ]
        )
        ensemble = await main.analyze_text(main.TextAnalyzeRequest(text=text))
        urlml = await main.analyze_url_ml_text(main.TextAnalyzeRequest(text=text))

        for payload, result_key in ((ensemble, "ensemble"), (urlml, "url_ml")):
            assert payload["count"] == 11, payload
            assert payload["unique_count"] == 5, payload
            assert payload["deduplicated"] is True, payload
            assert [item["index"] for item in payload["results"]] == list(range(11)), payload
            assert payload["results"][1]["duplicate_of"] == 0, payload
            assert payload["results"][3]["duplicate_of"] == 2, payload
            assert payload["results"][4]["duplicate_of"] is None, payload
            assert payload["results"][4]["url"].startswith("https://redirector.test/click?target="), payload
            assert payload["results"][5]["duplicate_of"] == 2, payload
            assert payload["results"][6]["duplicate_of"] is None, payload
            assert payload["results"][6]["url"].startswith("https://redirector.test/click?url="), payload
            assert payload["results"][7]["duplicate_of"] == 2, payload
            assert payload["results"][8]["duplicate_of"] is None, payload
            assert payload["results"][8]["url"] == "https://evil-checkout.test/pay_(1)", payload
            assert payload["results"][9]["duplicate_of"] == 2, payload
            assert payload["results"][10]["duplicate_of"] == 0, payload
            assert result_key in payload["results"][0]["result"], payload

        empty = await main.analyze_url_ml_text(main.TextAnalyzeRequest(text="no URL here"))
        assert empty["count"] == 0
        assert empty["results"] == []
    finally:
        main.analyze_url_batch = original_batch
        main.analyze_url_ml_batch = original_urlml_batch


def main_check() -> int:
    asyncio.run(_run_endpoint_checks())
    print("text API contract selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_check())
