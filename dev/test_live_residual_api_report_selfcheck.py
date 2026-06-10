#!/usr/bin/env python3
"""Validate the persisted live residual API quality report."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = BASE / "dev" / "model_quality" / "live_residual_api_quality.json"
EXPECTED_URLS = {
    "https://blog44investment.shop",
    "https://confirm-payment-id694787.com",
    "https://y-y-z-y-w-101.top",
    "https://y-y-z-y-w-1.top",
}


def valid_payload() -> dict:
    rows = [{"url": url, "risk": "malicious"} for url in sorted(EXPECTED_URLS)]
    return {
        "ok": True,
        "failures": [],
        "urls": sorted(EXPECTED_URLS),
        "results": {
            "urlml": {"single": list(rows), "batch": list(rows)},
            "ensemble": {"single": list(rows), "batch": list(rows)},
        },
    }


def validate_report(path: Path = DEFAULT_REPORT) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("ok") is True, payload
    assert payload.get("failures") == [], payload
    assert set(payload.get("urls") or []) == EXPECTED_URLS, payload
    results = payload.get("results")
    assert isinstance(results, dict), payload
    assert set(results) == {"urlml", "ensemble"}, results
    for engine, result in results.items():
        assert isinstance(result, dict), (engine, result)
        for mode in ("single", "batch"):
            rows = result.get(mode)
            assert isinstance(rows, list), (engine, mode, result)
            assert {row.get("url") for row in rows if isinstance(row, dict)} == EXPECTED_URLS, rows
            assert all(row.get("risk") == "malicious" for row in rows if isinstance(row, dict)), rows


def assert_rejects(payload: dict, tmpdir: Path) -> None:
    report = tmpdir / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_report(report)
    except AssertionError:
        return
    raise AssertionError(f"invalid report accepted: {payload}")


def validate_fail_closed_cases() -> None:
    with tempfile.TemporaryDirectory(prefix="live_residual_api_report_selfcheck_") as tmp:
        tmpdir = Path(tmp)

        payload = valid_payload()
        payload["ok"] = False
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["failures"] = ["urlml.single.example=unknown != malicious"]
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["urls"] = sorted(EXPECTED_URLS - {"https://confirm-payment-id694787.com"})
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["results"]["urlml"]["batch"][0]["risk"] = "unknown"
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        del payload["results"]["ensemble"]
        assert_rejects(payload, tmpdir)


def main() -> int:
    validate_report()
    validate_fail_closed_cases()
    print("live residual API report selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
