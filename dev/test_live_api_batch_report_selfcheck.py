#!/usr/bin/env python3
"""Validate the persisted live API batch consistency report."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = BASE / "dev" / "model_quality" / "live_api_batch_consistency.json"


def valid_payload() -> dict:
    return {
        "ok": True,
        "rows": 401,
        "engines": ["urlml", "ensemble"],
        "mismatches": [],
    }


def validate_report(path: Path = DEFAULT_REPORT) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("ok") is True, payload
    assert int(payload.get("rows") or 0) >= 401, payload
    assert set(payload.get("engines") or []) == {"urlml", "ensemble"}, payload
    assert payload.get("mismatches") == [], payload


def assert_rejects(payload: dict, tmpdir: Path) -> None:
    report = tmpdir / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_report(report)
    except AssertionError:
        return
    raise AssertionError(f"invalid report accepted: {payload}")


def validate_fail_closed_cases() -> None:
    with tempfile.TemporaryDirectory(prefix="live_api_batch_report_selfcheck_") as tmp:
        tmpdir = Path(tmp)

        payload = valid_payload()
        payload["ok"] = False
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["rows"] = 400
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["engines"] = ["urlml"]
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["mismatches"] = ["urlml: single=benign batch=malicious url=https://example.test"]
        assert_rejects(payload, tmpdir)


def main() -> int:
    validate_report()
    validate_fail_closed_cases()
    print("live API batch report selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
