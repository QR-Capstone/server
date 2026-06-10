#!/usr/bin/env python3
"""Validate the persisted live API quality report schema and floors."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = BASE / "dev" / "model_quality" / "live_api_quality.json"


def valid_payload() -> dict:
    source_stats = {
        "tranco_latest_nonoverlap": {"rows": 320, "fn": 0},
        "nurilab": {"rows": 80, "fn": 0},
        "phishing_database_active": {"rows": 80, "fn": 0},
        "phishtank": {"rows": 80, "fn": 0},
        "urlhaus_recent": {"rows": 80, "fn": 0},
    }
    return {
        "ok": True,
        "rows": 640,
        "results": [
            {
                "engine": "urlml",
                "false_negative": 0,
                "recall": 1.0,
                "accuracy": 0.99,
                "unknown_ratio": 0.0,
                "source_stats": source_stats,
            },
            {
                "engine": "ensemble",
                "false_negative": 0,
                "recall": 1.0,
                "accuracy": 0.99,
                "unknown_ratio": 0.0,
                "source_stats": source_stats,
            },
        ],
    }


def validate_report(path: Path = DEFAULT_REPORT) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload.get("ok") is True, payload
    assert int(payload.get("rows") or 0) >= 640, payload
    results = payload.get("results")
    assert isinstance(results, list), payload
    by_engine = {result.get("engine"): result for result in results if isinstance(result, dict)}
    assert set(by_engine) == {"urlml", "ensemble"}, by_engine
    for engine, result in by_engine.items():
        assert int(result.get("false_negative") or 0) == 0, result
        assert float(result.get("recall") or 0.0) == 1.0, result
        assert float(result.get("accuracy") or 0.0) >= 0.99, result
        assert float(result.get("unknown_ratio") or 0.0) <= 0.01, result
        source_stats = result.get("source_stats")
        assert isinstance(source_stats, dict), result
        assert int(source_stats["tranco_latest_nonoverlap"]["rows"]) >= 320, source_stats
        for source in ("nurilab", "phishing_database_active", "phishtank", "urlhaus_recent"):
            assert int(source_stats[source]["rows"]) >= 80, (engine, source_stats)
            assert int(source_stats[source]["fn"]) == 0, (engine, source_stats)


def assert_rejects(payload: dict, tmpdir: Path) -> None:
    report = tmpdir / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_report(report)
    except AssertionError:
        return
    raise AssertionError(f"invalid report accepted: {payload}")


def validate_fail_closed_cases() -> None:
    with tempfile.TemporaryDirectory(prefix="live_api_quality_report_selfcheck_") as tmp:
        tmpdir = Path(tmp)

        payload = valid_payload()
        payload["ok"] = False
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["rows"] = 639
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["results"][0]["false_negative"] = 1
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["results"][0]["recall"] = 0.999
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["results"][0]["source_stats"]["tranco_latest_nonoverlap"]["rows"] = 319
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["results"][0]["source_stats"]["nurilab"]["fn"] = 1
        assert_rejects(payload, tmpdir)


def main() -> int:
    validate_report()
    validate_fail_closed_cases()
    print("live API quality report selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
