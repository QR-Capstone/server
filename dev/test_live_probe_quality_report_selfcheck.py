#!/usr/bin/env python3
"""Validate the persisted live probe quality report schema and residual misses."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = BASE / "dev" / "model_quality" / "live_probe_quality.json"
EXPECTED_ALLOWED_MISSES = {
    "fp|tranco_latest_nonoverlap|https://blog44investment.shop",
    "fp|tranco_latest_nonoverlap|https://y-y-z-y-w-101.top",
    "fp|tranco_latest_nonoverlap|https://y-y-z-y-w-1.top",
    "fp|tranco_latest_nonoverlap|https://confirm-payment-id694787.com",
}


def valid_payload() -> dict:
    return {
        "allowed_residual_misses": sorted(EXPECTED_ALLOWED_MISSES),
        "unexpected_residual_misses": [],
        "overlap": [],
        "label_conflicts": [],
        "misses": [
            {"kind": "fp", "source": "tranco_latest_nonoverlap", "url": "https://blog44investment.shop"},
            {"kind": "fp", "source": "tranco_latest_nonoverlap", "url": "https://y-y-z-y-w-101.top"},
            {"kind": "fp", "source": "tranco_latest_nonoverlap", "url": "https://confirm-payment-id694787.com"},
        ],
        "stats": {
            "rows": 7000,
            "malicious": 3500,
            "benign": 3500,
            "tp": 3500,
            "tn": 3497,
            "fp": 3,
            "fn": 0,
            "unknown": 0,
            "malicious_unknown": 0,
            "benign_unknown": 0,
            "accuracy_ppm": 999571,
            "coverage_ppm": 1000000,
        },
        "by_source": {
            "tranco_latest_nonoverlap": {
                "rows": 3500,
                "benign": 3500,
                "malicious": 0,
                "fp": 3,
                "fn": 0,
                "unknown": 0,
                "benign_unknown": 0,
                "malicious_unknown": 0,
            },
            "urlhaus_recent": {"rows": 1500, "malicious": 1500, "fn": 0, "malicious_unknown": 0},
            "phishing_database_active": {"rows": 1000, "malicious": 1000, "fn": 0, "malicious_unknown": 0},
            "nurilab": {"rows": 890, "malicious": 890, "fn": 0, "malicious_unknown": 0},
            "phishtank": {"rows": 110, "malicious": 110, "fn": 0, "malicious_unknown": 0},
        },
    }


def _miss_key(miss: dict) -> str:
    return f"{miss.get('kind')}|{miss.get('source')}|{miss.get('url')}"


def validate_report(path: Path = DEFAULT_REPORT) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload.get("stats")
    assert isinstance(stats, dict), payload
    assert int(stats.get("rows") or 0) >= 7000, stats
    assert int(stats.get("malicious") or 0) >= 3500, stats
    assert int(stats.get("benign") or 0) >= 3500, stats
    assert int(stats.get("fn") or 0) == 0, stats
    assert int(stats.get("malicious_unknown") or 0) == 0, stats
    assert int(stats.get("fp") or 0) <= 4, stats
    assert int(stats.get("benign_unknown") or 0) == 0, stats
    # Floor matches the 4 allowed residual misses on 7000 rows.
    assert int(stats.get("accuracy_ppm") or 0) >= 999400, stats
    assert payload.get("overlap") == [], payload
    assert payload.get("label_conflicts") == [], payload
    assert payload.get("unexpected_residual_misses") == [], payload
    assert set(payload.get("allowed_residual_misses") or []) == EXPECTED_ALLOWED_MISSES, payload

    miss_keys = {_miss_key(miss) for miss in payload.get("misses") or [] if isinstance(miss, dict)}
    assert miss_keys.issubset(EXPECTED_ALLOWED_MISSES), miss_keys
    by_source = payload.get("by_source")
    assert isinstance(by_source, dict), payload
    assert int(by_source["tranco_latest_nonoverlap"]["rows"]) >= 3500, by_source
    for source in ("urlhaus_recent", "phishing_database_active", "nurilab", "phishtank"):
        assert int(by_source[source]["malicious"]) > 0, by_source
        assert int(by_source[source]["fn"]) == 0, by_source
        assert int(by_source[source]["malicious_unknown"]) == 0, by_source


def assert_rejects(payload: dict, tmpdir: Path) -> None:
    report = tmpdir / "report.json"
    report.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_report(report)
    except AssertionError:
        return
    raise AssertionError(f"invalid report accepted: {payload}")


def validate_fail_closed_cases() -> None:
    with tempfile.TemporaryDirectory(prefix="live_probe_quality_report_selfcheck_") as tmp:
        tmpdir = Path(tmp)

        payload = valid_payload()
        payload["stats"]["fn"] = 1
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["unexpected_residual_misses"] = ["fp|tranco_latest_nonoverlap|https://new-residual.test"]
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["misses"].append(
            {"kind": "fp", "source": "tranco_latest_nonoverlap", "url": "https://new-residual.test"}
        )
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["overlap"] = ["https://example.test"]
        assert_rejects(payload, tmpdir)

        payload = valid_payload()
        payload["by_source"]["urlhaus_recent"]["malicious_unknown"] = 1
        assert_rejects(payload, tmpdir)


def main() -> int:
    validate_report()
    validate_fail_closed_cases()
    print("live probe quality report selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
