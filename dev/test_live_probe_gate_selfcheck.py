#!/usr/bin/env python3
"""Self-check fail-closed behavior for live probe quality gates."""

from __future__ import annotations

import argparse
import os
import sys


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE, "dev") not in sys.path:
    sys.path.insert(0, os.path.join(BASE, "dev"))

from run_live_probe_gates import _failures  # noqa: E402


def _args(**overrides):
    defaults = {
        "min_rows": 7000,
        "min_malicious": 3500,
        "min_benign": 3500,
        "max_fp": 3,
        "max_fn": 0,
        "max_malicious_unknown": 0,
        "max_benign_unknown": 0,
        "allowed_residual_miss": [
            "fp|tranco_latest_nonoverlap|https://blog44investment.shop",
            "fp|tranco_latest_nonoverlap|https://y-y-z-y-w-101.top",
            "fp|tranco_latest_nonoverlap|https://confirm-payment-id694787.com",
        ],
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _report(**stats_overrides):
    stats = {
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
    }
    stats.update(stats_overrides)
    return {
        "stats": stats,
        "overlap": [],
        "label_conflicts": [],
        "misses": [
            {"kind": "fp", "source": "tranco_latest_nonoverlap", "url": "https://blog44investment.shop"},
            {"kind": "fp", "source": "tranco_latest_nonoverlap", "url": "https://y-y-z-y-w-101.top"},
            {
                "kind": "fp",
                "source": "tranco_latest_nonoverlap",
                "url": "https://confirm-payment-id694787.com",
            },
        ],
    }


def _contains(failures: list[str], text: str) -> bool:
    return any(text in failure for failure in failures)


def main() -> int:
    assert _failures(_report(), _args()) == []

    assert _contains(_failures(_report(rows=6999), _args()), "rows=6999 < 7000")
    assert _contains(_failures(_report(malicious=3499), _args()), "malicious=3499 < 3500")
    assert _contains(_failures(_report(benign=3499), _args()), "benign=3499 < 3500")
    assert _contains(_failures(_report(fp=4), _args()), "fp=4 > 3")
    assert _contains(_failures(_report(fn=1), _args()), "fn=1 > 0")
    assert _contains(_failures(_report(malicious_unknown=1), _args()), "malicious_unknown=1 > 0")
    assert _contains(_failures(_report(benign_unknown=1), _args()), "benign_unknown=1 > 0")

    overlap_report = _report()
    overlap_report["overlap"] = ["https://example.test"]
    assert _contains(_failures(overlap_report, _args()), "reference_overlap=1")

    conflict_report = _report()
    conflict_report["label_conflicts"] = ["https://conflict.test"]
    assert _contains(_failures(conflict_report, _args()), "label_conflicts=1")

    unexpected_report = _report(fp=2, benign_unknown=0, unknown=0)
    unexpected_report["misses"] = [
        {"kind": "fp", "source": "tranco_latest_nonoverlap", "url": "https://new-residual.test"}
    ]
    assert _contains(_failures(unexpected_report, _args()), "unexpected_residual_misses=1")

    print("live probe gate selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
