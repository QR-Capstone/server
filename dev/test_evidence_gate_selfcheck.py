#!/usr/bin/env python3
"""Self-checks for evidence gate invariants that should fail closed."""

from __future__ import annotations

import contextlib
import io

from run_evidence_gates import parse_min_source_counts, validate_combined_counts
from validate_evidence_report import validate_report


def wilson_lower_bound(successes: int, total: int, z: float = 1.6448536269514722) -> float:
    if total <= 0:
        return 0.0
    p_hat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = p_hat + z2 / (2.0 * total)
    margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * total)) / total) ** 0.5
    return max(0.0, (center - margin) / denominator)


def expect_system_exit(func, expected: str) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            func()
    except SystemExit as exc:
        message = str(exc)
        if expected not in message:
            raise AssertionError(f"expected {expected!r} in {message!r}") from exc
        return
    raise AssertionError(f"expected SystemExit containing {expected!r}")


def main() -> int:
    threshold = 0.9999
    assert wilson_lower_bound(91000, 91000) >= threshold
    assert wilson_lower_bound(90996, 91000) >= threshold
    assert wilson_lower_bound(90995, 91000) < threshold
    assert wilson_lower_bound(45500, 45500) >= threshold
    assert wilson_lower_bound(45499, 45500) >= threshold
    assert wilson_lower_bound(45498, 45500) < threshold

    parsed = parse_min_source_counts(["tranco_latest_nonoverlap=45500", "openphish=295"])
    assert parsed == {"tranco_latest_nonoverlap": 45500, "openphish": 295}

    with contextlib.redirect_stdout(io.StringIO()):
        validate_combined_counts(
            91000,
            {"tranco_latest_nonoverlap": 45500, "openphish": 295},
            91000,
            {"tranco_latest_nonoverlap": 45500, "openphish": 295},
        )

    expect_system_exit(
        lambda: validate_combined_counts(90999, {"tranco_latest_nonoverlap": 45500}, 91000, {}),
        "combined holdout too small",
    )
    expect_system_exit(
        lambda: validate_combined_counts(
            91000,
            {"tranco_latest_nonoverlap": 45499},
            91000,
            {"tranco_latest_nonoverlap": 45500},
        ),
        "too small",
    )
    expect_system_exit(lambda: parse_min_source_counts(["bad-spec"]), "invalid --min-source-count")

    valid_report = {
        "ok": True,
        "rows": 91000,
        "results": [
            {
                "engine": engine,
                "rows": 91000,
                "accuracy": 1.0,
                "coverage": 1.0,
                "decisive_accuracy": 1.0,
                "accuracy_lower_95": 0.9999702696371723,
                "recall_lower_95": 0.9999405410420813,
                "specificity_lower_95": 0.9999405410420813,
                "false_positive": 0,
                "false_negative": 0,
                "unknown": 0,
                "errors": 0,
                "source_stats": {
                    "tranco_latest_nonoverlap": {"rows": 45500, "fp": 0, "fn": 0, "unknown": 0, "errors": 0},
                },
            }
            for engine in ("urlml", "ensemble")
        ],
    }
    validate_report(
        valid_report,
        91000,
        0.9999,
        0.9999,
        {"tranco_latest_nonoverlap": 45500},
    )
    invalid_report = dict(valid_report)
    invalid_report["results"] = [dict(valid_report["results"][0]), dict(valid_report["results"][1])]
    invalid_report["results"][1]["false_positive"] = 1
    expect_system_exit(
        lambda: validate_report(
            invalid_report,
            91000,
            0.9999,
            0.9999,
            {"tranco_latest_nonoverlap": 45500},
        ),
        "ensemble.false_positive != 0",
    )
    print("PASS evidence gate selfcheck")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
