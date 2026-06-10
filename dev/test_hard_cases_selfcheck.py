#!/usr/bin/env python3
"""Self-check curated API hard cases."""
from __future__ import annotations

import os
import sys
from collections import Counter


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from dev.evaluate_api_holdout import read_labeled
from trusted_domains import is_trusted_official_url, strong_url_phishing_score


HARD_CASES = os.path.join(BASE, "dev", "model_quality", "hard_cases.csv")


def main() -> int:
    rows = read_labeled([HARD_CASES])
    assert len(rows) == 20, f"expected 20 unique hard cases, got {len(rows)}"
    labels = Counter(row.label for row in rows)
    assert labels[0] == 6 and labels[1] == 14, labels

    for row in rows:
        trusted = is_trusted_official_url(row.url)
        strong = strong_url_phishing_score(row.url)
        if row.label == 0:
            assert trusted, f"benign hard case must be trusted: {row.url}"
            assert strong == 0.0, f"trusted hard case strong score changed: {row.url}={strong}"
        else:
            assert not trusted, f"malicious hard case must not be trusted: {row.url}"
            assert strong >= 0.66, f"malicious hard case strong score too low: {row.url}={strong}"

    print("hard cases selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
