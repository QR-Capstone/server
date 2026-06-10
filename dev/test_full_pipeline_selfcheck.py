#!/usr/bin/env python3
"""Self-check small helpers used by the full training pipeline."""
from __future__ import annotations

import csv
import os
import tempfile

import run_full_pipeline


def _write_csv(path: str, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="full_pipeline_selfcheck_") as tmp:
        empty_path = os.path.join(tmp, "empty.csv")
        _write_csv(empty_path, [])
        assert run_full_pipeline.count_labeled_rows(empty_path) == 0

        labeled_path = os.path.join(tmp, "labeled.csv")
        _write_csv(
            labeled_path,
            [
                {"url": "https://example.com", "label": "0"},
                {"url": "https://phish.test/login", "label": "1"},
                {"url": "", "label": "1"},
                {"url": "https://ignored.test", "label": ""},
            ],
        )
        assert run_full_pipeline.count_labeled_rows(labeled_path) == 2

    print("full pipeline selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
