#!/usr/bin/env python3
"""Self-check URL split audit behavior."""
from __future__ import annotations

import csv
import os
import tempfile

import audit_url_splits


def _write(path: str, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    assert audit_url_splits.main([]) == 0

    with tempfile.TemporaryDirectory(prefix="url_split_audit_") as tmp:
        train = os.path.join(tmp, "train.csv")
        test = os.path.join(tmp, "test.csv")
        _write(
            train,
            [
                {"url": "https://example.com/login", "label": "0", "source": "unit"},
                {"url": "https://bad.test", "label": "1", "source": "unit"},
            ],
        )
        _write(
            test,
            [
                {"url": "https://example.com/login/", "label": "1", "source": "unit"},
                {"url": "https://safe.test", "label": "0", "source": "unit"},
            ],
        )
        assert (
            audit_url_splits.main(
                [
                    "--train",
                    train,
                    "--test",
                    test,
                    "--min-train-rows",
                    "1",
                    "--min-test-rows",
                    "1",
                    "--balanced-test",
                    "",
                ]
            )
            == 1
        )

    print("url split audit selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
