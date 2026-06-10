#!/usr/bin/env python3
"""Self-check expanded training builder conflict and reserve handling."""

from __future__ import annotations

import csv
import os
import tempfile

import build_expanded_training


def _write(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="build_expanded_training_") as tmp:
        input_a = os.path.join(tmp, "a.csv")
        input_b = os.path.join(tmp, "b.csv")
        reserve = os.path.join(tmp, "reserve.csv")
        out = os.path.join(tmp, "out.csv")
        _write(
            input_a,
            [
                {"url": "https://reserved.test/login", "label": "1", "source": "mal_feed"},
                {"url": "https://conflict.test/login", "label": "1", "source": "mal_feed"},
                {"url": "https://safe.test/", "label": "0", "source": "benign_feed"},
            ],
        )
        _write(
            input_b,
            [
                {"url": "https://conflict.test/login", "label": "0", "source": "benign_feed"},
                {"url": "https://newbad.test/pay", "label": "1", "source": "mal_feed"},
            ],
        )
        _write(reserve, [{"url": "https://reserved.test/login", "label": "1", "source": "eval"}])

        rel = lambda path: os.path.relpath(path, build_expanded_training.BASE)  # noqa: E731
        assert (
            build_expanded_training.main(
                [
                    "--input",
                    rel(input_a),
                    "--input",
                    rel(input_b),
                    "--reserve",
                    rel(reserve),
                    "--out",
                    rel(out),
                ]
            )
            == 1
        )

        assert (
            build_expanded_training.main(
                [
                    "--input",
                    rel(input_a),
                    "--input",
                    rel(input_b),
                    "--reserve",
                    rel(reserve),
                    "--out",
                    rel(out),
                    "--allow-label-conflicts",
                ]
            )
            == 0
        )
        rows = _read_rows(out)
        urls = {row["url"] for row in rows}
        assert "https://reserved.test/login" not in urls
        assert "https://safe.test/" in urls
        assert "https://newbad.test/pay" in urls

    print("build expanded training selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
