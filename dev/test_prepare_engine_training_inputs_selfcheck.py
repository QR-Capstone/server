#!/usr/bin/env python3
"""Self-check per-engine training input preparation conflict handling."""

from __future__ import annotations

import csv
import os
import tempfile

import prepare_engine_training_inputs as prep


def _write(path: str, rows: list[dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "canonical_key"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="prepare_engine_inputs_") as tmp:
        warehouse = os.path.join(tmp, "warehouse.csv")
        eval_path = os.path.join(tmp, "eval.csv")
        out_dir = os.path.join(tmp, "out")
        gnn_existing = os.path.join(tmp, "gnn.csv")
        _write(
            warehouse,
            [
                {"url": "https://reserved.test/path", "label": "1", "source": "mal"},
                {"url": "https://conflict.test/login", "label": "1", "source": "mal"},
                {"url": "https://conflict.test/login", "label": "0", "source": "benign"},
                {"url": "https://safe.test/", "label": "0", "source": "benign"},
                {"url": "https://bad.test/pay", "label": "1", "source": "mal"},
            ],
        )
        _write(eval_path, [{"url": "https://reserved.test/path", "label": "1", "source": "eval"}])
        _write(gnn_existing, [])

        assert (
            prep.main(
                [
                    "--warehouse",
                    warehouse,
                    "--exclude",
                    eval_path,
                    "--out-dir",
                    out_dir,
                    "--gnn-existing",
                    gnn_existing,
                    "--target",
                    "2",
                ]
            )
            == 1
        )
        assert (
            prep.main(
                [
                    "--warehouse",
                    warehouse,
                    "--exclude",
                    eval_path,
                    "--out-dir",
                    out_dir,
                    "--gnn-existing",
                    gnn_existing,
                    "--target",
                    "2",
                    "--allow-label-conflicts",
                ]
            )
            == 0
        )
        with open(os.path.join(out_dir, "xgboost_train_10k.csv"), "r", encoding="utf-8-sig", newline="") as f:
            urls = {row["url"] for row in csv.DictReader(f)}
        assert "https://reserved.test/path" not in urls
        assert urls <= {"https://safe.test/", "https://bad.test/pay", "https://conflict.test/login"}

    print("prepare engine training inputs selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
