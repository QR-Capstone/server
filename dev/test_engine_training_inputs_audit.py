#!/usr/bin/env python3
"""Self-check engine training input leakage audit behavior."""
from __future__ import annotations

import csv
import os
import tempfile

import audit_engine_training_inputs as audit


FIELDNAMES = ["url", "label", "text"]


def _write(path: str, rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    assert audit.main([]) == 0

    with tempfile.TemporaryDirectory(prefix="engine_training_audit_") as tmp:
        input_dir = os.path.join(tmp, "inputs")
        os.makedirs(input_dir)
        rows = [
            {"url": "https://example.com/path?a=one|two", "label": "0", "text": "safe example login page"},
            {"url": "https://bad.test/pay", "label": "1", "text": "malicious payment warning"},
        ]
        for filename in ("xgboost_train_10k.csv", "kobert_candidates_10k.csv", "kobert_text_train_10k.csv"):
            _write(os.path.join(input_dir, filename), rows)

        eval_path = os.path.join(tmp, "eval.csv")
        _write(eval_path, [{"url": "https://example.com/path?a=one%7Ctwo", "label": "0", "text": ""}])

        gnn_path = os.path.join(tmp, "gnn.csv")
        _write(gnn_path, rows)

        assert (
            audit.main(
                [
                    "--input-dir",
                    input_dir,
                    "--eval",
                    eval_path,
                    "--target",
                    "2",
                    "--gnn-existing",
                    gnn_path,
                ]
            )
            == 1
        )

    print("engine training inputs audit selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
