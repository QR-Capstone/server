#!/usr/bin/env python3
"""Self-check source and label balancing for live API quality samples."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEV = BASE / "dev"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from run_live_api_quality_gate import build_quality_sample, sample_balance_failures  # noqa: E402


def _write_probe(path: Path) -> None:
    rows: list[dict[str, str]] = []
    for source in ("feed_a", "feed_b"):
        for i in range(3):
            rows.append({"url": f"https://{source}-mal-{i}.test/path", "label": "1", "source": source})
    for i in range(8):
        rows.append({"url": f"https://benign-{i}.test/", "label": "0", "source": "tranco_latest_nonoverlap"})
    rows.append({"url": "https://feed-a-mal-0.test/path", "label": "1", "source": "feed_a"})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="live_api_quality_sample_selfcheck_") as tmp:
        probe = Path(tmp) / "probe.csv"
        _write_probe(probe)
        rows = build_quality_sample([probe], per_source_label=3)

    labels = [row["label"] for row in rows]
    keys = [row["canonical_key"] for row in rows]
    assert labels.count("1") == 6, labels
    assert labels.count("0") == 6, labels
    assert len(keys) == len(set(keys)), "duplicate canonical keys in sample"
    assert sample_balance_failures(rows) == []
    assert "sample labels are imbalanced" in sample_balance_failures(rows[:-1])[0]
    duplicate_rows = [*rows, dict(rows[0])]
    assert any("duplicate canonical keys" in failure for failure in sample_balance_failures(duplicate_rows))
    sources = {row["source"] for row in rows}
    assert {"feed_a", "feed_b", "tranco_latest_nonoverlap"}.issubset(sources), sources
    print("live API quality sample selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
