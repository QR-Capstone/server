#!/usr/bin/env python3
"""Self-check source-balanced live API batch samples."""

from __future__ import annotations

import csv
import sys
import tempfile
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
DEV = BASE / "dev"
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from run_live_api_batch_gate import build_source_balanced_sample, canonical_url_key  # noqa: E402


def _write_probe(path: Path) -> None:
    rows: list[dict[str, str]] = []
    for source in ("feed_a", "feed_b"):
        for label in ("0", "1"):
            for i in range(4):
                rows.append({"url": f"https://{source}-{label}-{i}.test/path/", "label": label, "source": source})
    rows.extend(
        [
            {"url": "https://feed-a-1-0.test/path", "label": "1", "source": "feed_a"},
            {"url": "https://ignored-invalid-label.test/", "label": "2", "source": "feed_a"},
            {"url": "", "label": "1", "source": "feed_a"},
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    assert canonical_url_key("HTTPS://WWW.Example.TEST/path/") == "https://example.test/path"
    with tempfile.TemporaryDirectory(prefix="live_api_batch_sample_selfcheck_") as tmp:
        probe = Path(tmp) / "probe.csv"
        _write_probe(probe)
        rows = build_source_balanced_sample([probe], per_source_label=2)

    assert len(rows) == 8, rows
    labels = Counter(row["label"] for row in rows)
    assert labels == {"0": 4, "1": 4}, labels
    source_labels = Counter((row["source"], row["label"]) for row in rows)
    assert all(count == 2 for count in source_labels.values()), source_labels
    keys = [row["canonical_key"] for row in rows]
    assert len(keys) == len(set(keys)), keys
    assert all("ignored-invalid-label" not in row["url"] for row in rows), rows
    print("live API batch sample selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
