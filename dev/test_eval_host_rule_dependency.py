#!/usr/bin/env python3
"""Ensure URLML quality does not depend on eval-host allow/block lists."""

from __future__ import annotations

import os
import subprocess
import sys


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOLDOUTS = [
    "dev/nonoverlap_feed_eval_40k_20260524.csv",
    "dev/nonoverlap_crossfeed_eval_20260524.csv",
    "dev/nonoverlap_phishingdb_eval_40k_20260524.csv",
    "dev/nonoverlap_korean_sources_eval_20260524.csv",
]


def main() -> int:
    cmd = [sys.executable, "dev/audit_eval_host_rule_dependency.py"]
    for holdout in HOLDOUTS:
        cmd.extend(["--eval", holdout])
    cmd.extend(
        [
            "--max-accuracy-drop",
            "0",
            "--max-new-fp",
            "0",
            "--max-new-fn",
            "0",
        ]
    )
    subprocess.run(cmd, cwd=BASE, check=True)
    print("eval host rule dependency selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
