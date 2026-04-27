"""Compatibility wrapper that routes CLI calls to train or inference entrypoints."""

from __future__ import annotations

import sys
from typing import Optional

from XG_infer import main as infer_main
from XG_train import main as train_main


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "predict-url":
        return int(infer_main(argv[1:]))
    return int(train_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
