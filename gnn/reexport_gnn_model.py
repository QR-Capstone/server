#!/usr/bin/env python3
"""
Re-export gnn_model.pkl for a target Python (e.g. 3.12 server).

Default: compress=0, protocol=4 — avoids zlib layers and 3.13-only pickle opcodes
when the destination is Python 3.12.

Run where the INPUT file loads (e.g. training Python 3.13). Deploy OUTPUT to server 3.12.
"""
from __future__ import annotations

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "gnn_model.pkl")
DEFAULT_FEATURES_PATH = os.path.join(BASE_DIR, "gnn_model_features.pkl")


def _dump(obj, path: str, compress: int) -> None:
    import joblib

    try:
        joblib.dump(obj, path, compress=compress, protocol=4)
    except TypeError:
        try:
            joblib.dump(obj, path, compress=compress)
        except TypeError:
            joblib.dump(obj, path)


def main() -> int:
    p = argparse.ArgumentParser(description="Re-export GNN RF pickle for portability.")
    p.add_argument(
        "--model-in",
        default=DEFAULT_MODEL_PATH,
        help="Existing model path (must load on this Python)",
    )
    p.add_argument(
        "--model-out",
        default=None,
        help="Output path (default: overwrite --model-in)",
    )
    p.add_argument(
        "--features-in",
        default=DEFAULT_FEATURES_PATH,
        help="Feature column list pickle",
    )
    p.add_argument(
        "--features-out",
        default=None,
        help="Output features path (default: overwrite --features-in)",
    )
    p.add_argument(
        "--compress",
        type=int,
        default=0,
        help="joblib compress level (0=none, safer for 3.12; default 0)",
    )
    args = p.parse_args()
    model_out = args.model_out or args.model_in
    feat_out = args.features_out or args.features_in

    import joblib

    if not os.path.isfile(args.model_in):
        print(f"error: not found: {args.model_in}", file=sys.stderr)
        return 1

    print(f"Python {sys.version.split()[0]} — loading {args.model_in} ...")
    model = joblib.load(args.model_in)
    print(
        f"Writing {model_out} (compress={args.compress}, protocol=4) ..."
    )
    _dump(model, model_out, args.compress)

    if os.path.isfile(args.features_in):
        cols = joblib.load(args.features_in)
        print(f"Writing {feat_out} ...")
        _dump(cols, feat_out, args.compress)
    else:
        print(f"(skip) {args.features_in} not found")

    print("Done. Deploy these files to the server and restart the API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
