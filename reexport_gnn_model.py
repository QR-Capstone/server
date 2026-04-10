#!/usr/bin/env python3
"""
Re-export gnn_model.pkl (and optional feature list) with compress=3 and pickle protocol 4.

Run this on the SAME machine / Python where the current pickle loads successfully
(e.g. your training laptop). Copy the output files to the server.

If unpickling still fails on the server, align Python versions (3.11+ recommended)
and sklearn/joblib versions with `pip install -r requirements.txt`.
"""
from __future__ import annotations

import argparse
import os
import sys


def _dump(obj, path: str) -> None:
    import joblib

    try:
        joblib.dump(obj, path, compress=3, protocol=4)
    except TypeError:
        joblib.dump(obj, path, compress=3)


def main() -> int:
    p = argparse.ArgumentParser(description="Re-export GNN RF pickle for portability.")
    p.add_argument(
        "--model-in",
        default="gnn_model.pkl",
        help="Existing model path (must load on this Python)",
    )
    p.add_argument(
        "--model-out",
        default=None,
        help="Output path (default: overwrite --model-in)",
    )
    p.add_argument(
        "--features-in",
        default="gnn_model_features.pkl",
        help="Feature column list pickle",
    )
    p.add_argument(
        "--features-out",
        default=None,
        help="Output features path (default: overwrite --features-in)",
    )
    args = p.parse_args()
    model_out = args.model_out or args.model_in
    feat_out = args.features_out or args.features_in

    import joblib

    if not os.path.isfile(args.model_in):
        print(f"error: not found: {args.model_in}", file=sys.stderr)
        return 1

    print(f"Loading {args.model_in} ...")
    model = joblib.load(args.model_in)
    print(f"Writing {model_out} (compress=3, protocol=4) ...")
    _dump(model, model_out)

    if os.path.isfile(args.features_in):
        cols = joblib.load(args.features_in)
        print(f"Writing {feat_out} ...")
        _dump(cols, feat_out)
    else:
        print(f"(skip) {args.features_in} not found")

    print("Done. Deploy these files to the server and restart the API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
