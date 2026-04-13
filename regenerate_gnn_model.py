#!/usr/bin/env python3
"""
Train lexical RF from CSV and save gnn_model.pkl + gnn_model_features.pkl with
protocol=4 so Python 3.12 can load them (avoids 3.13+ pickle opcode mismatch).

Uses gnn_engine.extract_lexical_features — must match inference.
"""
from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from gnn_engine import extract_lexical_features, resolve_gnn_features_path, resolve_gnn_model_path


def _dump_joblib(obj: object, path: str) -> None:
    try:
        joblib.dump(obj, path, compress=0, protocol=4)
    except TypeError:
        try:
            joblib.dump(obj, path, protocol=4)
        except TypeError:
            joblib.dump(obj, path)


def main() -> int:
    p = argparse.ArgumentParser(description="Regenerate GNN lexical RF pickles (Py3.12-safe).")
    p.add_argument(
        "--csv",
        default="gnn_total_dataset.csv",
        help="CSV with columns url,label",
    )
    p.add_argument("--model-out", default=None, help="Override model output path")
    p.add_argument("--features-out", default=None, help="Override features list output path")
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base, args.csv)
    if not os.path.isfile(csv_path):
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    model_out = args.model_out or resolve_gnn_model_path(base)
    feat_out = args.features_out or resolve_gnn_features_path(base)

    df = pd.read_csv(csv_path)
    if "url" not in df.columns or "label" not in df.columns:
        print("error: CSV needs columns: url, label", file=sys.stderr)
        return 1

    rows = [extract_lexical_features(str(u)) for u in df["url"].astype(str)]
    X = pd.DataFrame(rows)
    y = df["label"].astype(int)
    columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, random_state=args.random_state
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    _dump_joblib(model, model_out)
    _dump_joblib(columns, feat_out)
    print(f"Wrote: {model_out}")
    print(f"Wrote: {feat_out}")
    print(f"Python {sys.version.split()[0]} — protocol=4, features={columns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
