#!/usr/bin/env python3
"""
Train the real Torch GraphSAGE GNN lane from gnn_total_dataset.csv.

The output keeps the same filenames expected by main.py:

  gnn_model.pkl
  gnn_model_features.pkl
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from typing import List, Sequence, Tuple

from gnn_engine import (
    FEATURE_NAMES,
    resolve_gnn_features_path,
    resolve_gnn_model_path,
    save_gnn_artifact,
    train_web_structure_gnn_model_from_vectors,
    train_web_structure_gnn_model,
)


def _read_dataset(path: str) -> Tuple[List[str], List[int], List[dict]]:
    urls: List[str] = []
    labels: List[int] = []
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "url" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError("CSV needs columns: url, label")
        for row in reader:
            url = (row.get("url") or "").strip()
            label_raw = (row.get("label") or "").strip()
            if not url or label_raw not in {"0", "1"}:
                continue
            urls.append(url)
            labels.append(int(label_raw))
            rows.append(dict(row))
    if not urls:
        raise ValueError("no usable rows in CSV")
    return urls, labels, rows


def _has_stored_features(rows: Sequence[dict]) -> bool:
    return bool(rows) and all(name in rows[0] for name in FEATURE_NAMES)


def _feature_vectors_from_rows(rows: Sequence[dict]) -> List[List[float]]:
    vectors: List[List[float]] = []
    for row in rows:
        vector: List[float] = []
        for name in FEATURE_NAMES:
            raw = row.get(name, "")
            if raw in ("", None):
                raise ValueError(f"missing feature {name!r} for url={row.get('url')!r}")
            vector.append(float(raw))
        vectors.append(vector)
    return vectors


def _split_indices(labels: Sequence[int], test_size: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for i, label in enumerate(labels):
        by_label[int(label)].append(i)
    train: List[int] = []
    test: List[int] = []
    for bucket in by_label.values():
        rng.shuffle(bucket)
        n_test = max(1, int(round(len(bucket) * test_size))) if len(bucket) > 1 else 0
        test.extend(bucket[:n_test])
        train.extend(bucket[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _accuracy(
    model,
    urls: Sequence[str],
    labels: Sequence[int],
    fetch_pages: bool,
) -> Tuple[float, int, int, int, int]:
    tp = tn = fp = fn = 0
    for url, label in zip(urls, labels):
        prob = model.predict_proba_one(url, fetch=fetch_pages)
        pred = 1 if prob >= model.threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    total = max(1, len(labels))
    return (tp + tn) / total, tp, tn, fp, fn


def _subset(items: Sequence, indices: Sequence[int]):
    return [items[i] for i in indices]


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate web-structure GNN phishing model.")
    parser.add_argument("--csv", default="gnn_total_dataset.csv", help="CSV with columns url,label")
    parser.add_argument("--model-out", default=None, help="Override model output path")
    parser.add_argument("--features-out", default=None, help="Override features list output path")
    parser.add_argument(
        "--fetch-pages",
        action="store_true",
        help="Fetch live HTML during training to learn page-structure graph features.",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base, args.csv)
    if not os.path.isfile(csv_path):
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    model_out = args.model_out or resolve_gnn_model_path(base)
    feat_out = args.features_out or resolve_gnn_features_path(base)

    urls, labels, rows = _read_dataset(csv_path)
    train_idx, test_idx = _split_indices(labels, args.test_size, args.random_state)

    stored_features = _has_stored_features(rows)
    if stored_features and not args.fetch_pages:
        vectors = _feature_vectors_from_rows(rows)
        train_model = train_web_structure_gnn_model_from_vectors(
            _subset(vectors, train_idx),
            _subset(labels, train_idx),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            metadata_extra={"source": "stored_csv_features"},
        )
        test_vectors = _subset(vectors, test_idx)
        tp = tn = fp = fn = 0
        for vec, label in zip(test_vectors, _subset(labels, test_idx)):
            prob = train_model.predict_proba_from_features(
                {name: value for name, value in zip(FEATURE_NAMES, vec)}
            )
            pred = 1 if prob >= train_model.threshold else 0
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 0 and label == 0:
                tn += 1
            elif pred == 1 and label == 0:
                fp += 1
            else:
                fn += 1
        acc = (tp + tn) / max(1, len(test_vectors))
    else:
        train_model = train_web_structure_gnn_model(
            _subset(urls, train_idx),
            _subset(labels, train_idx),
            fetch_pages=args.fetch_pages,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        acc, tp, tn, fp, fn = _accuracy(
            train_model,
            _subset(urls, test_idx),
            _subset(labels, test_idx),
            args.fetch_pages,
        )
    print(f"Holdout accuracy: {acc:.4f}")
    print(f"Confusion matrix: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(
        "Graph summary: "
        f"train_urls={len(train_idx)} test_urls={len(test_idx)} "
        f"fetch_pages={train_model.metadata.get('fetch_pages_during_training')}"
    )

    if stored_features and not args.fetch_pages:
        final_model = train_web_structure_gnn_model_from_vectors(
            _feature_vectors_from_rows(rows),
            labels,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            metadata_extra={"source": "stored_csv_features"},
        )
    else:
        final_model = train_web_structure_gnn_model(
            urls,
            labels,
            fetch_pages=args.fetch_pages,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
    save_gnn_artifact(final_model, model_out, feat_out)
    print(f"Wrote: {model_out}")
    print(f"Wrote: {feat_out}")
    print(f"Model kind: {final_model.metadata.get('kind')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
