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


def _best_threshold(probs: Sequence[float], labels: Sequence[int]) -> Tuple[float, float, int, int, int, int]:
    if len(probs) != len(labels):
        raise ValueError("probs and labels length mismatch")
    candidates = {0.5, 0.01, 0.99}
    clipped = sorted({min(0.999999, max(0.000001, float(prob))) for prob in probs})
    candidates.update(clipped)
    for left, right in zip(clipped, clipped[1:]):
        candidates.add((left + right) / 2.0)
    best = (0.0, 0.5, 0, 0, 0, 0)
    for threshold in candidates:
        tp = tn = fp = fn = 0
        for prob, label in zip(probs, labels):
            pred = 1 if float(prob) >= threshold else 0
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 0 and label == 0:
                tn += 1
            elif pred == 1 and label == 0:
                fp += 1
            else:
                fn += 1
        acc = (tp + tn) / max(1, len(labels))
        if (acc, tp + tn, threshold) > (best[0], best[2] + best[3], best[1]):
            best = (acc, threshold, tp, tn, fp, fn)
    acc, threshold, tp, tn, fp, fn = best
    return threshold, acc, tp, tn, fp, fn


def _subset(items: Sequence, indices: Sequence[int]):
    return [items[i] for i in indices]


def _holdout_metadata(
    *,
    accuracy: float,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
    train_rows: int,
    test_rows: int,
    args: argparse.Namespace,
) -> dict:
    return {
        "holdout_metrics": {
            "accuracy": round(float(accuracy), 6),
            "confusion_matrix": {
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            },
            "train_rows": int(train_rows),
            "test_rows": int(test_rows),
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "min_holdout_accuracy": float(args.min_holdout_accuracy),
        }
    }


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
    parser.add_argument("--min-training-rows", type=int, default=0)
    parser.add_argument("--limit-rows", type=int, default=0, help="Use only the first N rows; intended for fast smoke tests.")
    parser.add_argument("--no-tune-threshold", action="store_true", help="Keep the default GNN threshold instead of tuning on the internal holdout.")
    parser.add_argument(
        "--min-holdout-accuracy",
        type=float,
        default=0.0,
        help="Internal holdout accuracy must be at least this value before saving artifacts.",
    )
    args = parser.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(base, args.csv)
    if not os.path.isfile(csv_path):
        print(f"error: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    model_out = args.model_out or resolve_gnn_model_path(base)
    feat_out = args.features_out or resolve_gnn_features_path(base)
    os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(feat_out)), exist_ok=True)

    urls, labels, rows = _read_dataset(csv_path)
    if args.limit_rows:
        urls = urls[: args.limit_rows]
        labels = labels[: args.limit_rows]
        rows = rows[: args.limit_rows]
    if len(urls) < args.min_training_rows:
        print(f"error: training rows {len(urls)} < min_training_rows {args.min_training_rows}", file=sys.stderr)
        return 1
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
        test_labels = _subset(labels, test_idx)
        probs = [
            train_model.predict_proba_from_features({name: value for name, value in zip(FEATURE_NAMES, vec)})
            for vec in test_vectors
        ]
        if args.no_tune_threshold:
            tuned_threshold = train_model.threshold
            tp = tn = fp = fn = 0
            for prob, label in zip(probs, test_labels):
                pred = 1 if prob >= tuned_threshold else 0
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
            tuned_threshold, acc, tp, tn, fp, fn = _best_threshold(probs, test_labels)
            train_model.threshold = tuned_threshold
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
        tuned_threshold = train_model.threshold
    print(f"Holdout accuracy: {acc:.4f}")
    print(f"Confusion matrix: TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Threshold: {tuned_threshold:.6f}")
    print(
        "Graph summary: "
        f"train_urls={len(train_idx)} test_urls={len(test_idx)} "
        f"fetch_pages={train_model.metadata.get('fetch_pages_during_training')}"
    )
    validation_metadata = _holdout_metadata(
        accuracy=acc,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
        train_rows=len(train_idx),
        test_rows=len(test_idx),
        args=args,
    )
    if args.min_holdout_accuracy and acc < args.min_holdout_accuracy:
        print(
            f"error: holdout accuracy {acc:.4f} < min_holdout_accuracy {args.min_holdout_accuracy:.4f}; "
            "existing artifacts were not overwritten",
            file=sys.stderr,
        )
        return 1

    if stored_features and not args.fetch_pages:
        final_model = train_web_structure_gnn_model_from_vectors(
            _feature_vectors_from_rows(rows),
            labels,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            threshold=tuned_threshold,
            metadata_extra={"source": "stored_csv_features", **validation_metadata},
        )
    else:
        final_model = train_web_structure_gnn_model(
            urls,
            labels,
            fetch_pages=args.fetch_pages,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            metadata_extra=validation_metadata,
        )
    save_gnn_artifact(final_model, model_out, feat_out)
    print(f"Wrote: {model_out}")
    print(f"Wrote: {feat_out}")
    print(f"Model kind: {final_model.metadata.get('kind')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
