#!/usr/bin/env python3
"""Train the fast URL lexical ML model."""

from __future__ import annotations

import argparse
import csv
import os
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(BASE_DIR, "url_ml_model.joblib")


def read_csvs(
    paths: Iterable[str],
    user_confirmed_weight: float,
    naver_benign_weight: float,
) -> tuple[list[str], list[int], list[float]]:
    urls: list[str] = []
    labels: list[int] = []
    weights: list[float] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label", "")).strip()
                if not url or label not in {"0", "1"} or url in seen:
                    continue
                seen.add(url)
                urls.append(url)
                labels.append(int(label))
                source = (row.get("source") or "").strip().lower()
                if source.startswith("user_normal_") or source.startswith("manual_eval_"):
                    weights.append(user_confirmed_weight)
                elif source.startswith("naver_search:") and label == "0":
                    weights.append(naver_benign_weight)
                else:
                    weights.append(1.0)
    return urls, labels, weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="CSV with url,label. Repeatable.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--user-confirmed-weight",
        type=float,
        default=5.0,
        help="Training weight for manually confirmed normal samples.",
    )
    parser.add_argument(
        "--naver-benign-weight",
        type=float,
        default=3.0,
        help="Training weight for live benign samples collected from Naver search.",
    )
    args = parser.parse_args()

    urls, labels, weights = read_csvs(args.input, args.user_confirmed_weight, args.naver_benign_weight)
    if len(set(labels)) < 2:
        raise SystemExit("need both benign and malicious labels")
    print(f"loaded={len(urls)} malicious={sum(labels)} benign={len(labels)-sum(labels)}")

    x_train, x_test, y_train, y_test, w_train, _w_test = train_test_split(
        urls,
        labels,
        weights,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    lowercase=True,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=4.0,
                    class_weight="balanced",
                    max_iter=2000,
                    solver="liblinear",
                    random_state=args.random_state,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train, clf__sample_weight=w_train)
    pred = model.predict(x_test)
    print(classification_report(y_test, pred, target_names=["benign", "malicious"], digits=4))
    cm = confusion_matrix(y_test, pred)
    print(f"confusion_matrix={cm.tolist()}")

    # The split above is only for reporting. Persist a model trained on the full
    # curated dataset so newly confirmed benign/malicious URLs are not randomly
    # left out of the production artifact.
    model.fit(urls, labels, clf__sample_weight=weights)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    joblib.dump(model, args.out)
    print(f"wrote={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
