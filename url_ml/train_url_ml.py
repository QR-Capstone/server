#!/usr/bin/env python3
"""Train the fast URL lexical ML model."""

from __future__ import annotations

import argparse
import csv
import os
import random
from urllib.parse import urlsplit, urlunsplit
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from url_features import URLLexicalFeatures

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(BASE_DIR, "url_ml_model.joblib")


def canonical_url_key(raw_url: str) -> str:
    """Stable key for duplicate/leakage checks across train and holdout files."""
    raw = (raw_url or "").strip()
    if not raw:
        return ""
    candidate = raw if "://" in raw else f"//{raw}"
    try:
        parsed = urlsplit(candidate)
    except Exception:
        return raw.lower().rstrip("/")
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    query = parsed.query
    if scheme:
        return urlunsplit((scheme, netloc, path, query, "")).lower()
    return urlunsplit(("", netloc, path, query, "")).lower()


def load_holdout_keys(paths: Iterable[str]) -> set[str]:
    keys: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = canonical_url_key(row.get("url") or "")
                if key:
                    keys.add(key)
    return keys


def read_csvs(
    paths: Iterable[str],
    user_confirmed_weight: float,
    naver_benign_weight: float,
    hard_benign_weight: float,
    synthetic_weight: float,
    exclude_keys: set[str] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    exclude_keys = exclude_keys or set()
    for path in paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                url = (row.get("url") or "").strip()
                label = str(row.get("label", "")).strip()
                key = canonical_url_key(url)
                if not url or label not in {"0", "1"} or not key or key in seen or key in exclude_keys:
                    continue
                seen.add(key)
                source = (row.get("source") or "").strip().lower()
                if source.startswith("user_normal_") or source.startswith("manual_eval_"):
                    weight = user_confirmed_weight
                elif source.startswith("round100_eval_train_"):
                    weight = user_confirmed_weight
                elif source == "benign_user_confirmed_fp":
                    weight = user_confirmed_weight
                elif source in {"benign_hard_korean_smb", "benign_ad_landing"} and label == "0":
                    weight = hard_benign_weight
                elif source.startswith("naver_search:") and label == "0":
                    weight = naver_benign_weight
                elif source == "malicious_synthetic_kr" and label == "1":
                    weight = synthetic_weight
                else:
                    weight = 1.0
                rows.append({"url": url, "label": int(label), "weight": float(weight), "source": source})
    return rows


def cap_synthetic_malicious(rows: list[dict[str, object]], max_ratio: float, seed: int) -> list[dict[str, object]]:
    if max_ratio <= 0:
        return rows
    real_mal = [r for r in rows if r["label"] == 1 and r.get("source") != "malicious_synthetic_kr"]
    synthetic = [r for r in rows if r["label"] == 1 and r.get("source") == "malicious_synthetic_kr"]
    others = [r for r in rows if r["label"] == 0]
    max_synthetic = int(len(real_mal) * max_ratio / max(0.000001, 1.0 - max_ratio))
    if len(synthetic) <= max_synthetic:
        return rows
    rng = random.Random(seed)
    synthetic = rng.sample(synthetic, max_synthetic)
    return real_mal + synthetic + others


def balance_rows(rows: list[dict[str, object]], seed: int) -> list[dict[str, object]]:
    malicious = [r for r in rows if r["label"] == 1]
    benign = [r for r in rows if r["label"] == 0]
    if not malicious or not benign:
        return rows
    target = min(len(malicious), len(benign))
    rng = random.Random(seed)
    if len(malicious) > target:
        malicious = rng.sample(malicious, target)
    if len(benign) > target:
        benign = rng.sample(benign, target)
    balanced = malicious + benign
    rng.shuffle(balanced)
    return balanced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="CSV with url,label. Repeatable.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--holdout",
        action="append",
        default=[],
        help="CSV reserved for evaluation; matching URLs are excluded from training. Repeatable.",
    )
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
    parser.add_argument(
        "--hard-benign-weight",
        type=float,
        default=4.0,
        help="Training weight for hard benign Korean SMB/ad landing/user-like URLs.",
    )
    parser.add_argument(
        "--synthetic-weight",
        type=float,
        default=0.35,
        help="Training weight for malicious_synthetic_kr samples.",
    )
    parser.add_argument(
        "--max-synthetic-malicious-ratio",
        type=float,
        default=0.25,
        help="Maximum share of malicious training rows that may come from malicious_synthetic_kr.",
    )
    parser.add_argument(
        "--balance-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downsample the majority class so URLML trains on a 1:1 malicious/benign batch.",
    )
    parser.add_argument(
        "--class-weight-balanced",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use sklearn class_weight='balanced' in addition to sample weights.",
    )
    parser.add_argument("--C", type=float, default=4.0, help="LogisticRegression regularization inverse strength.")
    args = parser.parse_args()

    holdout_keys = load_holdout_keys(args.holdout)
    rows = read_csvs(
        args.input,
        args.user_confirmed_weight,
        args.naver_benign_weight,
        args.hard_benign_weight,
        args.synthetic_weight,
        exclude_keys=holdout_keys,
    )
    rows = cap_synthetic_malicious(rows, args.max_synthetic_malicious_ratio, args.random_state)
    loaded_rows = len(rows)
    if args.balance_samples:
        rows = balance_rows(rows, args.random_state)
    urls = [str(r["url"]) for r in rows]
    labels = [int(r["label"]) for r in rows]
    weights = [float(r["weight"]) for r in rows]
    if len(set(labels)) < 2:
        raise SystemExit("need both benign and malicious labels")
    print(
        f"loaded={loaded_rows} training_rows={len(urls)} malicious={sum(labels)} "
        f"benign={len(labels)-sum(labels)} holdout_excluded={len(holdout_keys)} "
        f"balanced={args.balance_samples}"
    )

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
                "features",
                FeatureUnion(
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
                        ("lexical", URLLexicalFeatures()),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=args.C,
                    class_weight="balanced" if args.class_weight_balanced else None,
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
