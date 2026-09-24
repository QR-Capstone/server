#!/usr/bin/env python3
"""Train an XGBoost URL model on real open-site corpora, excluding holdout URLs."""

from __future__ import annotations

import json
import os
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from xgboost import XGBClassifier

URL_ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "url_ml")
if URL_ML_DIR not in sys.path:
    sys.path.insert(0, URL_ML_DIR)

from train_url_ml import balance_rows, load_holdout_keys, read_csvs  # noqa: E402
from url_features import URLLexicalFeatures  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
DEFAULT_OUT = os.path.join(BASE_DIR, "url_xgb_open_site.joblib")


def main() -> int:
    inputs = [
        os.path.join(ROOT, "dev/engine_training_inputs/expanded_url_train_20260611.csv"),
        os.path.join(ROOT, "dev/real_site_split/nurilab_train.csv"),
    ]
    holdouts = [
        os.path.join(ROOT, "dev/real_site_split/test_malicious.csv"),
        os.path.join(ROOT, "dev/real_site_split/test_benign.csv"),
    ]
    rows = read_csvs(inputs, 5.0, 3.0, 4.0, 0.35, exclude_keys=load_holdout_keys(holdouts))
    rows = balance_rows(rows, seed=42)
    urls = [str(row["url"]) for row in rows]
    labels = [int(row["label"]) for row in rows]
    weights = [float(row["weight"]) for row in rows]
    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=80000,
                                sublinear_tf=True,
                            ),
                        ),
                        ("lexical", URLLexicalFeatures()),
                    ]
                ),
            ),
            (
                "clf",
                XGBClassifier(
                    n_estimators=180,
                    max_depth=6,
                    learning_rate=0.12,
                    subsample=0.85,
                    colsample_bytree=0.7,
                    min_child_weight=2,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    tree_method="hist",
                    n_jobs=4,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(urls, labels, clf__sample_weight=weights)
    joblib.dump(model, DEFAULT_OUT)
    meta = {
        "kind": "xgboost_open_site_url",
        "training_rows": len(rows),
        "malicious": sum(labels),
        "benign": len(labels) - sum(labels),
        "holdouts": holdouts,
        "inputs": inputs,
    }
    with open(DEFAULT_OUT + ".meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
