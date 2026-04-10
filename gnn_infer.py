"""
GNN (service name): lexical URL features + RandomForest (opqr_model.pkl) from gnn-ready.py.
Loads the RF bundle, not a PyG checkpoint.
Python 3.10+ recommended (pickle from newer Python may fail on 3.9).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

# Feature dict order must match training (gnn-ready DataFrame columns).
FEATURE_ORDER: List[str] = [
    "url_len",
    "dot_count",
    "hyphen_count",
    "slash_count",
    "is_https",
    "digit_ratio",
    "keyword_match",
]

SUSPICIOUS_WORDS = ("login", "verify", "bank", "update", "free", "account")


def extract_lexical_features(url: str) -> Dict[str, float]:
    """Same feature dict as gnn-ready.py for one URL."""
    u = (url or "").strip()
    features: Dict[str, float] = {}
    features["url_len"] = float(len(u))
    features["dot_count"] = float(u.count("."))
    features["hyphen_count"] = float(u.count("-"))
    features["slash_count"] = float(u.count("/"))
    features["is_https"] = 1.0 if u.startswith("https") else 0.0
    digits = re.findall(r"\d", u)
    features["digit_ratio"] = float(len(digits) / len(u)) if len(u) > 0 else 0.0
    ul = u.lower()
    features["keyword_match"] = (
        1.0 if any(word in ul for word in SUSPICIOUS_WORDS) else 0.0
    )
    return features


def _feature_vector_row(url: str, column_order: List[str]) -> pd.DataFrame:
    feat = extract_lexical_features(url)
    row = {k: feat.get(k, 0.0) for k in column_order}
    return pd.DataFrame([row])


def load_opqr_model(
    model_path: str,
    feature_columns_path: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    """Load model and return column order for prediction."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(
            "Failed to load opqr_model.pkl (corrupt file or pickle from a newer Python). "
            "Use Python 3.11+ on the server or re-save with "
            "`joblib.dump(model, 'opqr_model.pkl', compress=3)` from a compatible env."
        ) from e
    cols: Optional[List[str]] = None
    if feature_columns_path and os.path.isfile(feature_columns_path):
        loaded = joblib.load(feature_columns_path)
        if isinstance(loaded, list):
            cols = [str(c) for c in loaded]
    if cols is None:
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
        else:
            cols = list(FEATURE_ORDER)
    return model, cols


def predict_opqr(
    model: Any,
    column_order: List[str],
    raw_url: str,
) -> Dict[str, Any]:
    """Return phishing probability, label, verdict; raises on failure."""
    u = (raw_url or "").strip()
    if not u:
        raise ValueError("empty url")

    if not u.startswith(("http://", "https://")):
        u = "https://" + u

    X = _feature_vector_row(u, column_order)
    prob_mal = float(model.predict_proba(X)[0][1])
    label = int(model.predict(X)[0])
    return {
        "url": u,
        "probability": round(prob_mal, 6),
        "label": label,
        "verdict": "malicious" if label == 1 else "benign",
    }
