"""
Lexical URL features + RandomForest (GNN lexical service name; RF checkpoint).
Not a PyG model — same pipeline as gnn-ready.py; column order from gnn_model_features.pkl.
Python 3.10+ recommended (pickle may fail on older runtimes).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Fallback if the model has no feature_names_in_ and no feature list file exists.
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


def resolve_gnn_model_path(base_dir: str = _BASE_DIR) -> str:
    """Env GNN_MODEL_PATH, legacy OPQR_MODEL_PATH, then gnn_model.pkl or opqr_model.pkl."""
    if p := os.getenv("GNN_MODEL_PATH"):
        return p
    if p := os.getenv("OPQR_MODEL_PATH"):
        return p
    for name in ("gnn_model.pkl", "opqr_model.pkl"):
        cand = os.path.join(base_dir, name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(base_dir, "gnn_model.pkl")


def resolve_gnn_features_path(base_dir: str = _BASE_DIR) -> str:
    """Env GNN_FEATURES_PATH, legacy OPQR_FEATURES_PATH, then feature pkl filenames."""
    if p := os.getenv("GNN_FEATURES_PATH"):
        return p
    if p := os.getenv("OPQR_FEATURES_PATH"):
        return p
    for name in ("gnn_model_features.pkl", "model_features.pkl"):
        cand = os.path.join(base_dir, name)
        if os.path.isfile(cand):
            return cand
    return os.path.join(base_dir, "gnn_model_features.pkl")


def default_gnn_paths() -> Tuple[str, str]:
    return resolve_gnn_model_path(), resolve_gnn_features_path()


def extract_lexical_features(url: str) -> Dict[str, float]:
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


def _explain_gnn_load_error(exc: Exception) -> str:
    """KeyError(239) often = pickle from Python 3.13+ read on 3.12 (unknown opcode)."""
    name = type(exc).__name__
    msg = str(exc)
    opcode_mismatch = isinstance(exc, KeyError) and (
        (exc.args and isinstance(exc.args[0], int)) or "239" in msg
    )
    if opcode_mismatch:
        return (
            "Failed to load gnn_model.pkl: pickle opcode mismatch "
            "(e.g. KeyError 239: file produced with Python 3.13+ unpickled on 3.12). "
            "Fix: regenerate gnn_model.pkl on Python 3.12.3, or run the API on Python 3.13+, "
            "or run reexport_gnn_model.py where the file loads and save with Python 3.12. "
            "Also keep scikit-learn/joblib aligned with training. "
            f"Detail: {name}: {msg}"
        )
    return (
        "Failed to load gnn_model.pkl. Align scikit-learn/joblib/numpy with training "
        "or regenerate via gnn-ready.py / reexport_gnn_model.py. "
        f"Detail: {name}: {msg}"
    )


def load_gnn_model(
    model_path: str,
    feature_columns_path: Optional[str] = None,
) -> Tuple[Any, List[str]]:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(_explain_gnn_load_error(e)) from e
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


def predict_gnn(
    model: Any,
    column_order: List[str],
    raw_url: str,
) -> Dict[str, Any]:
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


class GNN_Engine:
    """Lexical RF; paths default to repo root (gnn_model.pkl + gnn_model_features.pkl)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_columns_path: Optional[str] = None,
    ):
        self.model_path = model_path or resolve_gnn_model_path()
        self.feature_columns_path = feature_columns_path or resolve_gnn_features_path()
        self.model: Any = None
        self.columns: Optional[List[str]] = None
        self._load_error: Optional[str] = None

        if not os.path.isfile(self.model_path):
            self._load_error = f"missing_model:{self.model_path}"
            return

        try:
            fc = self.feature_columns_path if os.path.isfile(self.feature_columns_path) else None
            self.model, self.columns = load_gnn_model(self.model_path, fc)
        except Exception as e:
            self.model = None
            self.columns = None
            self._load_error = str(e)

    @property
    def ok(self) -> bool:
        return self.model is not None and bool(self.columns)

    def analyze(self, url: str) -> Dict[str, Any]:
        if not self.ok:
            return {
                "error": self._load_error or "model_not_loaded",
                "verdict": "unknown",
                "enabled": False,
            }
        try:
            return predict_gnn(self.model, self.columns, url)
        except Exception as e:
            return {
                "error": str(e),
                "verdict": "unknown",
                "enabled": True,
            }

    def analyze_legacy(self, url: str) -> Dict[str, Any]:
        out = self.analyze(url)
        if out.get("error"):
            return out
        p = float(out.get("probability", 0.0))
        mal = out.get("verdict") == "malicious"
        out = dict(out)
        out["target_url"] = out.get("url", url)
        out["danger_score"] = f"{p * 100:.2f}%"
        out["detection_result"] = "MALICIOUS" if mal else "SAFE"
        return out
