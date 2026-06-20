# import joblib
# import pandas as pd
# from pathlib import Path

# BASE_DIR = Path(__file__).resolve().parent.parent
# ARTIFACTS_PATH = BASE_DIR / "models" / "preprocess_artifacts.pkl"


# def load_artifacts():
#     return joblib.load(ARTIFACTS_PATH)


# def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
#     """Apply saved APS preprocessing pipeline to a new dataframe."""
#     artifacts = load_artifacts()
#     imputer         = artifacts["imputer"]
#     scaler          = artifacts["scaler"]
#     dropped_cols    = artifacts["dropped_cols"]
#     var_selector    = artifacts["var_selector"]
#     selected_features = artifacts["selected_features"]
#     dropped_corr    = artifacts["dropped_corr"]

#     X = df.copy()

#     # Drop label column if present
#     if "class" in X.columns:
#         y = X["class"]
#         X = X.drop(columns=["class"])

#     X = X.drop(columns= dropped_cols, errors="ignore")
#     feature_names = list(X.columns)

#     # ── Imputation ─────────────────────────────────────────
#     X = imputer.transform(X)
#     X = pd.DataFrame(X, columns= feature_names)

#     # ── Variance selection ─────────────────────────────────
#     X = var_selector.transform(X)
#     feature_names = [feature_names[i] for i, s in enumerate(var_selector.get_support()) if s]
#     X = pd.DataFrame(X, columns= feature_names)

#     # ── Scaling ────────────────────────────────────────────
#     X = scaler.transform(X)
#     X = pd.DataFrame(X, columns=feature_names)

#     # ── Drop correlated features ───────────────────────────
#     X = X.drop(columns=dropped_corr, errors='ignore')

#     # ── Select final features ──────────────────────────────
#     X = X.reindex(columns=selected_features, fill_value=0)

#     return X



import joblib
import pandas as pd
from pathlib import Path
import streamlit as st

# --------------------------------------------------
# PATH CONFIG
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_PATH = BASE_DIR / "models" / "preprocess_artifacts.pkl"


# --------------------------------------------------
# LOAD ARTIFACTS (cached)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    return joblib.load(ARTIFACTS_PATH)


# --------------------------------------------------
# PREPROCESS INPUT DATA
# --------------------------------------------------
def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply saved APS preprocessing pipeline to uploaded input data.
    Returns model-ready dataframe.
    """
    try:
        artifacts = load_artifacts()

        imputer = artifacts["imputer"]
        scaler = artifacts["scaler"]
        dropped_cols = artifacts["dropped_cols"]
        var_selector = artifacts["var_selector"]
        selected_features = artifacts["selected_features"]
        dropped_corr = artifacts["dropped_corr"]

    except KeyError as e:
        raise ValueError(f"Missing preprocessing artifact key: {e}")

    except Exception as e:
        raise ValueError(f"Failed to load preprocessing artifacts: {e}")

    # --------------------------------------------------
    # COPY INPUT
    # --------------------------------------------------
    X = df.copy()

    # Safety guard
    if X.empty:
        raise ValueError("Uploaded dataset is empty.")

    # Drop label column if somehow still present
    if "class" in X.columns:
        X = X.drop(columns=["class"])

    # --------------------------------------------------
    # DROP HIGH-MISSING COLUMNS
    # --------------------------------------------------
    X = X.drop(columns=dropped_cols, errors="ignore")

    if X.empty:
        raise ValueError(
            "No usable columns remain after dropping training-removed features."
        )

    feature_names = list(X.columns)

    # --------------------------------------------------
    # IMPUTATION
    # --------------------------------------------------
    try:
        X = imputer.transform(X)
        X = pd.DataFrame(X, columns=feature_names)

    except Exception as e:
        raise ValueError(
            f"Imputation failed. Uploaded dataset may not match APS training schema: {e}"
        )

    # --------------------------------------------------
    # VARIANCE FILTERING
    # --------------------------------------------------
    try:
        X = var_selector.transform(X)

        feature_names = [
            feature_names[i]
            for i, keep in enumerate(var_selector.get_support())
            if keep
        ]

        X = pd.DataFrame(X, columns=feature_names)

    except Exception as e:
        raise ValueError(f"Variance filtering failed: {e}")

    # --------------------------------------------------
    # SCALING
    # --------------------------------------------------
    try:
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns=feature_names)

    except Exception as e:
        raise ValueError(f"Scaling failed: {e}")

    # --------------------------------------------------
    # DROP CORRELATED FEATURES
    # --------------------------------------------------
    X = X.drop(columns=dropped_corr, errors="ignore")

    # --------------------------------------------------
    # FINAL FEATURE ALIGNMENT
    # --------------------------------------------------
    X = X.reindex(columns=selected_features, fill_value=0)

    if X.empty:
        raise ValueError("Final processed dataset is empty.")

    return X