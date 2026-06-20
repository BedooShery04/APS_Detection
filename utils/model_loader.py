# import os
# import joblib

# MODELS_DIR = "models"


# def list_models():
#     """
#     Return available saved models.
#     """
#     if not os.path.exists(MODELS_DIR):
#         return []

#     model_files = [
#         f for f in os.listdir(MODELS_DIR)
#         if f.endswith(".pkl") and f != "preprocess_artifacts.pkl"
#     ]

#     readable_names = {
#         "logistic_regression.pkl": "Logistic Regression",
#         "random_forest.pkl": "Random Forest",
#         "svm.pkl": "Support Vector Machine",
#         "xgboost.pkl": "XGBoost",
#         "lightgbm.pkl": "LightGBM"
#     }

#     return [
#         readable_names.get(file, file.replace(".pkl", ""))
#         for file in model_files
#     ]


# def load_model(model_name):
#     """
#     Load selected model.
#     """
#     mapping = {
#         "Logistic Regression": "logistic_regression.pkl",
#         "Random Forest": "random_forest.pkl",
#         "Support Vector Machine": "svm.pkl",
#         "XGBoost": "xgboost.pkl",
#         "LightGBM": "lightgbm.pkl"
#     }

#     filename = mapping.get(model_name)

#     if not filename:
#         raise ValueError(f"Unknown model: {model_name}")

#     path = os.path.join(MODELS_DIR, filename)

#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Model not found: {path}")

#     return joblib.load(path)


import os
import joblib

MODELS_DIR = "models"

MODEL_META = {
    "logistic_regression.pkl": {
        "label": "Logistic Regression",
        "icon": "🔵",
        "desc": "Linear baseline — fast and interpretable",
    },
    "random_forest.pkl": {
        "label": "Random Forest",
        "icon": "🟢",
        "desc": "Ensemble of decision trees, robust to noise",
    },
    "lightgbm.pkl": {
        "label": "LightGBM",
        "icon": "🟠",
        "desc": "Leaf-wise gradient boosting, best for high-dim tabular",
    },
}


def list_models():
    """Return list of (label, icon, desc) for available saved models."""
    if not os.path.exists(MODELS_DIR):
        return []

    available = []
    for fname, meta in MODEL_META.items():
        if os.path.exists(os.path.join(MODELS_DIR, fname)):
            available.append(meta)
    return available


def list_model_labels():
    """Return just the display labels for selectbox use."""
    return [m["label"] for m in list_models()]


def load_model(model_name: str):
    """Load model by display label."""
    reverse = {meta["label"]: fname for fname, meta in MODEL_META.items()}
    filename = reverse.get(model_name)
    if not filename:
        raise ValueError(f"Unknown model: {model_name}")
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_all_models():
    """Load all available models. Returns dict {label: model}."""
    models = {}
    for fname, meta in MODEL_META.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[meta["label"]] = joblib.load(path)
    return models