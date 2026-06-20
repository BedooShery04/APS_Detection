# import pandas as pd
# from utils.model_loader import load_model, load_all_models
# from utils.preprocessor import preprocess_input


# def _build_results(df_original, model, processed_df):
#     """Run a single model and return results DataFrame."""
#     predictions = model.predict(processed_df)

#     results = df_original.copy()
#     results["Prediction"] = [
#         "APS Failure" if pred == 1 else "Normal"
#         for pred in predictions
#     ]

#     if hasattr(model, "predict_proba"):
#         probs = model.predict_proba(processed_df)[:, 1]
#         results["Confidence (%)"] = (probs * 100).round(2)

#     return results


# def predict(df, model_name: str) -> pd.DataFrame:
#     """Run prediction for a single named model."""
#     model = load_model(model_name)
#     processed_df = preprocess_input(df)
#     return _build_results(df, model, processed_df)


# def predict_all(df) -> dict:
#     """
#     Run all available models on the same dataframe.
#     Returns dict {model_label: results_df}.
#     """
#     processed_df = preprocess_input(df)
#     all_models = load_all_models()
#     results = {}
#     for name, model in all_models.items():
#         results[name] = _build_results(df, model, processed_df)
#     return results



import pandas as pd
from utils.model_loader import load_model, load_all_models
from utils.preprocessor import preprocess_input
from utils.evaluator import evaluate_predictions


def _build_prediction_output(df_original, model, processed_df, true_labels=None):
    """
    Run a single model prediction and return structured output.
    """

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------
    y_pred = model.predict(processed_df)

    # --------------------------------------------------
    # BUILD PREDICTION DATAFRAME
    # --------------------------------------------------
    prediction_df = df_original.copy()
    prediction_df["Prediction"] = y_pred

    # Optional confidence scores
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_df)[:, 1]
        prediction_df["Confidence (%)"] = (probs * 100).round(2)

    result = {
        "y_pred": y_pred,
        "prediction_df": prediction_df
    }

    # --------------------------------------------------
    # OPTIONAL EVALUATION
    # --------------------------------------------------
    if true_labels is not None:
        metrics = evaluate_predictions(true_labels, y_pred)
        result["metrics"] = metrics

    return result


def predict(df, model_name: str, true_labels=None):
    """
    Run prediction for a single named model.

    Returns:
    {
        "y_pred": ...,
        "prediction_df": ...,
        "metrics": ... (if labels available)
    }
    """
    model = load_model(model_name)
    processed_df = preprocess_input(df)

    return _build_prediction_output(
        df_original=df,
        model=model,
        processed_df=processed_df,
        true_labels=true_labels
    )


def predict_all(df, true_labels=None):
    """
    Run all available models on the same dataframe.

    Returns:
    {
        "Logistic Regression": {...},
        "Random Forest": {...},
        "LightGBM": {...}
    }
    """
    processed_df = preprocess_input(df)
    all_models = load_all_models()

    results = {}

    for model_name, model in all_models.items():
        results[model_name] = _build_prediction_output(
            df_original=df,
            model=model,
            processed_df=processed_df,
            true_labels=true_labels
        )

    return results