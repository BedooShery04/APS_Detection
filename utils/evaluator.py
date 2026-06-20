from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_predictions(y_true, y_pred):
    """
    Evaluate numeric APS predictions.

    Label convention:
    0 = Normal
    1 = APS Failure
    """

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Label length mismatch: y_true={len(y_true)} vs y_pred={len(y_pred)}"
        )

    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        raise ValueError(
            "Confusion matrix shape invalid. Expected binary classification."
        )

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "cm": cm.tolist(),
        "cost": int((fn * 500) + (fp * 10))
    }

    return metrics