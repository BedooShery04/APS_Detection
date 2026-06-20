import json
from pathlib import Path

models_dir = Path("models")

with open(models_dir / "logistic_regression_metrics.json", "r") as f:
    lr_metrics = json.load(f)

with open(models_dir / "random_forest_metrics.json", "r") as f:
    rf_metrics = json.load(f)

with open(models_dir / "lightgbm_metrics.json", "r") as f:
    lgb_metrics = json.load(f)

all_metrics = {
    "Logistic Regression": lr_metrics,
    "Random Forest": rf_metrics,
    "LightGBM": lgb_metrics
}

with open(models_dir / "training_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)

print("training_metrics.json created successfully")