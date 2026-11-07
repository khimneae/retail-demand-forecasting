from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "model_features.csv"
BASELINE_PATH = ROOT / "reports" / "metrics" / "baseline_validation.json"
METRICS_PATH = ROOT / "reports" / "metrics" / "hgb_validation.json"
MODEL_PATH = ROOT / "models" / "hgb_model.joblib"

# 2. Define input features
FEATURES = [
    "store",
    "item",
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]

TARGET = "sales"

# 3. Load prepared dataset
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# 4. Chronological train/validation split
train = df[df["date"] <= "2015-12-24"].copy()

val = df[
    df["date"].between("2016-01-01", "2016-12-24")
].copy()

assert len(train) == 527000
assert len(val) == 179500

X_train = train[FEATURES]
y_train = train[TARGET]

X_val = val[FEATURES]
y_val = val[TARGET]

# 5. Initialize model
model = HistGradientBoostingRegressor(
    max_iter=120,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=False,
    random_state=42
)

# 6. Train
print("=== TRAINING MACHINE LEARNING MODEL ===")
print(f"Training samples: {len(X_train):,}")
print(f"Validation samples: {len(X_val):,}")
print(f"Features: {len(FEATURES)}")
print("Training started...", flush=True)

model.fit(X_train, y_train)

print("Training completed!")

# 7. Predict on validation only
predictions = model.predict(X_val)

# Sales quantities cannot be negative
predictions = np.clip(predictions, 0, None)

# 8. Evaluate
mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

wape = (
    np.abs(y_val.to_numpy() - predictions).sum()
    / np.abs(y_val.to_numpy()).sum()
    * 100
)

print("\n=== MACHINE LEARNING VALIDATION RESULTS ===")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%")

# 9. Compare with baseline
baseline = json.loads(
    BASELINE_PATH.read_text(encoding="utf-8")
)

baseline_mae = baseline["mae"]

improvement = (
    (baseline_mae - mae) / baseline_mae * 100
)

print("\n=== BASELINE COMPARISON ===")
print(f"Baseline MAE: {baseline_mae:.4f}")
print(f"ML Model MAE: {mae:.4f}")
print(f"MAE improvement: {improvement:.2f}%")

# 10. Save model and metrics
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {"model": model, "features": FEATURES},
    MODEL_PATH
)

results = {
    "model": "HistGradientBoostingRegressor",
    "dataset": "Validation 2016",
    "forecast_horizon_days": 7,
    "evaluation": "Rolling-origin with observed history",
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "mae": round(float(mae), 4),
    "rmse": round(float(rmse), 4),
    "wape_percent": round(float(wape), 2),
    "baseline_mae": baseline_mae,
    "mae_improvement_percent": round(float(improvement), 2),
    "features": FEATURES
}

METRICS_PATH.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("\nModel and metrics saved successfully!")
