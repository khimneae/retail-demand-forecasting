from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "model_features.csv"
MODEL_PATH = ROOT / "models" / "hgb_final_model.joblib"
OUTPUT_PATH = ROOT / "reports" / "metrics" / "final_test_results.json"

# 2. Load final trained model
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]

# 3. Load prepared dataset
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

# 4. Select untouched test period
test = df.loc[
    df["date"].between("2017-01-01", "2017-12-31")
].copy()

assert len(test) == 182500
assert test[features + ["sales", "lag_7"]].notna().all().all()
assert artifact["training_end"] == "2016-12-24"

X_test = test[features]
y_test = test["sales"].to_numpy(dtype=float)

# 5. Machine learning predictions
ml_predictions = model.predict(X_test)
ml_predictions = np.clip(ml_predictions, 0, None)

# 6. Seasonal naive baseline predictions
baseline_predictions = test["lag_7"].to_numpy(dtype=float)

# 7. Evaluation function
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    denominator = np.abs(y_true).sum()

    wape = (
        np.abs(y_true - y_pred).sum()
        / denominator * 100
        if denominator > 0
        else None
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "wape_percent": float(wape) if wape is not None else None
    }

# 8. Calculate metrics
baseline_metrics = evaluate(
    y_test, baseline_predictions
)

ml_metrics = evaluate(
    y_test, ml_predictions
)

# 9. Calculate MAE improvement
improvement = (
    (
        baseline_metrics["mae"] - ml_metrics["mae"]
    )
    / baseline_metrics["mae"] * 100
)

# 10. Display results
print("=== FINAL TEST EVALUATION ===")
print("Test period: 2017-01-01 to 2017-12-31")
print(f"Test samples: {len(test):,}")

print("\n=== BASELINE TEST RESULTS ===")
print(f"MAE: {baseline_metrics['mae']:.4f}")
print(f"RMSE: {baseline_metrics['rmse']:.4f}")
print(f"WAPE: {baseline_metrics['wape_percent']:.2f}%")

print("\n=== MACHINE LEARNING TEST RESULTS ===")
print(f"MAE: {ml_metrics['mae']:.4f}")
print(f"RMSE: {ml_metrics['rmse']:.4f}")
print(f"WAPE: {ml_metrics['wape_percent']:.2f}%")

print("\n=== FINAL COMPARISON ===")
print(f"Baseline MAE: {baseline_metrics['mae']:.4f}")
print(f"ML Model MAE: {ml_metrics['mae']:.4f}")
print(f"MAE improvement: {improvement:.2f}%")

# 11. Save reproducible evaluation results
results = {
    "model": "HistGradientBoostingRegressor",
    "dataset": "Test 2017",
    "samples": len(test),
    "forecast_horizon_days": 7,
    "evaluation_protocol": "Rolling-origin with observed history",
    "model_training_end": artifact["training_end"],
    "baseline": baseline_metrics,
    "machine_learning": ml_metrics,
    "mae_improvement_percent": float(improvement)
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("\nFinal test evaluation completed successfully!")
