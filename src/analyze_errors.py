from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "model_features.csv"
MODEL_PATH = ROOT / "models" / "hgb_model.joblib"
OUTPUT_PATH = ROOT / "reports" / "metrics" / "validation_by_store.csv"

# 2. Load trained model
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

# 3. Load validation data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

val = df.loc[
    df["date"].between("2016-01-01", "2016-12-24")
].copy()

assert len(val) == 179500

# 4. Generate predictions
val["prediction"] = np.clip(
    model.predict(val[features]),
    0,
    None
)

# 5. Calculate individual errors
val["absolute_error"] = np.abs(
    val["sales"] - val["prediction"]
)

val["baseline_error"] = np.abs(
    val["sales"] - val["lag_7"]
)

# 6. Evaluate each store
results = val.groupby("store").agg(
    samples=("sales", "size"),
    actual_sales=("sales", "sum"),
    mae=("absolute_error", "mean"),
    baseline_mae=("baseline_error", "mean"),
    total_error=("absolute_error", "sum")
)

results["wape_percent"] = (
    results["total_error"]
    / results["actual_sales"]
    * 100
)

results["improvement_percent"] = (
    (results["baseline_mae"] - results["mae"])
    / results["baseline_mae"]
    * 100
)

# 7. Display results
print("=== MODEL PERFORMANCE BY STORE ===")

print(
    results[
        [
            "samples",
            "mae",
            "baseline_mae",
            "wape_percent",
            "improvement_percent"
        ]
    ].round(2).to_string()
)

# 8. Identify best and worst stores
print("\n=== ERROR ANALYSIS SUMMARY ===")
print(f"Lowest MAE store: {results['mae'].idxmin()}")
print(f"Highest MAE store: {results['mae'].idxmax()}")

# 9. Save results
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(OUTPUT_PATH)

print("\nStore-level evaluation completed successfully!")
