from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "model_features.csv"
MODEL_PATH = ROOT / "models" / "hgb_model.joblib"
OUTPUT_PATH = ROOT / "reports" / "metrics" / "validation_by_item.csv"

# 2. Load model
artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
features = artifact["features"]

# 3. Load validation data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

val = df.loc[
    df["date"].between("2016-01-01", "2016-12-24")
].copy()

assert len(val) == 179500

# 4. Predict
val["prediction"] = np.clip(
    model.predict(val[features]),
    0,
    None
)

# 5. Calculate errors
val["ml_error"] = np.abs(
    val["sales"] - val["prediction"]
)

val["baseline_error"] = np.abs(
    val["sales"] - val["lag_7"]
)

# 6. Group results by item
results = val.groupby("item").agg(
    samples=("sales", "size"),
    actual_sales=("sales", "sum"),
    mae=("ml_error", "mean"),
    baseline_mae=("baseline_error", "mean"),
    total_error=("ml_error", "sum")
)

# 7. Calculate WAPE and improvement
results["wape_percent"] = np.where(
    results["actual_sales"] > 0,
    results["total_error"] / results["actual_sales"] * 100,
    np.nan
)

results["improvement_percent"] = np.where(
    results["baseline_mae"] > 0,
    (results["baseline_mae"] - results["mae"])
    / results["baseline_mae"] * 100,
    np.nan
)

# 8. Print summary
print("=== ITEM-LEVEL ERROR ANALYSIS ===")
print(f"Items analyzed: {len(results)}")

improved = (results["improvement_percent"] > 0).sum()
not_improved = (results["improvement_percent"] <= 0).sum()

print(f"Items improved: {improved}")
print(f"Items not improved: {not_improved}")

# 9. Show five items with highest relative error
print("\n=== TOP 5 HIGHEST WAPE ITEMS ===")

print(
    results.sort_values(
        "wape_percent", ascending=False
    )[["mae", "baseline_mae", "wape_percent",
       "improvement_percent"]]
    .head(5)
    .round(2)
    .to_string()
)

# 10. Show five items with lowest improvement
print("\n=== TOP 5 LOWEST IMPROVEMENT ITEMS ===")

print(
    results.sort_values("improvement_percent")
    [["mae", "baseline_mae", "wape_percent",
      "improvement_percent"]]
    .head(5)
    .round(2)
    .to_string()
)

# 11. Save results
OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

results.to_csv(OUTPUT_PATH)

print("\nItem-level evaluation completed successfully!")
