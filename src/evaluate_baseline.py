from pathlib import Path
import json
import numpy as np
import pandas as pd

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "model_features.csv"
)

OUTPUT_PATH = (
    ROOT / "reports" / "metrics" / "baseline_validation.json"
)

# 2. Load processed data
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"],
    usecols=["date", "store", "item", "sales", "lag_7"]
)

# 3. Select validation period only
val = df.loc[
    df["date"].between("2016-01-01", "2016-12-24")
].copy()

assert len(val) == 179500
assert val[["sales", "lag_7"]].notna().all().all()

# 4. Seasonal naive prediction
y_true = val["sales"].to_numpy(dtype=float)
y_pred = val["lag_7"].to_numpy(dtype=float)

# 5. Calculate evaluation metrics
errors = y_true - y_pred

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))

total_actual = np.sum(np.abs(y_true))
wape = (
    np.sum(np.abs(errors)) / total_actual * 100
    if total_actual > 0 else None
)

# 6. Display results
print("=== BASELINE FORECASTING RESULTS ===")
print("Method: Seasonal Naive (Lag 7)")
print("Dataset: Validation (2016)")
print(f"Samples: {len(val):,}")

print(f"\nMAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"WAPE: {wape:.2f}%" if wape is not None else "WAPE: N/A")

# 7. Save metrics for GitHub documentation
results = {
    "model": "Seasonal Naive (Lag 7)",
    "dataset": "Validation",
    "forecast_horizon_days": 7,
    "evaluation": "Rolling-origin with observed history",
    "samples": len(val),
    "mae": round(float(mae), 4),
    "rmse": round(float(rmse), 4),
    "wape_percent": round(float(wape), 2) if wape is not None else None
}

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH.write_text(
    json.dumps(results, indent=2),
    encoding="utf-8"
)

print("\nBaseline evaluation completed successfully!")
