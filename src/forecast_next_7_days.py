from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "raw" / "train.csv"
FEATURES_PATH = ROOT / "data" / "processed" / "model_features.csv"
MODEL_PATH = ROOT / "models" / "hgb_final_model.joblib"

STORE_ID = 4
ITEM_ID = 22
FORECAST_DAYS = 7


# ==================================================
# 2. LOAD MODEL AND HISTORICAL DATA
# ==================================================

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_names = artifact["features"]

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

last_date = df["date"].max()

history = df.loc[
    (df["store"] == STORE_ID)
    & (df["item"] == ITEM_ID)
].sort_values("date")

if history.empty:
    raise ValueError("Selected store-item has no history.")

if history["date"].duplicated().any():
    raise ValueError("Duplicate dates detected.")

sales = history.set_index("date")["sales"]

expected_dates = pd.date_range(
    sales.index.min(),
    sales.index.max(),
    freq="D"
)

if not sales.index.equals(expected_dates):
    raise ValueError("Historical date coverage is incomplete.")

if sales.index.max() != last_date:
    raise ValueError("Selected series is not up to date.")


# ==================================================
# 3. CREATE FEATURES FROM OBSERVED HISTORY
# ==================================================

def historical_sales(target_date, days_ago):
    source_date = target_date - pd.Timedelta(
        days=days_ago
    )

    if source_date > last_date:
        raise ValueError("Future data access detected.")

    return float(sales.loc[source_date])


def build_features(target_date):
    target_date = pd.Timestamp(target_date)

    lag_7 = historical_sales(target_date, 7)
    lag_14 = historical_sales(target_date, 14)
    lag_28 = historical_sales(target_date, 28)

    # Mean of observations from t-7 through t-13
    rolling_mean_7 = np.mean([
        historical_sales(target_date, i)
        for i in range(7, 14)
    ])

    # Mean of observations from t-7 through t-34
    rolling_mean_28 = np.mean([
        historical_sales(target_date, i)
        for i in range(7, 35)
    ])

    weekday = target_date.dayofweek

    return {
        "store": STORE_ID,
        "item": ITEM_ID,
        "year": target_date.year,
        "month": target_date.month,
        "day_of_week": weekday,
        "is_weekend": int(weekday >= 5),
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "rolling_mean_7": rolling_mean_7,
        "rolling_mean_28": rolling_mean_28
    }


# ==================================================
# 4. VALIDATE FEATURE CONSISTENCY
# ==================================================

reference = pd.read_csv(
    FEATURES_PATH,
    parse_dates=["date"]
)

reference_row = reference.loc[
    (reference["date"] == last_date)
    & (reference["store"] == STORE_ID)
    & (reference["item"] == ITEM_ID)
]

if len(reference_row) != 1:
    raise ValueError("Reference feature row not found.")

audit_features = build_features(last_date)
saved_row = reference_row.iloc[0]

for name in feature_names:
    if not np.isclose(
        audit_features[name],
        saved_row[name],
        rtol=1e-7,
        atol=1e-7
    ):
        raise ValueError(
            f"Feature mismatch detected: {name}"
        )

print("Feature consistency check passed!")


# ==================================================
# 5. PREPARE NEXT 7 DAYS
# ==================================================

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=FORECAST_DAYS,
    freq="D"
)

future_features = pd.DataFrame([
    build_features(day)
    for day in future_dates
])

X_future = future_features[feature_names]


# ==================================================
# 6. GENERATE FORECASTS
# ==================================================

predictions = model.predict(X_future)

predictions = np.clip(
    predictions,
    0,
    None
)

forecast = pd.DataFrame({
    "date": future_dates,
    "store": STORE_ID,
    "item": ITEM_ID,
    "predicted_units": predictions.round(2),
    "rounded_predicted_units": np.rint(
        predictions
    ).astype(int),
    "seasonal_naive_units": (
        future_features["lag_7"].astype(int)
    )
})


# ==================================================
# 7. DISPLAY AND SAVE RESULTS
# ==================================================

print("\n=== 7-DAY DEMAND FORECAST ===")
print(f"Store: {STORE_ID}")
print(f"Item: {ITEM_ID}")
print(f"Last observed date: {last_date.date()}")

print("\n", forecast.to_string(index=False))

print(
    "\nTotal predicted units:",
    forecast["predicted_units"].sum().round(2)
)

output_dir = ROOT / "reports" / "forecasts"
output_dir.mkdir(parents=True, exist_ok=True)

filename = (
    f"forecast_store_{STORE_ID}_"
    f"item_{ITEM_ID}_"
    f"{future_dates[0]:%Y-%m-%d}_"
    f"{future_dates[-1]:%Y-%m-%d}.csv"
)

output_path = output_dir / filename

forecast.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nForecast saved to: {output_path}")
print("7-day forecasting completed successfully!")
