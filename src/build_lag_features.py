from pathlib import Path
import pandas as pd

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "data" / "processed" / "calendar_features.csv"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "model_features.csv"
)

# 2. Load calendar features
df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["date"]
)

# 3. Sort each store-item time series
df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)

group_cols = ["store", "item"]

# 4. Create lag features
for lag in [7, 14, 28]:
    df[f"lag_{lag}"] = (
        df.groupby(group_cols)["sales"]
        .shift(lag)
    )

# 5. Create leakage-safe rolling features
# Shift by 7 days to support a 7-day forecast horizon.
for window in [7, 28]:
    df[f"rolling_mean_{window}"] = (
        df.groupby(group_cols)["sales"]
        .transform(
            lambda s: s.shift(7)
            .rolling(window=window, min_periods=window)
            .mean()
        )
    )

# 6. Define feature columns
feature_cols = [
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28"
]

print("=== FEATURE ENGINEERING REPORT ===")

print("\nMissing values before cleaning:")
print(df[feature_cols].isna().sum())

# 7. Remove rows without sufficient history
df = df.dropna(
    subset=feature_cols
).reset_index(drop=True)

# 8. Validate generated features
assert df[feature_cols].notna().all().all()
assert df["store"].nunique() == 10
assert df["item"].nunique() == 50

# 9. Save dataset
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(OUTPUT_PATH, index=False)

# 10. Display output
print("\n=== FINAL DATASET ===")
print(f"Rows: {len(df):,}")
print(f"Columns: {df.shape[1]}")

print("\n=== FEATURE SAMPLE ===")
print(
    df.loc[
        (df["store"] == 1) &
        (df["item"] == 1)
    ].head(5).to_string(index=False)
)

print("\nFeature engineering completed successfully!")
