from pathlib import Path
import pandas as pd

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "calendar_features.csv"

# 2. Load original data
df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

# 3. Sort the time series
df = df.sort_values(
    ["store", "item", "date"]
).reset_index(drop=True)

# 4. Create calendar features
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day_of_week"] = df["date"].dt.dayofweek
df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# 5. Validate the output
assert len(df) == 913000
assert df[["year", "month", "day_of_week", "is_weekend"]].notna().all().all()
assert df["day_of_week"].between(0, 6).all()
assert df["is_weekend"].isin([0, 1]).all()

# 6. Save processed data
OUTPUT_PATH.parent.mkdir(
    parents=True, exist_ok=True
)

df.to_csv(OUTPUT_PATH, index=False)

# 7. Display results
print("=== CALENDAR FEATURE ENGINEERING ===")
print(df.head(10).to_string(index=False))

print("\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n=== NEW FEATURES ===")
print(["year", "month", "day_of_week", "is_weekend"])

print("\nProcessed data saved successfully!")
