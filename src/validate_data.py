from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

# 1. Load data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

print("=== DATA VALIDATION REPORT ===")

# 2. Dataset dimensions
print(f"\nTotal records: {len(df):,}")
print(f"Stores: {df['store'].nunique()}")
print(f"Items: {df['item'].nunique()}")

# 3. Missing values
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# 4. Duplicate records
duplicates = df.duplicated(
    subset=["date", "store", "item"]
).sum()

print(f"\nDuplicate date-store-item keys: {duplicates}")

# 5. Invalid sales
negative_sales = (df["sales"] < 0).sum()
zero_sales = (df["sales"] == 0).sum()

print(f"Negative sales records: {negative_sales}")
print(f"Zero sales records: {zero_sales}")

# 6. Date coverage
start_date = df["date"].min()
end_date = df["date"].max()

expected_days = (end_date - start_date).days + 1
actual_days = df["date"].nunique()

print("\n=== DATE COVERAGE ===")
print(f"Start: {start_date.date()}")
print(f"End: {end_date.date()}")
print(f"Expected days: {expected_days}")
print(f"Observed days: {actual_days}")

# 7. Check store-item series completeness
series_counts = df.groupby(
    ["store", "item"]
)["date"].nunique()

incomplete_series = (series_counts != expected_days).sum()

print("\n=== TIME SERIES COVERAGE ===")
print(f"Store-item combinations: {len(series_counts)}")
print(f"Incomplete combinations: {incomplete_series}")
