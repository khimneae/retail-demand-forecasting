from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "model_features.csv"

# 1. Load prepared data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# 2. Split chronologically with 7-day gaps
train_mask = df["date"] <= "2015-12-24"

val_mask = (
    (df["date"] >= "2016-01-01")
    & (df["date"] <= "2016-12-24")
)

test_mask = df["date"] >= "2017-01-01"

train = df.loc[train_mask].copy()
val = df.loc[val_mask].copy()
test = df.loc[test_mask].copy()

# 3. Count excluded gap rows
gap_rows = len(df) - len(train) - len(val) - len(test)

# 4. Validate split
assert not train.empty
assert not val.empty
assert not test.empty

assert train["date"].max() < val["date"].min()
assert val["date"].max() < test["date"].min()

assert len(train) + len(val) + len(test) + gap_rows == len(df)

# 5. Display summary
print("=== TIME SERIES DATA SPLIT ===")

for name, subset in [
    ("TRAIN", train),
    ("VALIDATION", val),
    ("TEST", test)
]:
    print(f"\n{name}")
    print(f"Rows: {len(subset):,}")
    print(f"Start: {subset['date'].min().date()}")
    print(f"End: {subset['date'].max().date()}")

print(f"\nGap rows excluded: {gap_rows:,}")
print("\nTime-based splitting completed successfully!")
