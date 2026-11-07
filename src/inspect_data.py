
from pathlib import Path
import pandas as pd

# 1. Xac dinh duong dan du lieu
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"

# 2. Doc du lieu CSV
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# 3. Hien thi 5 dong dau tien
print("=== FIRST 5 ROWS ===")
print(df.head())

# 4. Kiem tra kich thuoc du lieu
print("\n=== DATASET SHAPE ===")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# 5. Kiem tra kieu du lieu
print("\n=== DATA TYPES ===")
print(df.dtypes)

# 6. Kiem tra du lieu bi thieu
print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

# 7. Kiem tra khoang thoi gian
print("\n=== DATE RANGE ===")
print(df["date"].min())
print(df["date"].max())
