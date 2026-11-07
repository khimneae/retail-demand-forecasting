from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Xac dinh thu muc du an
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

# 2. Doc du lieu
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# 3. Tinh tong so luong ban ra moi ngay
daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .sort_index()
)

# 4. Tinh trung binh truot 7 ngay
moving_avg = daily_sales.rolling(window=7).mean()

# 5. Hien thi thong tin
print("=== DAILY SALES ANALYSIS ===")
print(f"Number of days: {len(daily_sales)}")
print(f"Total units sold: {daily_sales.sum():,}")
print(f"Average daily sales: {daily_sales.mean():.2f}")

# 6. Ve bieu do
plt.figure(figsize=(12, 5))

plt.plot(
    daily_sales.index,
    daily_sales.values,
    label="Daily Sales",
    alpha=0.35
)

plt.plot(
    moving_avg.index,
    moving_avg.values,
    label="7-Day Moving Average",
    linewidth=2
)

plt.title("Retail Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

# 7. Luu bieu do
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "daily_sales_trend.png",
    dpi=160
)

plt.close()

print("Chart saved successfully!")
