from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Xac dinh duong dan du an
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

# 2. Doc du lieu
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"],
    usecols=["date", "sales"]
)

# 3. Tinh tong luong ban theo tung ngay
daily_sales = (
    df.groupby("date", as_index=False)["sales"]
    .sum()
)

# 4. Lay thang tu cot ngay
daily_sales["month"] = daily_sales["date"].dt.month

# 5. Tinh trung binh luong ban moi ngay theo thang
monthly_avg = (
    daily_sales.groupby("month")["sales"]
    .mean()
    .reindex(range(1, 13))
)

month_names = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

# 6. In ket qua
print("=== MONTHLY SALES ANALYSIS ===")

for month, value in zip(month_names, monthly_avg):
    print(f"{month}: {value:,.2f} units/day")

print("\nHighest month:",
      month_names[monthly_avg.idxmax() - 1])

print("Lowest month:",
      month_names[monthly_avg.idxmin() - 1])

# 7. Ve bieu do
plt.figure(figsize=(11, 5))

plt.bar(
    month_names,
    monthly_avg.values,
    color="steelblue"
)

plt.title("Average Daily Sales by Month")
plt.xlabel("Month")
plt.ylabel("Average Units Sold per Day")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

# 8. Luu bieu do
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "monthly_sales.png",
    dpi=160
)

plt.close()

print("\nMonthly chart saved successfully!")
