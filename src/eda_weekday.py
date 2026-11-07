from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1. Xac dinh duong dan
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

# 2. Doc du lieu
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"],
    usecols=["date", "sales"]
)

# 3. Tinh tong so luong ban ra moi ngay
daily_sales = (
    df.groupby("date", as_index=False)["sales"]
    .sum()
)

# 4. Xac dinh thu trong tuan
# 0 = Monday, 6 = Sunday
daily_sales["weekday"] = daily_sales["date"].dt.dayofweek

# 5. Tinh trung binh ban ra theo tung thu
weekday_avg = (
    daily_sales.groupby("weekday")["sales"]
    .mean()
    .reindex(range(7))
)

weekday_names = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

# 6. Hien thi ket qua
print("=== WEEKDAY SALES ANALYSIS ===")

for day, value in zip(weekday_names, weekday_avg):
    print(f"{day}: {value:,.2f} units")

# 7. Ve bieu do
plt.figure(figsize=(10, 5))

plt.bar(
    weekday_names,
    weekday_avg.values
)

plt.title("Average Daily Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Units Sold per Day")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

# 8. Luu bieu do
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "weekday_sales.png",
    dpi=160
)

plt.close()

print("\nWeekday chart saved successfully!")
