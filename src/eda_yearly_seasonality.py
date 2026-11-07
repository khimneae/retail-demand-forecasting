from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"

# 1. Read sales data
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# 2. Calculate daily total sales
daily_sales = df.groupby("date", as_index=False)["sales"].sum()

# 3. Extract year and month
daily_sales["year"] = daily_sales["date"].dt.year
daily_sales["month"] = daily_sales["date"].dt.month

# 4. Average daily sales by year and month
monthly = (
    daily_sales.groupby(["year", "month"])["sales"]
    .mean()
    .unstack("year")
)

# 5. Print the highest sales month for each year
print("=== PEAK SALES MONTH BY YEAR ===")
for year in monthly.columns:
    peak_month = monthly[year].idxmax()
    peak_sales = monthly[year].max()
    print(f"{year}: Month {peak_month} - {peak_sales:,.2f} units/day")

# 6. Plot one line per year
plt.figure(figsize=(11, 6))

for year in monthly.columns:
    plt.plot(
        monthly.index,
        monthly[year],
        marker="o",
        label=str(year)
    )

plt.title("Monthly Sales Seasonality by Year")
plt.xlabel("Month")
plt.ylabel("Average Daily Units Sold")
plt.xticks(range(1, 13))
plt.legend(title="Year")
plt.grid(alpha=0.25)
plt.tight_layout()

# 7. Save chart
OUTPUT_DIR = ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "yearly_seasonality.png",
    dpi=160
)

plt.close()

print("\nChart saved successfully!")
