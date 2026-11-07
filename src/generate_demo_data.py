
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    """
    Generate an independent synthetic retail dataset.

    This script does not read or reproduce Kaggle data.
    The dataset is intended only for dashboard demonstrations.
    """

    root = Path(__file__).resolve().parents[1]
    output_path = root / "data" / "demo" / "synthetic_sales.csv"

    rng = np.random.default_rng(42)

    dates = pd.date_range(
        start="2013-01-01",
        end="2017-12-31",
        freq="D",
    )

    # 10 stores x 50 items = 500 daily combinations
    combinations = 10 * 50
    total_rows = len(dates) * combinations

    index = np.arange(total_rows)

    day_indices = index // combinations
    stores = (index % combinations) // 50 + 1
    items = (index % 50) + 1

    row_dates = dates.take(day_indices)

    weekdays = row_dates.dayofweek.to_numpy()
    day_of_year = row_dates.dayofyear.to_numpy()
    years = row_dates.year.to_numpy()

    # Arbitrary synthetic demand patterns
    base_demand = (12 + items * 0.85) * (
        0.72 + stores * 0.06
    )

    weekend_effect = np.where(
        weekdays >= 5, 1.17, 1.0
    )

    annual_seasonality = (
        1.0
        + 0.10 * np.sin(
            2 * np.pi * day_of_year / 365.25
        )
    )

    yearly_growth = 1.0 + 0.02 * (years - 2013)

    expected_sales = (
        base_demand
        * weekend_effect
        * annual_seasonality
        * yearly_growth
    )

    expected_sales = np.clip(expected_sales, 1, None)

    synthetic_sales = rng.poisson(
        expected_sales
    ).astype("int32")

    df = pd.DataFrame({
        "date": row_dates.strftime("%Y-%m-%d"),
        "store": stores.astype("int16"),
        "item": items.astype("int16"),
        "sales": synthetic_sales,
    })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output_path, index=False)

    print("Synthetic demo dataset created.")
    print(f"Path: {output_path}")
    print(f"Records: {len(df):,}")
    print(f"Stores: {df['store'].nunique()}")
    print(f"Items: {df['item'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("Source: independently generated synthetic data")


if __name__ == "__main__":
    main()
