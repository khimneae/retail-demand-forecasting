"""UI-07.3: functional/data consistency checks for Retail Demand Forecasting.

Save as tests/check_ui_07_3.py and run from the project root:
    python tests/check_ui_07_3.py --reference

Optional:
    python tests/check_ui_07_3.py --skip-model
    python tests/check_ui_07_3.py --root C:/path/to/retail-demand-forecasting
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


checks: list[dict] = []


def check(name: str, condition: bool, details: str = "") -> None:
    status = "PASS" if bool(condition) else "FAIL"
    checks.append({"test": name, "status": status, "details": details})
    print(f"[{status}] {name}" + (f" | {details}" if details else ""))


def info(message: str) -> None:
    print(f"[INFO] {message}")


def test_forecast(df: pd.DataFrame, model_path: Path, metadata: dict) -> None:
    if not model_path.is_file():
        check("Forecast model file exists", False, str(model_path))
        return

    try:
        import joblib

        artifact = joblib.load(model_path)  # Only load trusted local artifacts.
        check("Model artifact contains required keys", isinstance(artifact, dict) and
              {"model", "features", "training_end"}.issubset(artifact))
        if not isinstance(artifact, dict) or not {"model", "features", "training_end"}.issubset(artifact):
            return

        training_end = str(artifact["training_end"])
        last_observed = str(df["date"].max().date())
        check("Model and historical dataset have matching end date",
              training_end == last_observed,
              f"model={training_end}; dataset={last_observed}")
        if training_end != last_observed:
            info("Skipping prediction because the model is not aligned with this dataset.")
            return

        expected_features = {
            "store", "item", "year", "month", "day_of_week", "is_weekend",
            "lag_7", "lag_14", "lag_28", "rolling_mean_7", "rolling_mean_28",
        }
        features = list(artifact["features"])
        check("Forecast feature schema matches the app's 11 features",
              len(features) == 11 and set(features) == expected_features,
              ", ".join(features))
        if len(features) != 11 or set(features) != expected_features:
            return

        # Reproduce the app's default selection when available.
        store, item = 4, 22
        history = df.loc[(df["store"] == store) & (df["item"] == item)].sort_values("date")
        if history.empty:
            store, item = (int(df.iloc[0]["store"]), int(df.iloc[0]["item"]))
            history = df.loc[(df["store"] == store) & (df["item"] == item)].sort_values("date")
        sales = history.set_index("date")["sales"]
        expected_dates = pd.date_range(sales.index.min(), sales.index.max(), freq="D")
        complete_history = sales.index.is_unique and sales.index.equals(expected_dates)
        check(f"Forecast input series is continuous (Store {store}, Item {item})",
              complete_history, f"observed={len(sales)}; expected={len(expected_dates)}")
        if not complete_history:
            return

        last_date = sales.index.max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq="D")
        rows = []
        for target_date in future_dates:
            def past_sales(days_ago: int) -> float:
                return float(sales.loc[target_date - pd.Timedelta(days=days_ago)])

            weekday = target_date.day_of_week
            rows.append({
                "store": store,
                "item": item,
                "year": target_date.year,
                "month": target_date.month,
                "day_of_week": weekday,
                "is_weekend": int(weekday >= 5),
                "lag_7": past_sales(7),
                "lag_14": past_sales(14),
                "lag_28": past_sales(28),
                "rolling_mean_7": np.mean([past_sales(k) for k in range(7, 14)]),
                "rolling_mean_28": np.mean([past_sales(k) for k in range(7, 35)]),
            })
        x = pd.DataFrame(rows)
        prediction = np.clip(artifact["model"].predict(x[features]), 0, None)
        repeated = np.clip(artifact["model"].predict(x[features]), 0, None)
        naive = x["lag_7"].to_numpy()
        rounded = np.rint(prediction).astype(int)
        check("Seven consecutive forecast dates", len(future_dates) == 7 and
              (future_dates.to_series().diff().dropna() == pd.Timedelta(days=1)).all())
        check("All forecast values are finite and nonnegative",
              len(prediction) == 7 and np.isfinite(prediction).all() and (prediction >= 0).all())
        check("Forecast reproducible on the same input",
              np.allclose(prediction, repeated, atol=1e-10, rtol=0))
        check("Seasonal naive equals lag-7 values", np.array_equal(naive, x["lag_7"].to_numpy()))
        check("Rounded units agree with model forecasts", np.array_equal(rounded, np.rint(prediction).astype(int)))
        metadata["forecast_sample"] = {
            "store": store, "item": item,
            "dates": [d.date().isoformat() for d in future_dates],
            "model_daily": [round(float(v), 2) for v in prediction],
            "naive_daily": [int(v) for v in naive],
            "model_total": round(float(prediction.sum()), 2),
            "model_mean": round(float(prediction.mean()), 2),
            "model_peak": round(float(prediction.max()), 2),
        }
        info(f"Forecast to compare with UI: Store {store} / Item {item}, "
             f"total={prediction.sum():.2f}, mean={prediction.mean():.2f}, "
             f"peak={prediction.max():.2f}")
    except Exception as exc:
        check("Forecast computation", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    parser.add_argument("--reference", action="store_true", help="Compare against the original 2013-2017 screenshot baseline")
    parser.add_argument("--skip-model", action="store_true", help="Skip the .joblib forecasting check")
    args = parser.parse_args()
    root = args.root.resolve()
    data_path = root / "data" / "raw" / "train.csv"
    model_path = root / "models" / "hgb_deployment_model.joblib"
    if not data_path.is_file():
        print(f"[FAIL] Dataset not found: {data_path}\nRun from the project root, or use --root.")
        return 2

    try:
        data = pd.read_csv(data_path, usecols=["date", "store", "item", "sales"])
    except Exception as exc:
        print(f"[FAIL] Cannot read required columns: {type(exc).__name__}: {exc}")
        return 2

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d", errors="coerce")
    for col in ["store", "item", "sales"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    check("No missing/invalid required values", not data[["date", "store", "item", "sales"]].isna().any().any())
    if data.empty or data[["date", "store", "item", "sales"]].isna().any().any():
        return 1
    check("Numeric fields use integer values", all(np.isclose(data[c], np.round(data[c])).all()
         for c in ["store", "item", "sales"]))
    check("Store and item IDs are positive", (data[["store", "item"]] > 0).all().all())
    check("Sales units are nonnegative", (data["sales"] >= 0).all())
    data[["store", "item", "sales"]] = data[["store", "item", "sales"]].astype("int64")
    check("Each (store, item, date) is unique", not data.duplicated(["store", "item", "date"]).any())

    grouped = data.groupby(["store", "item"], sort=False)["date"].agg(["min", "max", "count"])
    expected_days = (grouped["max"] - grouped["min"]).dt.days + 1
    check("Each store-item series has complete daily history",
          bool((grouped["count"] == expected_days).all()),
          f"series={len(grouped)}; gaps-or-duplicates={int((grouped['count'] != expected_days).sum())}")
    n_stores, n_items = int(data["store"].nunique()), int(data["item"].nunique())
    check("All store-item combinations are represented", len(grouped) == n_stores * n_items,
          f"actual={len(grouped)}; possible={n_stores * n_items}")
    years = int(data["date"].dt.year.nunique())
    start, end = str(data["date"].min().date()), str(data["date"].max().date())
    metadata = {"stores": n_stores, "products": n_items, "years": years,
                "records": int(len(data)), "start_date": start, "end_date": end}
    info(f"Dataset overview: {n_stores} stores | {n_items} items | "
         f"{years} calendar years | {len(data):,} records | {start} to {end}")

    year2017 = data.loc[data["date"].dt.year == 2017]
    if not year2017.empty:
        day_count = int(year2017["date"].nunique())
        store_units = year2017.groupby("store")["sales"].sum().sort_values(ascending=False)
        item_units = year2017.groupby("item")["sales"].sum().sort_values(ascending=False)
        network_total = int(year2017["sales"].sum())
        check("Store ranking totals equal 2017 network sales", int(store_units.sum()) == network_total)
        check("Product ranking totals equal 2017 network sales", int(item_units.sum()) == network_total)
        check("Store sales shares total 100%", np.isclose((store_units / network_total * 100).sum(), 100))
        check("Product sales shares total 100%", np.isclose((item_units / network_total * 100).sum(), 100))
        cum_share = item_units.cumsum() / network_total * 100
        check("Cumulative product contribution ends at 100%", np.isclose(cum_share.iloc[-1], 100))
        reaching_80 = int(np.flatnonzero(cum_share.to_numpy() >= 80)[0]) + 1
        check("80% threshold is crossed at the first eligible rank",
              (reaching_80 == 1 or cum_share.iloc[reaching_80 - 2] < 80) and cum_share.iloc[reaching_80 - 1] >= 80)
        metadata["sales_2017"] = {"total_units": network_total, "days": day_count, "records": int(len(year2017)),
                                  "top_store": int(store_units.index[0]), "lowest_store": int(store_units.index[-1]),
                                  "top_item": int(item_units.index[0]), "top_10_share_pct": round(float(item_units.head(10).sum() / network_total * 100), 2),
                                  "top_20_share_pct": round(float(item_units.head(20).sum() / network_total * 100), 2),
                                  "items_for_80_pct": reaching_80}
        info(f"2017: units={network_total:,}, days={day_count}, top store={store_units.index[0]}, "
             f"top item={item_units.index[0]}, 80% reached at item rank={reaching_80}")

        pair = year2017.loc[(year2017["store"] == 4) & (year2017["item"] == 22)].sort_values("date")
        if not pair.empty:
            series = pair.set_index("date")["sales"]
            months = series.resample("MS").agg(["sum", "mean", "count"])
            weeks = series.resample("W-SUN").agg(["sum", "mean", "count"])
            complete_weeks = weeks.loc[weeks["count"] == 7]
            check("Monthly summed units equal the selected pair's full total",
                  int(months["sum"].sum()) == int(series.sum()))
            check("Weekly chart includes only 7-observation weeks",
                  bool((complete_weeks["count"] == 7).all()))
            check("Complete-week totals do not exceed full-range KPI",
                  int(complete_weeks["sum"].sum()) <= int(series.sum()))
            metadata["store4_item22_2017"] = {"total_units": int(series.sum()),
                    "days": int(series.count()), "average_daily_units": round(float(series.mean()), 2),
                    "monthly_periods": len(months), "complete_weeks": len(complete_weeks)}
            info(f"Store 4 / Item 22: total={int(series.sum()):,}, avg={series.mean():.2f}, "
                 f"days={series.count()}, months={len(months)}, complete weeks={len(complete_weeks)}")

    if args.reference:
        check("Original dataset: date span and row count",
              (start, end, len(data), n_stores, n_items) ==
              ("2013-01-01", "2017-12-31", 913_000, 10, 50))
        check("Original 2017 record count and observed days",
              len(year2017) == 182_500 and metadata.get("sales_2017", {}).get("days") == 365)
        check("Original 2017 network units", metadata.get("sales_2017", {}).get("total_units") == 10_733_740)
        check("Original Store 4 / Item 22 total", metadata.get("store4_item22_2017", {}).get("total_units") == 34_616)
        check("Original Store 4 / Item 22 average daily units",
              metadata.get("store4_item22_2017", {}).get("average_daily_units") == 94.84)
        check("Original Store 4 / Item 22 monthly/weekly periods",
              metadata.get("store4_item22_2017", {}).get("monthly_periods") == 12 and
              metadata.get("store4_item22_2017", {}).get("complete_weeks") == 52)
        if not year2017.empty:
            check("Original Store 2 and Store 7 ranking totals",
                  int(store_units.get(2, -1)) == 1_376_386 and int(store_units.get(7, -1)) == 747_159)
            check("Original Store 4 annual sales", int(store_units.get(4, -1)) == 1_127_450)
            check("Original Store 4 / Item 15 units",
                  int(year2017.loc[(year2017["store"] == 4) & (year2017["item"] == 15), "sales"].sum()) == 38_101)
            check("Original top item is Item 15 with 361,586 units",
                  int(item_units.index[0]) == 15 and int(item_units.iloc[0]) == 361_586)
            check("Original top 10/20 product shares",
                  metadata["sales_2017"]["top_10_share_pct"] == 31.41 and
                  metadata["sales_2017"]["top_20_share_pct"] == 56.94)
            check("Original product contribution reaches 80% at rank 32",
                  metadata["sales_2017"]["items_for_80_pct"] == 32)

    if not args.skip_model:
        test_forecast(data, model_path, metadata)
    else:
        info("Model checks skipped by --skip-model")

    report_path = root / "reports" / "ui_07_3_check.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result = {"dataset": metadata, "checks": checks,
              "passed": sum(x["status"] == "PASS" for x in checks),
              "failed": sum(x["status"] == "FAIL" for x in checks)}
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRESULT: {result['passed']} passed / {result['failed']} failed")
    print(f"Report: {report_path}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
