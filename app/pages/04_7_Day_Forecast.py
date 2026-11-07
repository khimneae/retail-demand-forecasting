from pathlib import Path

import sys

# Make the repository root available for imports on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_source import get_data_source

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="7-Day Demand Forecast",
    page_icon="📈",
    layout="wide",
)

ROOT = Path(__file__).resolve().parents[2]

# Automatically select Kaggle or synthetic demo data.
DATA_PATH, IS_DEMO = get_data_source()

# Keep the existing deployment model.
MODEL_PATH = (
    ROOT / "models" / "hgb_deployment_model.joblib"
)
# ==================================================
# SYNTHETIC DATA DISCLOSURE
# ==================================================

if IS_DEMO:
    st.warning(
        "**Synthetic Forecast Demonstration** — "
        "Predictions use independently generated sales history "
        "with a model originally trained on Kaggle data. "
        "These predictions are illustrative only and have "
        "not been validated on the synthetic dataset. "
        "Published MAE, RMSE, and WAPE metrics refer "
        "exclusively to the original Kaggle evaluation."
    )

# ==================================================
# 2. LOAD DATA AND MODEL
# ==================================================

@st.cache_data(show_spinner="Loading historical sales...")
def load_sales_data():
    return pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        dtype={
            "store": "int16",
            "item": "int16",
            "sales": "int32",
        },
    )


@st.cache_resource
def load_forecast_model():
    return joblib.load(MODEL_PATH)


# ==================================================
# UI-07.4 - RESPONSIVE FORECAST OVERRIDES
# ==================================================
# app/main.py loads the premium theme from assets/retail.css.
# Load the shared responsive CSS here too (even if main.py has not yet
# been updated). Page-specific overrides are then applied last.
RESPONSIVE_CSS_PATH = ROOT / "app" / "assets" / "responsive.css"
if RESPONSIVE_CSS_PATH.is_file():
    st.html(RESPONSIVE_CSS_PATH)

st.html("""
<style>
/* Prevent long text or child grids from expanding the page width. */
.ri-forecast-hero,
.ri-forecast-hero-content,
.ri-forecast-hero-status,
.ri-forecast-kpi-grid,
.ri-forecast-kpi-card,
.ri-forecast-export-grid,
.ri-forecast-export-cell {
    min-width: 0;
    box-sizing: border-box;
}

@media (max-width: 1024px) {
    .ri-forecast-hero {
        display: grid !important;
        grid-template-columns: minmax(0, 1fr) !important;
        gap: 1rem !important;
        padding: 1.5rem !important;
    }
    .ri-forecast-hero-status {
        width: 100% !important;
        max-width: 100% !important;
    }
    .ri-forecast-kpi-grid {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 0.9rem !important;
    }
    .ri-forecast-kpi-grid > * {
        grid-column: auto !important;
    }
    .ri-forecast-export-grid {
        display: grid !important;
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        gap: 0.75rem !important;
    }
}

@media (max-width: 640px) {
    .ri-forecast-hero {
        padding: 1.1rem !important;
        border-radius: 14px !important;
    }
    .ri-forecast-hero h1 {
        font-size: clamp(1.65rem, 7vw, 2rem) !important;
        line-height: 1.2 !important;
        overflow-wrap: anywhere;
    }
    .ri-forecast-hero-badges {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 0.4rem !important;
    }
    .ri-forecast-hero-badges span {
        max-width: 100%;
        white-space: normal;
    }
    .ri-forecast-status-date {
        overflow-wrap: anywhere;
    }
    .ri-forecast-controls-footer {
        display: flex !important;
        flex-wrap: wrap !important;
        gap: 0.4rem !important;
    }
    .ri-forecast-controls-separator {
        display: none !important;
    }
    .ri-forecast-kpi-grid,
    .ri-forecast-export-grid {
        grid-template-columns: minmax(0, 1fr) !important;
        gap: 0.75rem !important;
    }
    .ri-forecast-kpi-card {
        padding: 1rem !important;
    }
    .ri-forecast-kpi-value {
        font-size: clamp(1.65rem, 8vw, 2.2rem) !important;
        overflow-wrap: anywhere;
    }
    .ri-forecast-kpi-bottom {
        flex-wrap: wrap !important;
        gap: 0.45rem !important;
    }
    .ri-panel-title {
        font-size: clamp(1.15rem, 5vw, 1.5rem) !important;
        overflow-wrap: anywhere;
    }
    [data-testid="stDataFrame"] {
        min-width: 0 !important;
        max-width: 100% !important;
    }
}

@media (prefers-reduced-motion: reduce) {
    .ri-forecast-kpi-card, .ri-forecast-hero, .ri-forecast-export-cell {
        animation: none !important;
        transition: none !important;
    }
}
</style>
""")


# ==================================================
# 3. PREMIUM PAGE HEADER
# ==================================================

st.html("""
<section class="ri-forecast-hero">

    <div class="ri-forecast-hero-content">

        <div class="ri-forecast-eyebrow">
            RETAIL INTELLIGENCE / AI FORECASTING
        </div>

        <h1>7-Day Demand Forecast</h1>

        <p>
            Generate machine learning demand predictions
            for individual stores and products.
            Explore estimated future sales patterns
            using historical retail data.
        </p>

        <div class="ri-forecast-hero-badges">

            <span>7-Day Forecast Horizon</span>

            <span>HistGradientBoosting</span>

            <span>Historical Sales: 2013-2017</span>

        </div>

    </div>

    <div class="ri-forecast-hero-status">

        <div class="ri-forecast-status-label">
            FORECAST WINDOW
        </div>

        <div class="ri-forecast-status-date">
            01 - 07 Jan 2018
        </div>

        <div class="ri-forecast-status-detail">
            Last observed: 31 Dec 2017
        </div>

        <div class="ri-forecast-status-chip">
            <span class="ri-forecast-status-dot"></span>
            Historical Simulation
        </div>

    </div>

</section>
""")


# ==================================================
# 4. PREMIUM FORECAST CONFIGURATION
# ==================================================

with st.container(
    border=False,
    key="ri_forecast_controls",
):

    st.html("""
    <div class="ri-forecast-controls-heading">

        <div class="ri-forecast-section-label">
            01 / FORECAST CONFIGURATION
        </div>

        <h2>Configure Your Forecast</h2>

        <p>
            Select a store and product to generate
            a seven-day demand forecast.
        </p>

    </div>
    """)

    # Keep the existing form workflow
    with st.form(
        "forecast_form",
        clear_on_submit=False,
        border=False,
    ):

        col1, col2 = st.columns(
            2,
            gap="medium",
        )

        with col1:

            selected_store = st.selectbox(
                "Select Store",
                options=list(range(1, 11)),
                index=3,
                format_func=lambda x: f"Store {x}",
                key="ri_forecast_store",
            )

        with col2:

            selected_item = st.selectbox(
                "Select Item",
                options=list(range(1, 51)),
                index=21,
                format_func=lambda x: f"Item {x}",
                key="ri_forecast_item",
            )

        generate = st.form_submit_button(
            label="Generate 7-Day Forecast",
            type="primary",
            icon=":material/auto_awesome:",
            width="stretch",
        )

    st.html("""
    <div class="ri-forecast-controls-footer">

        <span class="ri-forecast-controls-dot"></span>

        <span>
            Historical data available through
            31 Dec 2017
        </span>

        <span class="ri-forecast-controls-separator">
            |
        </span>

        <span>
            Forecast horizon: 7 days
        </span>

    </div>
    """)


# ==================================================
# 5. GENERATE FORECAST
# ==================================================
# A rerun (e.g. clicking a download button) must not silently change
# the store/item associated with the most recently generated forecast.
if generate:
    st.session_state["ri_forecast_request"] = (
        int(selected_store),
        int(selected_item),
    )

forecast_request = st.session_state.get("ri_forecast_request")

if forecast_request is not None:
    selected_store, selected_item = forecast_request

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        st.error("Historical data or deployment model not found.")
        st.stop()

    artifact = load_forecast_model()

    if artifact.get("training_end") != "2017-12-31":
        st.error("Unexpected deployment model version.")
        st.stop()

    model = artifact["model"]
    feature_names = artifact["features"]

    df = load_sales_data()

    history = df.loc[
        (df["store"] == selected_store)
        & (df["item"] == selected_item)
    ].sort_values("date")

    if history.empty:
        st.error("No historical data found.")
        st.stop()

    sales = history.set_index("date")["sales"]

    expected_index = pd.date_range(
        sales.index.min(),
        sales.index.max(),
        freq="D",
    )

    if (
        not sales.index.is_unique
        or not sales.index.equals(expected_index)
    ):
        st.error("Historical sales series is incomplete.")
        st.stop()

    last_date = sales.index.max()

    if last_date != pd.Timestamp("2017-12-31"):
        st.error("Unexpected historical end date.")
        st.stop()

    # Generate 7 forecast dates
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=7,
        freq="D",
    )

    feature_rows = []

    for target_date in future_dates:

        def past_sales(days_ago):
            date = target_date - pd.Timedelta(
                days=days_ago
            )
            return float(sales.loc[date])

        weekday = target_date.dayofweek

        feature_rows.append({
            "store": selected_store,
            "item": selected_item,
            "year": target_date.year,
            "month": target_date.month,
            "day_of_week": weekday,
            "is_weekend": int(weekday >= 5),
            "lag_7": past_sales(7),
            "lag_14": past_sales(14),
            "lag_28": past_sales(28),
            "rolling_mean_7": np.mean([
                past_sales(k)
                for k in range(7, 14)
            ]),
            "rolling_mean_28": np.mean([
                past_sales(k)
                for k in range(7, 35)
            ]),
        })

    X_future = pd.DataFrame(feature_rows)

    # Predict using the deployment model
    predictions = np.clip(
        model.predict(X_future[feature_names]),
        0,
        None,
    )

    baseline = X_future["lag_7"].to_numpy()

    forecast = pd.DataFrame({
        "Date": future_dates,
        "ML Forecast": np.round(predictions, 2),
        "Seasonal Naive": baseline,
        "Rounded Units": np.rint(predictions).astype(int),
    })
    # ==================================================
    # 6. PREMIUM FORECAST RESULTS - UI-06.2
    # ==================================================

    st.divider()

    # Calculate display metrics from model predictions
    total_forecast = float(predictions.sum())
    average_forecast = float(predictions.mean())
    peak_forecast = float(predictions.max())

    peak_index = int(np.argmax(predictions))
    lowest_index = int(np.argmin(predictions))

    peak_date = future_dates[peak_index].strftime(
        "%d %b %Y"
    )

    lowest_date = future_dates[lowest_index].strftime(
        "%d %b %Y"
    )

    forecast_start = future_dates[0].strftime(
        "%d %b %Y"
    )

    forecast_end = future_dates[-1].strftime(
        "%d %b %Y"
    )

    # Premium section heading
    st.html(f"""
    <div class="ri-forecast-results-heading">

        <div class="ri-forecast-results-eyebrow">
            02 / FORECAST RESULTS
        </div>

        <h2>Forecast Results</h2>

        <p>
            Store {selected_store}
            <span class="ri-result-separator">/</span>
            Item {selected_item}
            <span class="ri-result-separator">/</span>
            {forecast_start} - {forecast_end}
        </p>

    </div>
    """)

    # Premium KPI cards
    st.html(f"""
    <div class="ri-forecast-kpi-grid">

        <!-- TOTAL FORECAST -->

        <div class="ri-forecast-kpi-card
                    ri-forecast-kpi-total">

            <div class="ri-forecast-kpi-top">

                <span class="ri-forecast-kpi-label">
                    TOTAL FORECASTED UNITS
                </span>

                <span class="ri-forecast-kpi-index">
                    01
                </span>

            </div>

            <div class="ri-forecast-kpi-value">
                {total_forecast:,.2f}
            </div>

            <div class="ri-forecast-kpi-unit">
                Units across 7 forecast days
            </div>

            <div class="ri-forecast-kpi-bottom">

                <span class="ri-forecast-kpi-pill">
                    7-Day Forecast
                </span>

                <span class="ri-forecast-kpi-detail">
                    Estimated demand
                </span>

            </div>

        </div>


        <!-- AVERAGE DAILY FORECAST -->

        <div class="ri-forecast-kpi-card
                    ri-forecast-kpi-average">

            <div class="ri-forecast-kpi-top">

                <span class="ri-forecast-kpi-label">
                    AVERAGE DAILY FORECAST
                </span>

                <span class="ri-forecast-kpi-index">
                    02
                </span>

            </div>

            <div class="ri-forecast-kpi-value">
                {average_forecast:,.2f}
            </div>

            <div class="ri-forecast-kpi-unit">
                Predicted units per day
            </div>

            <div class="ri-forecast-kpi-bottom">

                <span class="ri-forecast-kpi-pill">
                    Daily Average
                </span>

                <span class="ri-forecast-kpi-detail">
                    7-day mean
                </span>

            </div>

        </div>


        <!-- PEAK DAILY FORECAST -->

        <div class="ri-forecast-kpi-card
                    ri-forecast-kpi-peak">

            <div class="ri-forecast-kpi-top">

                <span class="ri-forecast-kpi-label">
                    PEAK DAILY FORECAST
                </span>

                <span class="ri-forecast-kpi-index">
                    03
                </span>

            </div>

            <div class="ri-forecast-kpi-value">
                {peak_forecast:,.2f}
            </div>

            <div class="ri-forecast-kpi-unit">
                Highest predicted daily demand
            </div>

            <div class="ri-forecast-kpi-bottom">

                <span class="ri-forecast-kpi-pill">
                    {peak_date}
                </span>

                <span class="ri-forecast-kpi-detail">
                    Peak demand day
                </span>

            </div>

        </div>

    </div>
    """)

    st.caption(
        "Forecasts are model estimates based on "
        "historical sales through 31 Dec 2017. "
        f"Lowest predicted demand: "
        f"{predictions[lowest_index]:,.2f} units "
        f"on {lowest_date}."
    )


    # ==================================================
    # 7. UI-06.3 - PREMIUM DAILY FORECAST TABLE
    # ==================================================

    display_forecast = forecast.copy()
    display_forecast["Date"] = (
        display_forecast["Date"].dt.strftime("%Y-%m-%d")
    )

    display_table = display_forecast.copy()

    display_table.insert(
        1,
        "Day",
        forecast["Date"].dt.strftime("%a").to_numpy(),
    )

    display_table["Difference vs Naive"] = (
        display_table["ML Forecast"]
        - display_table["Seasonal Naive"]
    ).round(2)

    display_table = display_table[
        [
            "Date",
            "Day",
            "ML Forecast",
            "Seasonal Naive",
            "Difference vs Naive",
            "Rounded Units",
        ]
    ]

    with st.container(border=True, key="ri_forecast_table"):

        st.html("""
        <div class="ri-panel-eyebrow">
            03 / DAILY DEMAND OUTLOOK
        </div>
        <div class="ri-panel-title">
            Daily Forecast Table
        </div>
        <p class="ri-panel-description">
            Explore daily model predictions, benchmark estimates,
            and the difference between the two forecasting methods.
        </p>
        """)

        st.dataframe(
            display_table,
            hide_index=True,
            width="stretch",
            height="content",
            row_height=36,
            column_config={
                "Date": st.column_config.TextColumn(
                    "Forecast Date",
                    width=120,
                    pinned=True,
                ),
                "Day": st.column_config.TextColumn(
                    "Day",
                    width="small",
                ),
                "ML Forecast": st.column_config.NumberColumn(
                    "ML Forecast",
                    format="%.2f",
                    help="Predicted demand in units.",
                ),
                "Seasonal Naive": st.column_config.NumberColumn(
                    "Seasonal Naive",
                    format="%d",
                    help="Sales observed seven days earlier.",
                ),
                "Difference vs Naive": st.column_config.NumberColumn(
                    "Difference vs Naive",
                    format="%+.2f",
                    help="ML forecast minus seasonal naive estimate.",
                ),
                "Rounded Units": st.column_config.NumberColumn(
                    "Rounded Units",
                    format="%d",
                    help="ML prediction rounded to whole units.",
                ),
            },
        )

        st.caption(
            "On smaller screens, scroll the table horizontally to see "
            "all columns. Positive differences mean the ML estimate "
            "exceeds the Seasonal Naive estimate. These differences "
            "do not measure forecasting accuracy."
        )

    # ==================================================
    # 8. UI-07.4 - RESPONSIVE FORECAST VISUALIZATION
    # ==================================================

    chart_data = forecast.melt(
        id_vars="Date",
        value_vars=["ML Forecast", "Seasonal Naive"],
        var_name="Model",
        value_name="Units",
    )

    model_colors = alt.Scale(
        domain=["ML Forecast", "Seasonal Naive"],
        range=["#38BDF8", "#3B82F6"],
    )

    base_chart = (
        alt.Chart(chart_data)
        .encode(
            x=alt.X(
                "Date:T",
                title="Forecast Date",
                axis=alt.Axis(
                    format="%d %b",
                    tickCount=7,
                    labelAngle=0,
                    labelOverlap="greedy",
                    labelLimit=60,
                    labelPadding=8,
                    labelFlush=False,
                    grid=False,
                    labelColor="#B8CBE4",
                    titleColor="#B8CBE4",
                ),
            ),
            y=alt.Y(
                "Units:Q",
                title="Forecasted Units",
                scale=alt.Scale(zero=False, nice=True),
                axis=alt.Axis(
                    gridColor="#293C55",
                    labelColor="#B8CBE4",
                    titleColor="#B8CBE4",
                    tickCount=6,
                ),
            ),
            color=alt.Color(
                "Model:N",
                scale=model_colors,
                legend=alt.Legend(
                    orient="top",
                    direction="horizontal",
                    title=None,
                    labelColor="#D8E5F5",
                    labelLimit=140,
                    labelFontSize=11,
                ),
            ),
            tooltip=[
                alt.Tooltip(
                    "Date:T",
                    title="Forecast Date",
                    format="%d %b %Y",
                ),
                alt.Tooltip(
                    "Model:N",
                    title="Forecast Method",
                ),
                alt.Tooltip(
                    "Units:Q",
                    title="Predicted Units",
                    format=",.2f",
                ),
            ],
        )
    )

    comparison_chart = (
        alt.layer(
            base_chart.mark_line(strokeWidth=3),
            base_chart.mark_circle(size=65),
        )
        .properties(height=300)
        .configure_view(stroke=None)
    )

    # Difference chart
    delta_data = forecast[
        ["Date", "ML Forecast", "Seasonal Naive"]
    ].copy()

    delta_data["Difference"] = (
        delta_data["ML Forecast"]
        - delta_data["Seasonal Naive"]
    )

    delta_data["Comparison"] = np.where(
        delta_data["Difference"] >= 0,
        "ML above naive",
        "ML below naive",
    )

    delta_chart = (
        alt.Chart(delta_data)
        .mark_bar(
            size=26,
            cornerRadius=3,
        )
        .encode(
            x=alt.X(
                "Date:T",
                title=None,
                axis=alt.Axis(
                    format="%d %b",
                    tickCount=7,
                    labelAngle=0,
                    labelOverlap="greedy",
                    labelLimit=60,
                    labelPadding=8,
                    labelFlush=False,
                    grid=False,
                    labelColor="#B8CBE4",
                ),
            ),
            y=alt.Y(
                "Difference:Q",
                title="Difference (Units)",
                scale=alt.Scale(zero=True, nice=True),
                axis=alt.Axis(
                    gridColor="#293C55",
                    labelColor="#B8CBE4",
                    titleColor="#B8CBE4",
                ),
            ),
            color=alt.Color(
                "Comparison:N",
                scale=alt.Scale(
                    domain=[
                        "ML above naive",
                        "ML below naive",
                    ],
                    range=["#34D399", "#F59E0B"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(
                    "Date:T",
                    title="Date",
                    format="%d %b %Y",
                ),
                alt.Tooltip(
                    "Difference:Q",
                    title="ML - Naive",
                    format="+.2f",
                ),
            ],
        )
        .properties(height=160)
    )

    with st.container(border=True, key="ri_forecast_chart"):

        st.html("""
        <div class="ri-panel-eyebrow">
            04 / FORECAST ANALYTICS
        </div>
        <div class="ri-panel-title">
            7-Day Forecast Comparison
        </div>
        <p class="ri-panel-description">
            Compare the machine learning forecast with the
            seasonal naive benchmark over the same seven-day period.
        </p>
        """)

        st.altair_chart(
            comparison_chart,
            width="stretch",
            theme=None,
        )

        st.markdown("**Daily Difference vs Seasonal Naive**")

        st.altair_chart(
            delta_chart,
            width="stretch",
            theme=None,
        )

        ml_total = float(forecast["ML Forecast"].sum())
        naive_total = float(forecast["Seasonal Naive"].sum())
        total_difference = ml_total - naive_total

        if total_difference >= 0:
            comparison_label = "above"
        else:
            comparison_label = "below"

        st.caption(
            f"Across seven days, the ML estimate is "
            f"{abs(total_difference):,.2f} units "
            f"{comparison_label} the Seasonal Naive estimate. "
            "This is a comparison of two predictions, "
            "not a measured accuracy difference."
        )

    # ==================================================
    # 9. UI-06.3 - PREMIUM FORECAST EXPORT
    # ==================================================

    export_df = display_forecast.copy()
    export_df.insert(0, "Store", int(selected_store))
    export_df.insert(1, "Item", int(selected_item))

    csv_data = export_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    start_label = future_dates[0].strftime("%Y-%m-%d")
    end_label = future_dates[-1].strftime("%Y-%m-%d")

    filename = (
        f"forecast_store_{selected_store}_"
        f"item_{selected_item}_"
        f"{start_label}_{end_label}.csv"
    )

    with st.container(border=True, key="ri_forecast_export"):

        st.html(f"""
        <div class="ri-panel-eyebrow">
            05 / EXPORT REPORT
        </div>
        <div class="ri-panel-title">
            Export Forecast Analysis
        </div>
        <p class="ri-panel-description">
            Download the seven-day demand forecast
            for further analysis or reporting.
        </p>

        <div class="ri-forecast-export-grid">

            <div class="ri-forecast-export-cell">
                <span>STORE</span>
                <strong>Store {selected_store}</strong>
            </div>

            <div class="ri-forecast-export-cell">
                <span>PRODUCT</span>
                <strong>Item {selected_item}</strong>
            </div>

            <div class="ri-forecast-export-cell">
                <span>FORECAST PERIOD</span>
                <strong>01 - 07 Jan 2018</strong>
            </div>

            <div class="ri-forecast-export-cell">
                <span>REPORT RECORDS</span>
                <strong>{len(export_df)} Days</strong>
            </div>

        </div>
        """)

        st.download_button(
            label="Download 7-Day Forecast (CSV)",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            width="stretch",
            type="primary",
            icon=":material/download:",
            on_click="ignore",
        )

        st.caption(
            "CSV | UTF-8 with BOM | Microsoft Excel compatible"
        )

    # ==================================================
    # 10. EVALUATION DISCLOSURE
    # ==================================================

    st.html("""
    <div class="ri-forecast-disclaimer">
        <strong>Historical Forecast Simulation</strong>
        <p>
            This simulation predicts demand for 01-07 January 2018
            using historical sales available through 31 December 2017.
            Actual sales for the forecast period are not available
            in this dataset. Forecasts cannot be evaluated against
            ground truth.
        </p>
    </div>
    """)

else:
    st.html("""
    <div style="
        margin-top: 22px;
        padding: 25px;
        background: #111F33;
        border: 1px dashed #355476;
        border-radius: 14px;
        text-align: center;
    ">

        <div style="
            color: #F8FAFC;
            font-size: 17px;
            font-weight: 750;
            margin-bottom: 10px;
        ">
            Ready to Generate Forecast
        </div>

        <div style="
            color: #9CB8D8;
            font-size: 13px;
            line-height: 1.7;
        ">
            Select your store and product, then click
            Generate 7-Day Forecast to view demand
            predictions for 01 - 07 January 2018.
        </div>

    </div>
    """)
