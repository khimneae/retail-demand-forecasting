from pathlib import Path
import json

import altair as alt
import pandas as pd
import streamlit as st


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Retail Demand Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROJECT PATHS & DATA LOADING
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = (
    ROOT / "reports" / "metrics" / "final_test_results.json"
)

if not METRICS_PATH.exists():
    st.error(
        "Final test results not found. "
        "Please run src/evaluate_final_test.py first."
    )
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    results = json.load(file)

baseline = results["baseline"]
ml = results["machine_learning"]
improvement = results["mae_improvement_percent"]

# =========================================================
# 4. MAIN DASHBOARD HEADER
# =========================================================


st.html("""
<section class="ri-hero">
    <div class="ri-hero-eyebrow">
        RETAIL INTELLIGENCE / MODEL PERFORMANCE
    </div>

    <h1 class="ri-hero-title">
        Retail Demand Forecasting
    </h1>

    <p class="ri-hero-subtitle">
        AI-powered demand intelligence for retail operations.
        Explore forecasting performance, evaluate predictive
        accuracy, and transform historical sales data into
        actionable business insights.
    </p>

    <div class="ri-hero-tags">
        <span class="ri-hero-tag">
            2017 Holdout Evaluation
        </span>

        <span class="ri-hero-tag">
            182,500 Test Observations
        </span>

        <span class="ri-hero-tag">
            HistGradientBoosting
        </span>
    </div>
</section>
""")


st.divider()

st.header("Model Performance Overview")

st.write(
    "Final evaluation on **182,500 store-item-day observations** "
    "from the 2017 test dataset."
)


# =========================================================
# PREMIUM MODEL PERFORMANCE KPI CARDS
# ==================================================

# Differences compared with Seasonal Naive baseline
mae_delta = ml["mae"] - baseline["mae"]
rmse_delta = ml["rmse"] - baseline["rmse"]

# WAPE differences are percentage points
wape_delta = (
    ml["wape_percent"] - baseline["wape_percent"]
)

st.html(f"""
<div class="ri-kpi-grid">

    <div class="ri-kpi-card">
        <div class="ri-kpi-header">
            <span>MEAN ABSOLUTE ERROR</span>
            <span class="ri-kpi-symbol">◈</span>
        </div>

        <div class="ri-kpi-value">
            {ml["mae"]:.4f}
        </div>

        <span class="ri-kpi-change">
            {mae_delta:+.4f} vs Baseline
        </span>

        <div class="ri-kpi-description">
            MAE · Lower values are better
        </div>
    </div>

    <div class="ri-kpi-card">
        <div class="ri-kpi-header">
            <span>ROOT MEAN SQUARED ERROR</span>
            <span class="ri-kpi-symbol">◈</span>
        </div>

        <div class="ri-kpi-value">
            {ml["rmse"]:.4f}
        </div>

        <span class="ri-kpi-change">
            {rmse_delta:+.4f} vs Baseline
        </span>

        <div class="ri-kpi-description">
            RMSE · Penalizes larger errors
        </div>
    </div>

    <div class="ri-kpi-card">
        <div class="ri-kpi-header">
            <span>WEIGHTED ABSOLUTE PERCENT ERROR</span>
            <span class="ri-kpi-symbol">◈</span>
        </div>

        <div class="ri-kpi-value">
            {ml["wape_percent"]:.2f}%
        </div>

        <span class="ri-kpi-change">
            {wape_delta:+.2f} pp vs Baseline
        </span>

        <div class="ri-kpi-description">
            WAPE · Sales-weighted error rate
        </div>
    </div>

    <div class="ri-kpi-card featured">
        <div class="ri-kpi-header">
            <span>MAE IMPROVEMENT</span>
            <span class="ri-kpi-symbol">↗</span>
        </div>

        <div class="ri-kpi-value">
            {improvement:.2f}%
        </div>

        <span class="ri-kpi-change">
            Relative MAE reduction
        </span>

        <div class="ri-kpi-description">
            HistGradientBoosting vs Seasonal Naive
        </div>
    </div>

</div>
""")

st.caption(
    "Changes are measured against the Seasonal Naive (Lag 7) "
    "baseline. Lower error values are better."
)

# =========================================================
# 6. BASELINE VS MACHINE LEARNING TABLE
# =========================================================

st.subheader("Baseline vs Machine Learning")

comparison = pd.DataFrame(
    {
        "Model": [
            "Seasonal Naive (Lag 7)",
            "HistGradientBoosting",
        ],
        "MAE": [
            baseline["mae"],
            ml["mae"],
        ],
        "RMSE": [
            baseline["rmse"],
            ml["rmse"],
        ],
        "WAPE (%)": [
            baseline["wape_percent"],
            ml["wape_percent"],
        ],
    }
)

st.dataframe(
    comparison.round(4),
    hide_index=True,
    width="stretch",
)


st.subheader("Forecast Error Breakdown")

st.caption(
    "Comparison against Seasonal Naive (Lag 7). "
    "Lower prediction errors indicate better performance."
)

metrics_config = [
    ("MAE", "mae", "Units"),
    ("RMSE", "rmse", "Units"),
    ("WAPE", "wape_percent", "Percent"),
]

chart_columns = st.columns(3, gap="medium")

for column, (title, metric_key, unit) in zip(
    chart_columns, metrics_config
):

    chart_df = pd.DataFrame({
        "Method": ["Baseline", "ML Model"],
        "Value": [
            baseline[metric_key],
            ml[metric_key],
        ],
    })

    upper_bound = chart_df["Value"].max() * 1.2

    common = alt.Chart(chart_df).encode(
        y=alt.Y(
            "Method:N",
            sort=["Baseline", "ML Model"],
            title=None,
        ),
        x=alt.X(
            "Value:Q",
            title=None,
            scale=alt.Scale(
                domain=[0, upper_bound]
            ),
            axis=alt.Axis(
                tickCount=4,
                format=".1f",
            ),
        ),
    )

    bars = common.mark_bar(
        size=28,
        cornerRadiusEnd=6,
    ).encode(
        color=alt.Color(
            "Method:N",
            scale=alt.Scale(
                domain=["Baseline", "ML Model"],
                range=["#3B82F6", "#38BDF8"],
            ),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("Method:N"),
            alt.Tooltip(
                "Value:Q",
                format=".4f",
            ),
        ],
    )

    labels = common.mark_text(
        align="left",
        dx=5,
        fontSize=11,
        color="#F8FAFC",
    ).encode(
        text=alt.Text(
            "Value:Q",
            format=".2f",
        )
    )

    chart = (
        (bars + labels)
        .properties(height=125)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#CBD5E1",
            gridColor="#29364D",
            domain=False,
        )
    )

    with column:
        with st.container(border=True):
            st.markdown(f"**{title} Comparison**")

            st.altair_chart(
                chart,
                width="stretch",
            )

            st.caption(
                f"{unit} | Baseline vs ML Model"
            )



# =========================================================
# 8. EVALUATION NOTES
# =========================================================


# ==========================================
# EVALUATION METHODOLOGY
# ==========================================

st.divider()

st.subheader("Evaluation Methodology")

method_col1, method_col2, method_col3 = st.columns(
    3, gap="medium"
)

with method_col1:
    with st.container(border=True):
        st.caption("TEST PERIOD")
        st.markdown("### 2017")
        st.caption("Jan 01 - Dec 31")

with method_col2:
    with st.container(border=True):
        st.caption("TEST OBSERVATIONS")
        st.markdown(f"### {results['samples']:,}")
        st.caption("Store-item-day records")

with method_col3:
    with st.container(border=True):
        st.caption("FORECAST HORIZON")
        st.markdown(
            f"### {results['forecast_horizon_days']} Days"
        )
        st.caption("Historical evaluation")

st.info(
    "**Rolling-Origin Evaluation**\n\n"
    "Results are based on historical evaluation using "
    "observed sales history available at prediction time. "
    "The model does not use the actual sales value "
    "of the predicted date."
)

st.caption(
    "Model: HistGradientBoostingRegressor | "
    "Evaluation: 2017 Test Dataset | "
    "Historical experimental results"
)

st.caption(
    "This dashboard is not a live production "
    "forecasting system."
)
