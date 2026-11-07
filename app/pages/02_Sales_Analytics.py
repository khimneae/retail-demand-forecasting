from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


# 1. Page configuration
st.set_page_config(
    page_title="Sales Analytics | Retail Intelligence",
    page_icon="📊",
    layout="wide",
)


# 2. Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "train.csv"


# 3. Load data with caching
@st.cache_data(show_spinner="Loading sales data...")
def load_data():
    return pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
        dtype={
            "store": "int16",
            "item": "int16",
            "sales": "int32",
        },
    )


if not DATA_PATH.exists():
    st.error("Dataset not found: data/raw/train.csv")
    st.stop()

df = load_data()


# 4. Page header

# ==========================================
# PREMIUM SALES ANALYTICS HEADER
# ==========================================

st.html("""
<section class="ri-hero">

    <div class="ri-hero-eyebrow">
        RETAIL INTELLIGENCE / SALES ANALYTICS
    </div>

    <h1 class="ri-hero-title">
        Sales Intelligence & Analytics
    </h1>

    <p class="ri-hero-subtitle">
        Explore historical retail demand, analyze
        sales performance across stores and products,
        identify seasonal patterns, and discover
        actionable business insights.
    </p>

    <div class="ri-hero-tags">

        <span class="ri-hero-tag">
            Historical Data: 2013-2017
        </span>

        <span class="ri-hero-tag">
            10 Stores / 50 Products
        </span>

        <span class="ri-hero-tag">
            Interactive Sales Analysis
        </span>

    </div>

</section>
""")


st.divider()


# 5. Interactive filters

# =========================================================
# 5. PREMIUM SALES ANALYSIS CONTROLS
# =========================================================

with st.container(border=True, key="ri_sales_filters"):

    st.markdown("#### Sales Analysis Controls")

    st.caption(
        "Configure your historical demand analysis "
        "by store, product, and date range."
    )

    # Store and item selectors
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        selected_store = st.selectbox(
            "Select Store",
            options=sorted(df["store"].unique()),
            format_func=lambda x: f"Store {x}",
            key="ri_sales_store",
        )

    with col2:
        selected_item = st.selectbox(
            "Select Item",
            options=sorted(df["item"].unique()),
            format_func=lambda x: f"Item {x}",
            key="ri_sales_item",
        )

    # Historical date range
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()

    date_range = st.date_input(
        "Analysis Period",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD",
        key="ri_sales_date_range",
    )

# =========================================================
# 6. DATA FILTERING - PRESERVE BUSINESS LOGIC
# =========================================================

if len(date_range) != 2:
    st.info(
        "Please select both a start date and an end date."
    )
    st.stop()

start_date, end_date = date_range

filtered = df.loc[
    (df["store"] == selected_store)
    & (df["item"] == selected_item)
    & (df["date"] >= pd.Timestamp(start_date))
    & (df["date"] <= pd.Timestamp(end_date))
].sort_values("date")

if filtered.empty:
    st.warning(
        "No sales records found for this selection."
    )
    st.stop()

# =========================================================
# 7. PREMIUM SALES PERFORMANCE KPI CARDS
# =========================================================

st.subheader("Sales Overview")

st.caption(
    f"Store {selected_store} | "
    f"Item {selected_item} | "
    f"{start_date:%d %b %Y} - "
    f"{end_date:%d %b %Y}"
)

# Preserve original KPI calculations
total_units = filtered["sales"].sum()
average_daily_units = filtered["sales"].mean()
historical_days = len(filtered)

st.html(f"""
<div class="ri-sales-kpi-grid">

    <div class="ri-sales-kpi-card">
        <div class="ri-sales-kpi-top">
            <span>TOTAL UNITS SOLD</span>
            <span class="ri-sales-kpi-icon">∑</span>
        </div>

        <div class="ri-sales-kpi-value">
            {total_units:,.0f}
        </div>

        <div class="ri-sales-kpi-footer">
            Total units in selected analysis period
        </div>
    </div>

    <div class="ri-sales-kpi-card">
        <div class="ri-sales-kpi-top">
            <span>AVERAGE DAILY UNITS</span>
            <span class="ri-sales-kpi-icon">↗</span>
        </div>

        <div class="ri-sales-kpi-value">
            {average_daily_units:,.2f}
        </div>

        <div class="ri-sales-kpi-footer">
            Average units sold per observed day
        </div>
    </div>

    <div class="ri-sales-kpi-card">
        <div class="ri-sales-kpi-top">
            <span>HISTORICAL DAYS</span>
            <span class="ri-sales-kpi-icon">▦</span>
        </div>

        <div class="ri-sales-kpi-value">
            {historical_days:,}
        </div>

        <div class="ri-sales-kpi-footer">
            Daily observations in selected period
        </div>
    </div>

</div>
""")

# =========================================================
# 8. HISTORICAL SALES TREND
# Keep all existing code below this point unchanged.
# =========================================================

# =========================================================
# 8. TIME GRANULARITY FILTER

# =========================================================
# 8. PREMIUM HISTORICAL SALES TREND
# =========================================================

with st.container(border=True, key="ri_sales_trend_panel"):

    # Header and time controls
    header_col, control_col = st.columns(
        [2.5, 1.8],
        gap="medium",
        vertical_alignment="center",
    )

    with header_col:
        st.markdown("### Historical Sales Trend")
        st.caption(
            "Explore historical demand patterns "
            "at different time granularities."
        )

    with control_col:
        granularity = st.segmented_control(
            "Time Granularity",
            options=["Daily", "Weekly", "Monthly"],
            default="Monthly",
            selection_mode="single",
            required=True,
            key="ri_sales_granularity",
            width="stretch",
        )

    # =====================================================
    # 8.1. AGGREGATION - KEEP ORIGINAL BUSINESS LOGIC
    # =====================================================

    frequency_map = {
        "Daily": "D",
        "Weekly": "W-SUN",
        "Monthly": "MS",
    }

    frequency = frequency_map[granularity]

    trend_data = (
        filtered.set_index("date")["sales"]
        .resample(frequency)
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "average_daily_units",
                "count": "days_in_period",
            }
        )
    )

    # Exclude incomplete weeks
    if granularity == "Weekly":
        trend_data = trend_data.loc[
            trend_data["days_in_period"] == 7
        ].copy()

    # Remove periods without observations
    trend_data = trend_data.dropna(
        subset=["average_daily_units"]
    ).copy()

    if trend_data.empty:
        st.warning(
            "No complete periods available. "
            "Please choose a wider date range."
        )
        st.stop()

    # =====================================================
    # 8.2. DISPLAY LABELS
    # =====================================================

    if granularity == "Monthly":
        trend_data["month_label"] = (
            trend_data["date"].dt.strftime("%Y-%m")
        )

        trend_data["period_label"] = (
            trend_data["date"].dt.strftime("%b %Y")
        )

        period_title = "Month"
        axis_title = "Month"

    elif granularity == "Weekly":
        trend_data["period_label"] = (
            trend_data["date"].dt.strftime("%d %b %Y")
        )

        period_title = "Week Ending (Sunday)"
        axis_title = "Week Ending (Sunday)"

    else:
        trend_data["period_label"] = (
            trend_data["date"].dt.strftime("%d %b %Y")
        )

        period_title = "Date"
        axis_title = "Date"

    # Adaptive date labels
    selected_days = (end_date - start_date).days + 1

    if granularity == "Monthly":
        date_format = "%b %Y"
        tick_count = 6

    elif selected_days <= 90:
        date_format = "%d %b"
        tick_count = 6

    else:
        date_format = "%b %Y"
        tick_count = 8

    x_axis = alt.X(
    "date:T",
    title=axis_title,
    axis=alt.Axis(
        format=date_format,
        tickCount=tick_count,
        labelAngle=0,
        labelOverlap=True,
        labelLimit=95,
    ),
)

    base = alt.Chart(trend_data).encode(

        x=x_axis,

    y=alt.Y(
        "average_daily_units:Q",
        title="Average Daily Units Sold",
        scale=alt.Scale(
            zero=False,
            nice=True,
        ),
    ),
    )

    # Main trend line
    line = base.mark_line(
        color="#38BDF8",
        strokeWidth=2.8,
        interpolate="linear",
    )

    # Hover selection: nearest historical observation
    nearest = alt.selection_point(
        nearest=True,
        on="pointerover",
        fields=["date"],
        empty=False,
    )

    # Transparent points detect cursor position
    selectors = (
        alt.Chart(trend_data)
    .mark_point()
    .encode(
        x="date:T",
        opacity=alt.value(0),
        tooltip=[
            alt.Tooltip(
                "period_label:N",
                title=period_title,
            ),
            alt.Tooltip(
                "average_daily_units:Q",
                title="Average Units / Day",
                format=".2f",
            ),
            alt.Tooltip(
                "days_in_period:Q",
                title="Observed Days",
                format=".0f",
            ),
        ],
    )
        .add_params(nearest)
    )

    # Highlight the selected observation
    hover_points = base.mark_circle(
        size=90,
        color="#E0F2FE",
    ).encode(
        opacity=alt.condition(
            nearest,
            alt.value(1),
            alt.value(0),
        )
    )

    # Tooltip with source data
    hover_rules = (
        alt.Chart(trend_data)
        .mark_rule(
            color="#64748B",
            strokeWidth=1,
            strokeDash=[4, 4],
        )
        .encode(
            x="date:T",
            tooltip=[
                alt.Tooltip(
                    "period_label:N",
                    title=period_title,
                ),
                alt.Tooltip(
                    "average_daily_units:Q",
                    title="Average Units / Day",
                    format=".2f",
                ),
                alt.Tooltip(
                    "days_in_period:Q",
                    title="Observed Days",
                    format=".0f",
                ),
            ],
        )
        .transform_filter(nearest)
    )

    # Build chart layers
    layers = [line]

    # Show small markers for aggregated periods
    if granularity != "Daily":
        layers.append(
            base.mark_circle(
                size=32,
                color="#7DD3FC",
                opacity=0.9,
            )
        )

    layers.extend([
        selectors,
        hover_points,
        hover_rules,
    ])

    chart = (
        alt.layer(*layers)
        .properties(height=345)
        .configure_view(
            stroke=None,
        )
        .configure_axis(
            labelColor="#9FB2C9",
            titleColor="#CBD5E1",
            gridColor="#26364B",
            gridOpacity=0.65,
            domain=False,
            tickColor="#334155",
            labelFontSize=11,
            titleFontSize=12,
        )
        .configure_axisX(
            grid=False,
        )
    )

    st.altair_chart(
        chart,
        width="stretch",
    )

    # =====================================================
    # 8.4. ANALYSIS INFORMATION
    # =====================================================

    if granularity == "Weekly":
        st.caption(
            f"Displaying {len(trend_data):,} complete weeks. "
            "Incomplete weeks are excluded. "
            "KPI totals still cover the entire "
            "selected date range."
        )
    else:
        st.caption(
            f"Displaying {len(trend_data):,} "
            f"{granularity.lower()} observations. "
            "Values represent average daily units "
            "within each period."
        )

# BUSINESS INSIGHTS
# =========================================================

st.divider()

# =========================================================
# UI-04.4 - PREMIUM BUSINESS INSIGHTS
# =========================================================

from html import escape

st.subheader("Business Insights")

# 1. Extract aggregated sales values
values = trend_data["average_daily_units"]

# 2. Identify highest and lowest periods
peak_row = trend_data.loc[values.idxmax()]
lowest_row = trend_data.loc[values.idxmin()]

peak_value = float(peak_row["average_daily_units"])
lowest_value = float(lowest_row["average_daily_units"])

# 3. Format period labels
def format_period(date_value):
    date_value = pd.Timestamp(date_value)

    if granularity == "Monthly":
        return date_value.strftime("%b %Y")

    if granularity == "Weekly":
        week_start = date_value - pd.Timedelta(days=6)
        return (
            f"{week_start:%d %b} - "
            f"{date_value:%d %b %Y}"
        )

    return date_value.strftime("%d %b %Y")


peak_period = format_period(peak_row["date"])
lowest_period = format_period(lowest_row["date"])

# 4. Calculate coefficient of variation
# Preserve original calculation and ddof=0
mean_sales = values.mean()

if len(values) >= 2 and mean_sales > 0:
    cv = values.std(ddof=0) / mean_sales * 100
else:
    cv = None

# 5. Prepare display values
cv_display = f"{cv:.2f}%" if cv is not None else "N/A"

if cv is not None:
    cv_description = (
        f"Across the displayed {granularity.lower()} "
        f"periods, the coefficient of variation "
        f"is {cv:.2f}%."
    )
else:
    cv_description = (
        "Demand variability is unavailable because "
        "at least two periods with a positive average "
        "are required."
    )

# Escape dynamic text used inside HTML
safe_store = escape(str(selected_store))
safe_item = escape(str(selected_item))
safe_peak_period = escape(peak_period)
safe_lowest_period = escape(lowest_period)
safe_cv_description = escape(cv_description)

# =========================================================
# 6. PREMIUM KPI CARDS AND BUSINESS INTERPRETATION
# =========================================================

st.html(f"""
<div class="ri-insight-grid">

    <article class="ri-insight-card">
        <div class="ri-insight-top">
            <span>PEAK DEMAND PERIOD</span>
            <span class="ri-insight-icon ri-insight-green">
                ↗
            </span>
        </div>

        <div class="ri-insight-value">
            {peak_value:,.2f}
        </div>

        <div class="ri-insight-unit">
            Average units / day
        </div>

        <div class="ri-insight-period ri-insight-green">
            {safe_peak_period}
        </div>

        <div class="ri-insight-footnote">
            Highest average among displayed periods
        </div>
    </article>

    <article class="ri-insight-card">
        <div class="ri-insight-top">
            <span>LOWEST DEMAND PERIOD</span>
            <span class="ri-insight-icon ri-insight-amber">
                ↘
            </span>
        </div>

        <div class="ri-insight-value">
            {lowest_value:,.2f}
        </div>

        <div class="ri-insight-unit">
            Average units / day
        </div>

        <div class="ri-insight-period ri-insight-amber">
            {safe_lowest_period}
        </div>

        <div class="ri-insight-footnote">
            Lowest average among displayed periods
        </div>
    </article>

    <article class="ri-insight-card">
        <div class="ri-insight-top">
            <span>DEMAND VARIABILITY (CV)</span>
            <span class="ri-insight-icon ri-insight-cyan">
                ~
            </span>
        </div>

        <div class="ri-insight-value">
            {cv_display}
        </div>

        <div class="ri-insight-unit">
            Coefficient of variation
        </div>

        <div class="ri-insight-period ri-insight-cyan">
            {granularity} analysis
        </div>

        <div class="ri-insight-footnote">
            Variation across displayed period averages
        </div>
    </article>

</div>

<section class="ri-insight-summary">

    <div class="ri-insight-summary-head">
        <h4>Business Interpretation</h4>
        <span>
            Store {safe_store} / Item {safe_item}
        </span>
    </div>

    <div class="ri-insight-row">

        <span class="ri-insight-marker ri-insight-green">
            01
        </span>

        <p>
            The highest average daily sales occurred
            during <strong>{safe_peak_period}</strong>,
            reaching
            <strong>{peak_value:,.2f} units/day</strong>.
        </p>

    </div>

    <div class="ri-insight-row">

        <span class="ri-insight-marker ri-insight-amber">
            02
        </span>

        <p>
            The lowest average daily sales occurred
            during <strong>{safe_lowest_period}</strong>,
            with
            <strong>{lowest_value:,.2f} units/day</strong>.
        </p>

    </div>

    <div class="ri-insight-row">

        <span class="ri-insight-marker ri-insight-cyan">
            03
        </span>

        <p>
            {safe_cv_description}
        </p>

    </div>

    <div class="ri-insight-note">
        These insights describe historical sales patterns,
        not future demand predictions.
        Partial monthly periods use only observed days
        within the selected date range.
    </div>

</section>
""")

# =========================================================
# EXPORT SALES ANALYTICS REPORT
# =========================================================

st.divider()

# =========================================================
# UI-04.5 - PREMIUM SALES REPORT EXPORT CENTER
# =========================================================

from html import escape

st.subheader("Export Sales Analysis")

st.caption(
    "Download aggregated historical sales data "
    "for the selected store, item, date range, "
    "and time granularity."
)

# =========================================================
# 1. PREPARE EXPORT DATASET
# Preserve existing business logic
# =========================================================

export_df = trend_data[
    ["date", "days_in_period", "average_daily_units"]
].copy()

# 2. Add business metadata
export_df["store"] = int(selected_store)
export_df["item"] = int(selected_item)
export_df["granularity"] = granularity

# 3. Add readable period labels
export_df["period"] = export_df["date"].apply(
    format_period
)

# 4. Calculate total units per period
export_df["total_units_sold"] = (
    export_df["average_daily_units"]
    * export_df["days_in_period"]
).round().astype(int)

# 5. Preserve the original CSV column structure
export_df = export_df[
    [
        "store",
        "item",
        "granularity",
        "period",
        "days_in_period",
        "total_units_sold",
        "average_daily_units",
    ]
]

export_df["average_daily_units"] = (
    export_df["average_daily_units"].round(2)
)

# 6. Convert to Excel-compatible CSV
csv_data = export_df.to_csv(
    index=False
).encode("utf-8-sig")

# 7. File metadata
report_rows = len(export_df)

report_filename = (
    f"sales_store_{selected_store}_"
    f"item_{selected_item}_"
    f"{granularity.lower()}_"
    f"{start_date}_{end_date}.csv"
)

# Escape dynamic text used in HTML
safe_store = escape(str(selected_store))
safe_item = escape(str(selected_item))
safe_granularity = escape(str(granularity))

safe_period = escape(
    f"{start_date:%d %b %Y} - "
    f"{end_date:%d %b %Y}"
)

# =========================================================
# 8. PREMIUM EXPORT CENTER
# =========================================================

with st.container(
    border=True,
    key="ri_export_center",
):

    st.html(f"""
    <div class="ri-export-content">

        <div class="ri-export-header">

            <div class="ri-export-header-left">

                <div class="ri-export-icon">
                    &#8681;
                </div>

                <div>
                    <div class="ri-export-title">
                        Sales Analysis Report
                    </div>

                    <div class="ri-export-subtitle">
                        Historical sales data export
                    </div>
                </div>

            </div>

            <span class="ri-export-status">
                READY TO EXPORT
            </span>

        </div>

        <div class="ri-export-metadata">

            <div class="ri-export-meta-item">
                <span class="ri-export-meta-label">
                    SELECTION
                </span>
                <span class="ri-export-meta-value">
                    Store {safe_store} / Item {safe_item}
                </span>
            </div>

            <div class="ri-export-meta-item">
                <span class="ri-export-meta-label">
                    GRANULARITY
                </span>
                <span class="ri-export-meta-value">
                    {safe_granularity}
                </span>
            </div>

            <div class="ri-export-meta-item">
                <span class="ri-export-meta-label">
                    ANALYSIS PERIOD
                </span>
                <span class="ri-export-meta-value">
                    {safe_period}
                </span>
            </div>

            <div class="ri-export-meta-item">
                <span class="ri-export-meta-label">
                    REPORT RECORDS
                </span>
                <span class="ri-export-meta-value">
                    {report_rows:,}
                </span>
            </div>

        </div>

        <div class="ri-export-description">
            Download your aggregated sales analysis
            as a CSV file, including total units sold,
            average daily units, period information
            and selected store and product identifiers.
        </div>

    </div>
    """)

    # Native Streamlit download button
    st.download_button(
        label="Download Sales Report (CSV)",
        data=csv_data,
        file_name=report_filename,
        mime="text/csv",
        type="primary",
        icon=":material/download:",
        width="stretch",
        on_click="ignore",
        key="ri_sales_export_download",
    )

    st.caption(
        "CSV format | UTF-8 with BOM | "
        "Compatible with Microsoft Excel"
    )

    # Explain the complete-week policy
    if granularity == "Weekly":
        st.caption(
            "Weekly exports contain complete weeks "
            "only. Export totals may differ from "
            "the full date-range KPI totals."
        )

# =========================================================
# 9. HISTORICAL ANALYSIS DISCLAIMER
# =========================================================

st.info(
    f"Historical sales analysis for "
    f"Store {selected_store}, Item {selected_item}. "
    f"The chart uses {granularity.lower()} granularity. "
    "These values represent historical observations, "
    "not future forecasts."
)
