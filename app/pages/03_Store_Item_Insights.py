from pathlib import Path
from datetime import date
import sys

# Make the repository root available for imports on Streamlit Cloud.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_source import get_data_source
import pandas as pd
import altair as alt
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Store & Item Insights",
    page_icon="🏬",
    layout="wide",
)


# ==================================================
# 2. LOAD DATA
# ==================================================

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH, IS_DEMO = get_data_source()


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
    st.error("Dataset not found.")
    st.stop()

df = load_data()



# ==================================================
# UI-05.1 - PREMIUM STORE & ITEM HEADER
# ==================================================

# Calculate dataset metadata dynamically
store_count = df["store"].nunique()
item_count = df["item"].nunique()
record_count = len(df)

data_start = df["date"].min()
data_end = df["date"].max()

history_period = (
    f"{data_start:%Y} - {data_end:%Y}"
)

# Premium header
st.html(f"""
<section class="ri-store-hero">

    <div class="ri-store-hero-eyebrow">
        RETAIL INTELLIGENCE / STORE & ITEM INSIGHTS
    </div>

    <h1>Store & Item Insights</h1>

    <p class="ri-store-hero-description">
        Analyze store performance, compare product sales,
        and uncover demand concentration across your
        retail network. Transform historical sales data
        into actionable business intelligence.
    </p>

    <div class="ri-store-hero-badges">

        <span class="ri-store-hero-badge">
            Historical Data: {history_period}
        </span>

        <span class="ri-store-hero-badge">
            {store_count:,} Stores
        </span>

        <span class="ri-store-hero-badge">
            {item_count:,} Products
        </span>

        <span class="ri-store-hero-badge">
            {record_count:,} Records
        </span>

    </div>

</section>
""")

st.divider()


# ==================================================
# UI-05.2 - PREMIUM STORE SALES RANKING
# ==================================================

st.header("Store Sales Ranking")

st.caption(
    "Compare historical sales performance across stores, "
    "identify sales leaders, and evaluate market share "
    "by units sold."
)

# ==================================================
# 4. PREMIUM DATE FILTER
# ==================================================

with st.container(
    border=True,
    key="ri_store_filter_panel",
):
    st.markdown("#### Analysis Period")

    st.caption(
        "Select the historical period used "
        "for store and product analysis."
    )

    date_range = st.date_input(
        "Select Date Range",
        value=(date(2017, 1, 1), date(2017, 12, 31)),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
        format="YYYY-MM-DD",
        key="ri_store_date_range",
    )

if len(date_range) != 2:
    st.info("Please select a start and end date.")
    st.stop()

start_date, end_date = date_range

if start_date > end_date:
    st.error("Start date must not exceed end date.")
    st.stop()


# ==================================================
# 5. FILTER DATA - ORIGINAL LOGIC
# ==================================================

filtered = df.loc[
    df["date"].between(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    )
].copy()

if filtered.empty:
    st.warning("No data found.")
    st.stop()

observed_days = filtered["date"].nunique()


# ==================================================
# 6. STORE AGGREGATION - ORIGINAL LOGIC
# ==================================================

ranking = (
    filtered.groupby("store", as_index=False)
    .agg(
        total_units=("sales", "sum"),
        observations=("sales", "size"),
    )
)

ranking["average_daily_units"] = (
    ranking["total_units"] / observed_days
)

total_sales = int(ranking["total_units"].sum())

if total_sales > 0:
    ranking["sales_share_percent"] = (
        ranking["total_units"] / total_sales * 100
    )
else:
    ranking["sales_share_percent"] = 0.0

ranking = ranking.sort_values(
    ["total_units", "store"],
    ascending=[False, True],
).reset_index(drop=True)

ranking.insert(
    0,
    "rank",
    range(1, len(ranking) + 1),
)

ranking["store_name"] = ranking["store"].apply(
    lambda value: f"Store {value}"
)


# ==================================================
# 7. PREMIUM STORE OVERVIEW KPIs
# ==================================================

st.subheader("Store Overview")

st.caption(
    f"Retail network performance | "
    f"{start_date:%d %b %Y} - "
    f"{end_date:%d %b %Y}"
)

st.html(f"""
<div class="ri-store-kpi-grid">

    <div class="ri-store-kpi-card">
        <div class="ri-store-kpi-header">
            <span>STORES ANALYZED</span>
            <span class="ri-store-kpi-icon">▦</span>
        </div>

        <div class="ri-store-kpi-value">
            {len(ranking):,}
        </div>

        <div class="ri-store-kpi-footer">
            Stores with recorded sales in the period
        </div>
    </div>

    <div class="ri-store-kpi-card">
        <div class="ri-store-kpi-header">
            <span>TOTAL UNITS SOLD</span>
            <span class="ri-store-kpi-icon">∑</span>
        </div>

        <div class="ri-store-kpi-value">
            {total_sales:,}
        </div>

        <div class="ri-store-kpi-footer">
            Combined sales volume across all stores
        </div>
    </div>

    <div class="ri-store-kpi-card">
        <div class="ri-store-kpi-header">
            <span>HISTORICAL DAYS</span>
            <span class="ri-store-kpi-icon">◷</span>
        </div>

        <div class="ri-store-kpi-value">
            {observed_days:,}
        </div>

        <div class="ri-store-kpi-footer">
            Unique observed days in selected period
        </div>
    </div>

</div>
""")


# ==================================================
# 8. PREMIUM STORE RANKING TABLE
# ==================================================

display_df = ranking[
    [
        "rank",
        "store_name",
        "total_units",
        "average_daily_units",
        "sales_share_percent",
    ]
].copy()

display_df.columns = [
    "Rank",
    "Store",
    "Total Units Sold",
    "Average Units / Day",
    "Sales Share (%)",
]

display_df[
    ["Average Units / Day", "Sales Share (%)"]
] = display_df[
    ["Average Units / Day", "Sales Share (%)"]
].round(2)

with st.container(
    border=True,
    key="ri_store_table_panel",
):
    st.markdown("#### Store Ranking Table")

    st.caption(
        "Stores ranked by total units sold. "
        "Select a column header to sort the table."
    )

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Rank": st.column_config.NumberColumn(
                "Rank",
                format="%d",
                width="small",
            ),
            "Store": st.column_config.TextColumn(
                "Store",
                width="medium",
            ),
            "Total Units Sold": (
                st.column_config.NumberColumn(
                    "Total Units Sold",
                    format="%,d",
                )
            ),
            "Average Units / Day": (
                st.column_config.NumberColumn(
                    "Average Units / Day",
                    format="%,.2f",
                )
            ),
            "Sales Share (%)": (
                st.column_config.NumberColumn(
                    "Sales Share (%)",
                    format="%.2f",
                )
            ),
        },
    )


# ==================================================
# 9. PREMIUM STORE SALES CHART
# ==================================================

with st.container(
    border=True,
    key="ri_store_chart_panel",
):
    st.markdown("#### Sales Volume by Store")

    st.caption(
        "Compare total units sold across the network. "
        "The leading store is highlighted in cyan."
    )

    chart_base = alt.Chart(ranking).encode(
        y=alt.Y(
            "store_name:N",
            sort="-x",
            title=None,
            axis=alt.Axis(
                labelFontSize=12,
                labelPadding=12,
            ),
        ),
        x=alt.X(
            "total_units:Q",
            title="Total Units Sold",
            scale=alt.Scale(
                zero=True,
                nice=True,
            ),
            axis=alt.Axis(
                format="~s",
                tickCount=6,
            ),
        ),
    )

    # Highlight the highest-ranked store
    bars = (
        chart_base
        .mark_bar(
            cornerRadiusEnd=6,
            size=25,
        )
        .encode(
            color=alt.condition(
                alt.datum.rank == 1,
                alt.value("#38BDF8"),
                alt.value("#3B82F6"),
            ),
            tooltip=[
                alt.Tooltip(
                    "rank:Q",
                    title="Rank",
                    format=".0f",
                ),
                alt.Tooltip(
                    "store_name:N",
                    title="Store",
                ),
                alt.Tooltip(
                    "total_units:Q",
                    title="Total Units",
                    format=",d",
                ),
                alt.Tooltip(
                    "average_daily_units:Q",
                    title="Average Units / Day",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "sales_share_percent:Q",
                    title="Sales Share (%)",
                    format=".2f",
                ),
            ],
        )
    )

    # Show actual sales totals inside each bar
    labels = (
        chart_base
        .mark_text(
            align="right",
            baseline="middle",
            dx=-10,
            color="#FFFFFF",
            fontSize=11,
            fontWeight="bold",
        )
        .encode(
            text=alt.Text(
                "total_units:Q",
                format=",.0f",
            ),
        )
    )

    chart = (
        alt.layer(bars, labels)
        .properties(
            height=max(360, len(ranking) * 40),
        )
        .configure_view(
            stroke=None,
        )
        .configure_axis(
            labelColor="#A8BBD3",
            titleColor="#CBD5E1",
            domain=False,
            tickColor="#34465F",
        )
        .configure_axisX(
            grid=True,
            gridColor="#2B3A50",
            gridOpacity=0.5,
        )
        .configure_axisY(
            grid=False,
            ticks=False,
        )
    )

    st.altair_chart(
        chart,
        width="stretch",
    )


# ==================================================
# 10. PREMIUM BUSINESS SUMMARY
# ==================================================

st.subheader("Business Summary")

top_store = ranking.iloc[0]
lowest_store = ranking.iloc[-1]

top_store_id = int(top_store["store"])
lowest_store_id = int(lowest_store["store"])

top_units = int(top_store["total_units"])
lowest_units = int(lowest_store["total_units"])

top_share = float(top_store["sales_share_percent"])
lowest_share = float(lowest_store["sales_share_percent"])

st.html(f"""
<div class="ri-store-summary-grid">

    <div class="ri-store-summary-card">

        <div class="ri-store-summary-label">
            <span class="ri-store-summary-green">●</span>
            HIGHEST SALES VOLUME
        </div>

        <div class="ri-store-summary-name">
            Store {top_store_id}
        </div>

        <div class="ri-store-summary-value">
            {top_units:,}
            <span>units sold</span>
        </div>

        <div class="ri-store-summary-foot">
            {top_share:.2f}% of total sales volume
        </div>

    </div>

    <div class="ri-store-summary-card">

        <div class="ri-store-summary-label">
            <span class="ri-store-summary-amber">●</span>
            LOWEST SALES VOLUME
        </div>

        <div class="ri-store-summary-name">
            Store {lowest_store_id}
        </div>

        <div class="ri-store-summary-value">
            {lowest_units:,}
            <span>units sold</span>
        </div>

        <div class="ri-store-summary-foot">
            {lowest_share:.2f}% of total sales volume
        </div>

    </div>

</div>
""")

st.caption(
    "Rankings are based on units sold, not revenue "
    "or profitability. All products in the selected "
    "period are included in the store analysis."
)


# ==================================================
# UI-05.3 - PREMIUM ITEM SALES RANKING
# ==================================================

st.divider()

st.header("Item Sales Ranking")

st.caption(
    "Identify best-selling products, compare sales volume, "
    "and analyze product contributions across the selected "
    "retail network."
)

# ==================================================
# 11. PREMIUM ITEM ANALYSIS CONTROLS
# ==================================================

with st.container(
    border=True,
    key="ri_item_filter_panel",
):
    st.markdown("#### Item Analysis Controls")

    st.caption(
        "Select all stores or an individual store "
        "to analyze product sales performance."
    )

    store_options = ["All Stores"] + [
        f"Store {store_id}"
        for store_id in sorted(filtered["store"].unique())
    ]

    selected_scope = st.selectbox(
        "Select Store Scope",
        options=store_options,
        key="item_ranking_store_scope",
    )

    st.caption(
        f"Analysis Period: "
        f"{start_date:%d %b %Y} - "
        f"{end_date:%d %b %Y}"
    )

# ==================================================
# 12. FILTER ITEM DATA - PRESERVE ORIGINAL LOGIC
# ==================================================

if selected_scope == "All Stores":
    item_filtered = filtered.copy()
else:
    selected_store_id = int(selected_scope.split()[-1])

    item_filtered = filtered.loc[
        filtered["store"] == selected_store_id
    ].copy()

if item_filtered.empty:
    st.warning("No product sales found for this scope.")
    st.stop()

item_days = item_filtered["date"].nunique()

if item_days == 0:
    st.warning("No observed sales dates found.")
    st.stop()

# ==================================================
# 13. ITEM AGGREGATION - ORIGINAL LOGIC
# ==================================================

item_ranking = (
    item_filtered.groupby("item", as_index=False)
    .agg(
        total_units=("sales", "sum")
    )
)

if item_ranking.empty:
    st.warning("No products found for this selection.")
    st.stop()

scope_total_units = int(
    item_ranking["total_units"].sum()
)

item_ranking["average_daily_units"] = (
    item_ranking["total_units"] / item_days
)

if scope_total_units > 0:
    item_ranking["sales_share_percent"] = (
        item_ranking["total_units"]
        / scope_total_units * 100
    )
else:
    item_ranking["sales_share_percent"] = 0.0

item_ranking = item_ranking.sort_values(
    ["total_units", "item"],
    ascending=[False, True],
).reset_index(drop=True)

item_ranking.insert(
    0,
    "rank",
    range(1, len(item_ranking) + 1),
)

item_ranking["item_name"] = (
    "Item " + item_ranking["item"].astype(str)
)

# Extract top item information
top_item = item_ranking.iloc[0]

top_item_id = int(top_item["item"])
top_item_units = int(top_item["total_units"])
top_item_share = float(
    top_item["sales_share_percent"]
)

# Keep top_10_items available for downstream analysis
top_10_items = item_ranking.head(10).copy()

if scope_total_units > 0:
    top_10_share = (
        top_10_items["total_units"].sum()
        / scope_total_units * 100
    )
else:
    top_10_share = 0.0

# ==================================================
# 14. PREMIUM ITEM OVERVIEW
# ==================================================

st.subheader("Item Overview")

st.caption(
    f"{selected_scope} | "
    f"{start_date:%d %b %Y} - "
    f"{end_date:%d %b %Y}"
)

# Reuse the existing premium KPI card design
# from UI-05.2 for visual consistency

st.html(f"""
<div class="ri-store-kpi-grid">

    <div class="ri-store-kpi-card">
        <div class="ri-store-kpi-header">
            <span>ITEMS ANALYZED</span>
            <span class="ri-store-kpi-icon">▦</span>
        </div>

        <div class="ri-store-kpi-value">
            {len(item_ranking):,}
        </div>

        <div class="ri-store-kpi-footer">
            Products with recorded observations
        </div>
    </div>

    <div class="ri-store-kpi-card ri-item-featured-kpi">
        <div class="ri-store-kpi-header">
            <span>TOP-SELLING ITEM</span>
            <span class="ri-store-kpi-icon">↗</span>
        </div>

        <div class="ri-store-kpi-value">
            Item {top_item_id}
        </div>

        <div class="ri-store-kpi-footer">
            {top_item_units:,} units sold
        </div>
    </div>

    <div class="ri-store-kpi-card">
        <div class="ri-store-kpi-header">
            <span>TOTAL UNITS IN SCOPE</span>
            <span class="ri-store-kpi-icon">∑</span>
        </div>

        <div class="ri-store-kpi-value">
            {scope_total_units:,}
        </div>

        <div class="ri-store-kpi-footer">
            Combined sales volume in selected scope
        </div>
    </div>

</div>
""")


# ==================================================
# 15. PREMIUM ITEM RANKING TABLE
# ==================================================

item_display = item_ranking[
    [
        "rank",
        "item_name",
        "total_units",
        "average_daily_units",
        "sales_share_percent",
    ]
].copy()

item_display.columns = [
    "Rank",
    "Item",
    "Total Units Sold",
    "Average Units / Day",
    "Sales Share (%)",
]

item_display = item_display.round(2)

with st.container(
    border=True,
    key="ri_item_table_panel",
):
    st.markdown("#### Item Ranking Table")

    st.caption(
        "Products ranked by total units sold. "
        "Click a column header to sort the table."
    )
    st.dataframe(
    item_display,
    hide_index=True,
    width="stretch",
    height=360,
    row_height=34,
    column_config={
        "Rank": st.column_config.NumberColumn(
            "Rank",
            width="small",
            pinned=True,
            format="%d",
        ),
        "Item": st.column_config.TextColumn(
            "Item",
            width="small",
        ),
        "Total Units Sold": st.column_config.NumberColumn(
            "Units Sold",
            format="%,d",
        ),
        "Average Units / Day": st.column_config.NumberColumn(
            "Units / Day",
            format="%.2f",
        ),
        "Sales Share (%)": st.column_config.NumberColumn(
            "Share (%)",
            format="%.2f",
        ),
    },
)
    st.caption(
        f"Showing {len(item_ranking):,} ranked products "
        f"for {selected_scope}."
    )


# ==================================================
# 16. PREMIUM TOP 10 PRODUCTS CHART
# ==================================================

with st.container(
    border=True,
    key="ri_item_chart_panel",
):
    st.markdown("#### Top 10 Best-Selling Items")

    st.caption(
        "Compare the ten highest-selling products. "
        "The leading item is highlighted in cyan."
    )

    # Preserve the ranking order, including tie-breaks
    item_order = top_10_items["item_name"].tolist()

    chart_base = alt.Chart(top_10_items).encode(
        y=alt.Y(
            "item_name:N",
            sort=item_order,
            title=None,
            axis=alt.Axis(
                labelFontSize=12,
                labelPadding=12,
            ),
        ),
        x=alt.X(
            "total_units:Q",
            title="Total Units Sold",
            scale=alt.Scale(
                zero=True,
                nice=True,
            ),
            axis=alt.Axis(
                format="~s",
                tickCount=6,
            ),
        ),
    )

    bars = (
        chart_base
        .mark_bar(
            cornerRadiusEnd=6,
            size=25,
        )
        .encode(
            color=alt.condition(
                alt.datum.rank == 1,
                alt.value("#38BDF8"),
                alt.value("#3B82F6"),
            ),
            tooltip=[
                alt.Tooltip(
                    "rank:Q",
                    title="Rank",
                    format=".0f",
                ),
                alt.Tooltip(
                    "item_name:N",
                    title="Product",
                ),
                alt.Tooltip(
                    "total_units:Q",
                    title="Units Sold",
                    format=",d",
                ),
                alt.Tooltip(
                    "average_daily_units:Q",
                    title="Average Units / Day",
                    format=",.2f",
                ),
                alt.Tooltip(
                    "sales_share_percent:Q",
                    title="Sales Share (%)",
                    format=".2f",
                ),
            ],
        )
    )

    labels = (
        chart_base
        .mark_text(
            align="right",
            baseline="middle",
            dx=-10,
            color="#FFFFFF",
            fontSize=11,
            fontWeight="bold",
        )
        .encode(
            text=alt.Text(
                "total_units:Q",
                format=",.0f",
            ),
        )
    )

    item_chart = (
        alt.layer(bars, labels)
        .properties(height=420)
        .configure_view(stroke=None)
        .configure_axis(
            labelColor="#A8BBD3",
            titleColor="#CBD5E1",
            domain=False,
            tickColor="#34465F",
        )
        .configure_axisX(
            grid=True,
            gridColor="#2B3A50",
            gridOpacity=0.5,
        )
        .configure_axisY(
            grid=False,
            ticks=False,
        )
    )

    st.altair_chart(
        item_chart,
        width="stretch",
    )


# ==================================================
# 16.1 PREMIUM ITEM BUSINESS SUMMARY
# ==================================================

st.subheader("Item Business Summary")

st.html(f"""
<div class="ri-item-summary-grid">

    <div class="ri-item-summary-card">

        <div class="ri-item-summary-label">
            <span class="ri-item-summary-green">●</span>
            BEST-SELLING PRODUCT
        </div>

        <div class="ri-item-summary-name">
            Item {top_item_id}
        </div>

        <div class="ri-item-summary-value">
            {top_item_units:,}
            <span>units sold</span>
        </div>

        <div class="ri-item-summary-foot">
            {top_item_share:.2f}% of selected sales volume
        </div>

    </div>

    <div class="ri-item-summary-card">

        <div class="ri-item-summary-label">
            <span class="ri-item-summary-blue">●</span>
            TOP 10 PRODUCT CONTRIBUTION
        </div>

        <div class="ri-item-summary-name">
            {top_10_share:.2f}%
        </div>

        <div class="ri-item-summary-value">
            Top {len(top_10_items)} products
        </div>

        <div class="ri-item-summary-foot">
            Share of total unit sales in selected scope
        </div>

    </div>

</div>
""")

st.caption(
    f"Analysis scope: {selected_scope}. "
    f"Period: {start_date} to {end_date}. "
    "Products are identified by numeric IDs. "
    "Average daily units are calculated across the "
    "selected store scope. Rankings are based on "
    "units sold, not revenue or profit."
)

# ==================================================
# UI-05.4 - PREMIUM PRODUCT CONTRIBUTION ANALYSIS
# ==================================================

st.divider()

st.header("Product Contribution Analysis")

st.caption(
    "Explore sales concentration across ranked products, "
    "measure cumulative unit-sales contribution, and "
    "identify the minimum number of items needed to "
    "reach the 80% threshold."
)

# ==================================================
# 17.1 PREPARE CONTRIBUTION DATA
# ==================================================

# Preserve original ranking and business logic
contribution = item_ranking[
    ["rank", "item_name", "total_units"]
].copy()

contribution = contribution.sort_values(
    "rank"
).reset_index(drop=True)

total_items = len(contribution)

if total_items == 0:
    st.warning("No products available for analysis.")
    st.stop()

if scope_total_units <= 0:
    st.warning(
        "Product contribution cannot be calculated "
        "because the selected scope has no positive "
        "unit sales."
    )
    st.stop()

# Individual product share
contribution["product_share_pct"] = (
    contribution["total_units"]
    / scope_total_units * 100
)

# Cumulative product sales
contribution["cumulative_units"] = (
    contribution["total_units"].cumsum()
)

# Cumulative sales contribution percentage
contribution["cumulative_share_pct"] = (
    contribution["cumulative_units"]
    / scope_total_units * 100
)

# ==================================================
# 17.2 CONTRIBUTION INDICATORS
# ==================================================

top_10_count = min(10, total_items)
top_20_count = min(20, total_items)

top_10_share = (
    contribution.head(10)["total_units"].sum()
    / scope_total_units * 100
)

top_20_share = (
    contribution.head(20)["total_units"].sum()
    / scope_total_units * 100
)

# First ranked product reaching 80%
threshold_rows = contribution.loc[
    contribution["cumulative_share_pct"] >= 80.0
]

if threshold_rows.empty:
    # Numerical precision fallback
    threshold_row = contribution.iloc[-1]
else:
    threshold_row = threshold_rows.iloc[0]

items_for_80 = int(threshold_row["rank"])

actual_threshold_share = float(
    threshold_row["cumulative_share_pct"]
)

threshold_item_name = str(
    threshold_row["item_name"]
)

# Percentage of products required for 80% sales
catalog_coverage_pct = (
    items_for_80 / total_items * 100
)

# Evaluate the actual top 20% of the catalog
top_fifth_count = max(
    1,
    (total_items + 4) // 5
)

top_fifth_share = (
    contribution.head(top_fifth_count)["total_units"].sum()
    / scope_total_units * 100
)

top_fifth_catalog_pct = (
    top_fifth_count / total_items * 100
)

# ==================================================
# 17.3 PREMIUM CONTRIBUTION KPI CARDS
# ==================================================

st.subheader("Portfolio Contribution Overview")

st.caption(
    f"{selected_scope} | "
    f"{start_date:%d %b %Y} - "
    f"{end_date:%d %b %Y}"
)

st.html(f"""
<div class="ri-contribution-kpi-grid">

    <div class="ri-contribution-kpi-card">

        <div class="ri-contribution-kpi-label">
            TOP {top_10_count} CONTRIBUTION
        </div>

        <div class="ri-contribution-kpi-value">
            {top_10_share:.2f}%
        </div>

        <div class="ri-contribution-kpi-accent">
            Top {top_10_count} products
        </div>

        <div class="ri-contribution-kpi-description">
            Share of total unit sales generated by the
            highest-ranked products.
        </div>

    </div>

    <div class="ri-contribution-kpi-card">

        <div class="ri-contribution-kpi-label">
            TOP {top_20_count} CONTRIBUTION
        </div>

        <div class="ri-contribution-kpi-value">
            {top_20_share:.2f}%
        </div>

        <div class="ri-contribution-kpi-accent">
            Top {top_20_count} products
        </div>

        <div class="ri-contribution-kpi-description">
            Combined unit-sales contribution from
            the top-ranked product group.
        </div>

    </div>

    <div class="ri-contribution-kpi-card ri-contribution-kpi-focus">

        <div class="ri-contribution-kpi-label">
            ITEMS COVERING 80% OF SALES
        </div>

        <div class="ri-contribution-kpi-value">
            {items_for_80}
            <span>/ {total_items}</span>
        </div>

        <div class="ri-contribution-kpi-accent ri-contribution-positive">
            {catalog_coverage_pct:.0f}% of catalog
        </div>

        <div class="ri-contribution-kpi-description">
            Minimum number of top-ranked products
            required to reach 80% of unit sales.
        </div>

    </div>

</div>
""")

# ==================================================
# 17.4 PREMIUM CUMULATIVE CONTRIBUTION CHART
# ==================================================

with st.container(
    border=True,
    key="ri_contribution_chart_panel",
):

    st.markdown("#### Cumulative Sales Contribution")

    st.caption(
        "Products are ordered from highest to lowest "
        "sales volume. The curve shows their cumulative "
        "share of total unit sales."
    )

    # Shared chart axes
    chart_base = alt.Chart(contribution).encode(

        x=alt.X(
            "rank:Q",
            title="Product Rank (Highest to Lowest Sales)",
            scale=alt.Scale(
                domain=[1, max(2, total_items)]
            ),
            axis=alt.Axis(
                tickCount=10,
                format="d",
            ),
        ),

        y=alt.Y(
            "cumulative_share_pct:Q",
            title="Cumulative Sales Share (%)",
            scale=alt.Scale(
                domain=[0, 100]
            ),
            axis=alt.Axis(
                values=[0, 20, 40, 60, 80, 100],
                format=".0f",
            ),
        ),

    )

    # Tooltip fields
    chart_tooltip = [
        alt.Tooltip(
            "rank:Q",
            title="Product Rank",
            format=".0f",
        ),
        alt.Tooltip(
            "item_name:N",
            title="Product",
        ),
        alt.Tooltip(
            "total_units:Q",
            title="Units Sold",
            format=",d",
        ),
        alt.Tooltip(
            "product_share_pct:Q",
            title="Product Share (%)",
            format=".2f",
        ),
        alt.Tooltip(
            "cumulative_units:Q",
            title="Cumulative Units",
            format=",d",
        ),
        alt.Tooltip(
            "cumulative_share_pct:Q",
            title="Cumulative Share (%)",
            format=".2f",
        ),
    ]

    # Subtle area underneath cumulative line
    area_layer = (
        chart_base
        .mark_area(
            color="#38BDF8",
            opacity=0.10,
        )
    )

    # Main cumulative contribution line
    line_layer = (
        chart_base
        .mark_line(
            color="#38BDF8",
            strokeWidth=3,
        )
        .encode(
            tooltip=chart_tooltip,
        )
    )

    # Interactive product points
    points_layer = (
        chart_base
        .mark_circle(
            color="#7DD3FC",
            size=40,
            opacity=0.95,
        )
        .encode(
            tooltip=chart_tooltip,
        )
    )

    # Horizontal 80% reference
    reference_data = pd.DataFrame({
        "threshold_pct": [80.0]
    })

    threshold_layer = (
        alt.Chart(reference_data)
        .mark_rule(
            color="#F59E0B",
            strokeWidth=2,
            strokeDash=[7, 5],
        )
        .encode(
            y=alt.Y("threshold_pct:Q")
        )
    )

    # Vertical line at the first item reaching 80%
    vertical_data = pd.DataFrame({
        "target_rank": [items_for_80]
    })

    vertical_layer = (
        alt.Chart(vertical_data)
        .mark_rule(
            color="#34D399",
            strokeWidth=1.5,
            strokeDash=[4, 5],
            opacity=0.85,
        )
        .encode(
            x=alt.X("target_rank:Q")
        )
    )

    # Highlight actual cumulative value at target
    target_data = pd.DataFrame({
        "rank": [items_for_80],
        "cumulative_share_pct": [
            actual_threshold_share
        ],
        "item_name": [threshold_item_name],
    })

    target_point = (
        alt.Chart(target_data)
        .mark_circle(
            color="#34D399",
            size=220,
            stroke="#0B1220",
            strokeWidth=2,
        )
        .encode(
            x=alt.X("rank:Q"),
            y=alt.Y("cumulative_share_pct:Q"),
            tooltip=[
                alt.Tooltip(
                    "rank:Q",
                    title="Items Required",
                    format=".0f",
                ),
                alt.Tooltip(
                    "item_name:N",
                    title="Threshold Product",
                ),
                alt.Tooltip(
                    "cumulative_share_pct:Q",
                    title="Actual Cumulative Share (%)",
                    format=".2f",
                ),
            ],
        )
    )

    # Combine visual layers
    contribution_chart = (
        alt.layer(
            area_layer,
            line_layer,
            points_layer,
            threshold_layer,
            vertical_layer,
            target_point,
        )
        .properties(height=390)
        .configure_view(
            stroke=None,
        )
        .configure_axis(
            labelColor="#A8BBD3",
            titleColor="#CBD5E1",
            labelFontSize=11,
            titleFontSize=12,
            domain=False,
            tickColor="#34465F",
        )
        .configure_axisX(
            grid=False,
        )
        .configure_axisY(
            grid=True,
            gridColor="#2B3A50",
            gridOpacity=0.55,
        )
    )

    st.altair_chart(
        contribution_chart,
        width="stretch",
    )

    # Chart legend
    st.html(f"""
    <div class="ri-contribution-legend">

        <div class="ri-contribution-legend-item">
            <span class="ri-contribution-line-blue"></span>
            <span>Cumulative sales contribution</span>
        </div>

        <div class="ri-contribution-legend-item">
            <span class="ri-contribution-line-amber"></span>
            <span>80% reference threshold</span>
        </div>

        <div class="ri-contribution-legend-item">
            <span class="ri-contribution-dot-green"></span>
            <span>
                Target reached at product rank
                {items_for_80}
            </span>
        </div>

    </div>
    """)

    st.caption(
        f"The first {items_for_80} ranked products "
        f"account for {actual_threshold_share:.2f}% "
        "of total unit sales. Hover over a point "
        "to inspect its individual and cumulative "
        "contributions."
    )


# ==================================================
# 17.5 PREMIUM CONTRIBUTION INSIGHTS
# ==================================================

st.subheader("Contribution Insights")

# Data-based 80/20 interpretation
if top_fifth_share >= 80.0:
    concentration_interpretation = (
        f"The top {top_fifth_count} products "
        f"({top_fifth_catalog_pct:.0f}% of the catalog) "
        f"generate {top_fifth_share:.2f}% of sales, "
        "meeting or exceeding the 80% benchmark."
    )
else:
    concentration_interpretation = (
        f"The top {top_fifth_count} products "
        f"({top_fifth_catalog_pct:.0f}% of the catalog) "
        f"generate {top_fifth_share:.2f}% of sales. "
        "The observed distribution is less concentrated "
        "than the classic 80/20 benchmark."
    )

st.html(f"""
<div class="ri-contribution-insights-panel">

    <div class="ri-contribution-insights-header">
        <strong>Business Interpretation</strong>
        <span>{selected_scope}</span>
    </div>

    <div class="ri-contribution-insight-row">

        <div class="ri-contribution-insight-number">
            01
        </div>

        <div class="ri-contribution-insight-text">
            The top <strong>{top_10_count} products</strong>
            contribute
            <strong>{top_10_share:.2f}%</strong>
            of unit sales, while the top
            <strong>{top_20_count} products</strong>
            contribute
            <strong>{top_20_share:.2f}%</strong>.
        </div>

    </div>

    <div class="ri-contribution-insight-row">

        <div class="ri-contribution-insight-number">
            02
        </div>

        <div class="ri-contribution-insight-text">
            A minimum of
            <strong>{items_for_80} products</strong>
            are required to reach the
            <strong>80% sales threshold</strong>.
            This represents
            <strong>{catalog_coverage_pct:.2f}%</strong>
            of the analyzed product catalog.
        </div>

    </div>

    <div class="ri-contribution-insight-row">

        <div class="ri-contribution-insight-number">
            03
        </div>

        <div class="ri-contribution-insight-text">
            {concentration_interpretation}
        </div>

    </div>

    <div class="ri-contribution-insights-footer">
        All contribution metrics use units sold,
        not revenue or profitability.
        The 80% threshold is a reference,
        not an assumption about the data.
    </div>

</div>
""")