from pathlib import Path

import pandas as pd
import streamlit as st

from app.data_source import get_data_source

# ==================================================
# 1. GLOBAL PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Retail Intelligence",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="auto",
)

# Paths are resolved relative to this file (app/main.py).
ROOT = Path(__file__).resolve().parents[1]

# Automatically select the local Kaggle or synthetic demo dataset.
DATA_PATH, IS_DEMO = get_data_source()

CSS_PATH = Path(__file__).resolve().parent / "assets" / "retail.css"

# ==================================================
# 2. UI-07.2 - DYNAMIC DATASET OVERVIEW
# ==================================================
@st.cache_data(show_spinner=False)
def get_dataset_overview(
    csv_path: str,
    file_size: int,
    file_mtime_ns: int,
) -> dict:
    """Read the training CSV and calculate sidebar dataset statistics.

    File size and modification time are cache-key inputs so replacing or
    editing the CSV causes the overview to be recalculated.
    """
    required_columns = {"date", "store", "item", "sales"}
    available_columns = set(pd.read_csv(csv_path, nrows=0).columns)
    missing_columns = required_columns - available_columns

    if missing_columns:
        raise ValueError(
            "Missing dataset columns: "
            + ", ".join(sorted(missing_columns))
        )

    # Loading only the relevant columns reduces memory consumption.
    data = pd.read_csv(
        csv_path,
        usecols=["date", "store", "item"],
        dtype={"store": "category", "item": "category"},
    )

    if data.empty:
        raise ValueError("The dataset has no rows.")

    if data[["store", "item"]].isna().any().any():
        raise ValueError("The dataset contains missing store/item IDs.")

    dates = pd.to_datetime(
        data["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )
    if dates.isna().any():
        raise ValueError("The dataset contains invalid or missing dates.")

    return {
        "stores": int(data["store"].nunique()),
        "products": int(data["item"].nunique()),
        "years": int(dates.dt.year.nunique()),
        "records": int(len(data)),
        "start_date": dates.min().date().isoformat(),
        "end_date": dates.max().date().isoformat(),
    }


def compact_number(value: int) -> str:
    """Compact record counts for the narrow sidebar (913K, 1.25M)."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}".rstrip("0").rstrip(".") + "K"
    return f"{value:,}"


def load_current_dataset_overview() -> dict:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    file_info = DATA_PATH.stat()
    return get_dataset_overview(
        csv_path=str(DATA_PATH),
        file_size=file_info.st_size,
        file_mtime_ns=file_info.st_mtime_ns,
    )


def sidebar_dataset_values() -> tuple[dict, dict | None, str | None]:
    """Return formatted metrics and a readable error, if any."""
    try:
        stats = load_current_dataset_overview()
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError) as exc:
        empty_values = {
            "stores": "N/A",
            "products": "N/A",
            "history": "N/A",
            "records": "N/A",
        }
        return empty_values, None, str(exc)

    year_count = stats["years"]
    history = f"{year_count} Year" if year_count == 1 else f"{year_count} Years"
    values = {
        "stores": f"{stats['stores']:,}",
        "products": f"{stats['products']:,}",
        "history": history,
        "records": compact_number(stats["records"]),
    }
    return values, stats, None


# ==================================================
# 3. REGISTER ALL COMPLETED MODULES
# ==================================================
overview = st.Page(
    "dashboard.py",
    title="Model Performance",
    icon=":material/dashboard:",
    default=True,
)

sales = st.Page(
    "pages/02_Sales_Analytics.py",
    title="Sales Analytics",
    icon=":material/show_chart:",
)

stores = st.Page(
    "pages/03_Store_Item_Insights.py",
    title="Store & Item Insights",
    icon=":material/storefront:",
)

forecast = st.Page(
    "pages/04_7_Day_Forecast.py",
    title="7-Day AI Forecast",
    icon=":material/auto_awesome:",
)

# Hide the automatic navigation: the app has a custom shared sidebar.
selected_page = st.navigation(
    [overview, sales, stores, forecast],
    position="hidden",
)

metrics, dataset_stats, dataset_error = sidebar_dataset_values()


# ==================================================
# 4. SHARED SIDEBAR
# ==================================================
with st.sidebar:
    st.markdown("### :material/monitoring: Retail Intelligence")
    st.caption("AI-Powered Retail Analytics Platform")

    st.divider()
    st.caption("WORKSPACE")

    st.page_link(
        overview,
        label="01 · Model Performance",
        icon=":material/dashboard:",
    )
    st.page_link(
        sales,
        label="02 · Sales Analytics",
        icon=":material/show_chart:",
    )
    st.page_link(
        stores,
        label="03 · Store & Item Insights",
        icon=":material/storefront:",
    )
    st.page_link(
        forecast,
        label="04 · 7-Day AI Forecast",
        icon=":material/auto_awesome:",
    )

    st.divider()
    st.caption("DATASET OVERVIEW")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Stores", metrics["stores"])
        st.metric("History", metrics["history"])
    with col2:
        st.metric("Products", metrics["products"])
        st.metric("Records", metrics["records"])

    if dataset_stats is not None:
        st.caption(
            f"{dataset_stats['start_date']} to {dataset_stats['end_date']}"
        )
    elif dataset_error is not None:
        st.warning("Dataset overview unavailable.")
        with st.expander("Dataset error details"):
            st.caption(dataset_error)

    st.divider()
    st.link_button(
        "View Source on GitHub",
        "https://github.com/khimneae/retail-demand-forecasting",
        icon=":material/open_in_new:",
        width="stretch",
    )
    st.caption("Retail Intelligence · Portfolio Edition")


# ==================================================
# 5. LOAD SHARED VISUAL DESIGN AND EXECUTE PAGE
# ==================================================

# ==================================================
# 4. SHARED DESIGN & RESPONSIVE STYLES
# ==================================================

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

BASE_CSS_PATH = ASSETS_DIR / "retail.css"
RESPONSIVE_CSS_PATH = ASSETS_DIR / "responsive.css"

# Load premium dark theme
st.html(BASE_CSS_PATH)

# Load responsive overrides after the main stylesheet
st.html(RESPONSIVE_CSS_PATH)

# ==================================================
# 5. EXECUTE CURRENT PAGE
# ==================================================
# Clearly distinguish synthetic demo data from Kaggle evaluation results.
if IS_DEMO:
    st.info(
        "**SYNTHETIC DEMO MODE** — This application is using "
        "independently generated retail sales data. "
        "The reported machine learning evaluation metrics were "
        "obtained from the original Kaggle dataset, not from "
        "the synthetic demo dataset.",
        icon="ℹ️",
    )
selected_page.run()
