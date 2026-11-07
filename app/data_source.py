
"""Resolve the retail sales dataset for local and demo environments."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ORIGINAL_DATA_PATH = ROOT / "data" / "raw" / "train.csv"
DEMO_DATA_PATH = ROOT / "data" / "demo" / "synthetic_sales.csv"


def get_data_source() -> tuple[Path, bool]:
    """
    Return (dataset_path, is_demo).

    RETAIL_DATA_MODE:
        auto     : prefer original, otherwise use demo
        demo     : always use synthetic demo data
        original : require the local Kaggle dataset
    """
    mode = os.getenv("RETAIL_DATA_MODE", "auto").strip().lower()

    if mode not in {"auto", "demo", "original"}:
        raise ValueError(
            "RETAIL_DATA_MODE must be auto, demo, or original."
        )

    if mode == "demo":
        path = DEMO_DATA_PATH
        is_demo = True

    elif mode == "original":
        path = ORIGINAL_DATA_PATH
        is_demo = False

    elif ORIGINAL_DATA_PATH.is_file():
        path = ORIGINAL_DATA_PATH
        is_demo = False

    else:
        path = DEMO_DATA_PATH
        is_demo = True

    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return path, is_demo
