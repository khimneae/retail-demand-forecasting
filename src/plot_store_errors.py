from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "reports" / "metrics" / "validation_by_store.csv"
)

OUTPUT_PATH = (
    ROOT / "reports" / "figures" / "store_mae_comparison.png"
)

# 2. Load store-level evaluation results
df = pd.read_csv(INPUT_PATH)
df = df.sort_values("store")

# 3. Prepare chart positions
x = np.arange(len(df))
width = 0.38

fig, ax = plt.subplots(figsize=(12, 6))

# 4. Plot baseline MAE
ax.bar(
    x - width / 2,
    df["baseline_mae"],
    width,
    label="Seasonal Naive Baseline"
)

# 5. Plot ML model MAE
ax.bar(
    x + width / 2,
    df["mae"],
    width,
    label="HistGradientBoosting"
)

# 6. Chart formatting
ax.set_title(
    "Baseline vs Machine Learning: MAE by Store"
)
ax.set_xlabel("Store")
ax.set_ylabel("Mean Absolute Error (Units)")

ax.set_xticks(x)
ax.set_xticklabels(df["store"].astype(str))

ax.legend()
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()

# 7. Save chart
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

fig.savefig(OUTPUT_PATH, dpi=180)
plt.close(fig)

print("=== STORE MAE COMPARISON ===")
print(f"Stores analyzed: {len(df)}")
print(f"Chart saved: {OUTPUT_PATH}")
print("Visualization completed successfully!")
