from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "model_features.csv"
OUTPUT_PATH = ROOT / "models" / "hgb_final_model.joblib"

# 2. Feature configuration
FEATURES = [
    "store",
    "item",
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]

TARGET = "sales"

# 3. Load dataset
df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

# 4. Combine Train and Validation
# Keep the original split boundaries.
train_mask = df["date"] <= "2015-12-24"

val_mask = df["date"].between(
    "2016-01-01",
    "2016-12-24"
)

final_train = df.loc[
    train_mask | val_mask
].copy()

# 5. Validate data
assert len(final_train) == 706500
assert final_train["date"].max() == pd.Timestamp("2016-12-24")
assert final_train[FEATURES + [TARGET]].notna().all().all()

X_train = final_train[FEATURES]
y_train = final_train[TARGET]

# 6. Use the same approved hyperparameters
model = HistGradientBoostingRegressor(
    max_iter=120,
    learning_rate=0.08,
    max_leaf_nodes=31,
    l2_regularization=0.1,
    early_stopping=False,
    random_state=42
)

# 7. Refit
print("=== FINAL MODEL REFITTING ===")
print(f"Training samples: {len(X_train):,}")
print(f"Features: {len(FEATURES)}")
print(f"Start: {final_train['date'].min().date()}")
print(f"End: {final_train['date'].max().date()}")

print("\nRefitting model...", flush=True)

model.fit(X_train, y_train)

# 8. Save final model
OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "training_end": "2016-12-24",
        "forecast_horizon_days": 7
    },
    OUTPUT_PATH
)

print("\nFinal model refitting completed!")
print(f"Model saved to: {OUTPUT_PATH}")
print("Test dataset has not been evaluated.")
