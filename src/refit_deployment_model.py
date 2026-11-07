from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "model_features.csv"
)

SOURCE_MODEL = (
    ROOT / "models" / "hgb_final_model.joblib"
)

OUTPUT_MODEL = (
    ROOT / "models" / "hgb_deployment_model.joblib"
)


# ==================================================
# 2. LOAD TRAINED MODEL CONFIGURATION
# ==================================================

artifact = joblib.load(SOURCE_MODEL)

features = artifact["features"]
target = "sales"

# Clone the algorithm and its hyperparameters
# without copying the previously learned model.
model = clone(artifact["model"])


# ==================================================
# 3. LOAD ALL AVAILABLE HISTORICAL DATA
# ==================================================

df = pd.read_csv(
    DATA_PATH,
    parse_dates=["date"]
)

assert len(df) == 896000
assert df["date"].min() == pd.Timestamp("2013-02-04")
assert df["date"].max() == pd.Timestamp("2017-12-31")
assert df[features + [target]].notna().all().all()

X_train = df[features]
y_train = df[target]


# ==================================================
# 4. REFIT DEPLOYMENT MODEL
# ==================================================

print("=== DEPLOYMENT MODEL TRAINING ===")

print(f"Training samples: {len(df):,}")
print(f"Features: {len(features)}")
print(f"Start: {df['date'].min().date()}")
print(f"End: {df['date'].max().date()}")

print("\nTraining deployment model...", flush=True)

model.fit(X_train, y_train)

print("Training completed!")


# ==================================================
# 5. SAVE DEPLOYMENT MODEL
# ==================================================

OUTPUT_MODEL.parent.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    {
        "model": model,
        "features": features,
        "target": target,
        "training_end": "2017-12-31",
        "forecast_horizon_days": 7,
        "purpose": "Historical demo forecast for 2018",
    },
    OUTPUT_MODEL
)

print("\nDeployment model saved successfully!")
print(f"Model path: {OUTPUT_MODEL}")
print("Ready for simulated 7-day forecasting.")
