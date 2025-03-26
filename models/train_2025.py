import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib

df = pd.read_csv("data/processed/f1_cleaned.csv")

df.columns = df.columns.str.strip().str.lower()

results = pd.read_csv("data/raw/results.csv", usecols=["raceId", "driverId", "constructorId", "grid", "positionOrder"])
results.columns = results.columns.str.strip().str.lower()

df = df.merge(results, on=["raceid", "driverid"], how="left")
df["grid_scaled"] = df["grid"].fillna(df["grid"].median()) / 20
df["position_scaled"] = df["positionorder"].fillna(df["positionorder"].median()) / 20

df = df[df["year"] >= 2020]

available_columns = df.columns.tolist()
print("Available Columns:", available_columns)

features = [
    "circuitid", "driverid", "constructorid",
    "grid_scaled", "position_scaled", "tire_age_laps", "track_temp"
]

features = [col for col in features if col in df.columns]
print("Selected Features:", features)

target = "milliseconds"

missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"Warning: Missing features {missing_features}")

train_df = df[df["year"] < 2024]
test_df = df[df["year"] == 2024]

X_train = train_df[features]
y_train = train_df[target]

# Check for NaN values
if X_train.isnull().values.any():
    print("Warning: NaN detected in training features. Filling with median.")
    X_train = X_train.fillna(X_train.median())

if y_train.isnull().values.any():
    print("Warning: NaN detected in target variable. Filling with median.")
    y_train = y_train.fillna(y_train.median())

model = lgb.LGBMRegressor(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=7,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

joblib.dump(model, "models/lap_time_model_2025.pkl")
print(" 2025 Prediction Model Trained & Saved!\n")

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n Feature Importances:")
print(feature_importance)
