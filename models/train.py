import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/f1_cleaned.csv")
results = pd.read_csv("data/raw/results.csv", usecols=["raceId", "driverId", "constructorId", "grid", "positionOrder"])

df = df.merge(results, on=["raceId", "driverId"], how="left")

df["lap_time_sec"] = df["milliseconds"] / 1000

df = df[(df["positionOrder"] <= 20) & (df["grid"] <= 20) & (df["lap_time_sec"] < 200)]

df["lap_time_sec"] = df.groupby("circuitId")["lap_time_sec"].transform(lambda x: (x - x.mean()) / x.std())

df["tire_age_laps"] = np.log1p(df["tire_age_laps"])  # Log transform tire age
df["grid_scaled"] = df["grid"] / 20
df["position_scaled"] = df["positionOrder"] / 20

features = ["circuitId", "driverId", "constructorId", "grid_scaled", "position_scaled", "tire_age_laps", "track_temp"]
target = "lap_time_sec"

df = df.dropna()

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

params = {
    "objective": "regression",
    "metric": "mae",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,  # Lower for better accuracy
    "num_leaves": 50,
    "max_depth": 8,
    "n_estimators": 700
}
model = lgb.LGBMRegressor(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f" Final Fixed MAE: {mae:.2f} sec")

joblib.dump(model, "models/lap_time_model.pkl")

lgb.plot_importance(model, figsize=(10, 6))
plt.show()