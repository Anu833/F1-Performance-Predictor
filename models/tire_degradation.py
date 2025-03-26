import pandas as pd
import numpy as np
import json

df = pd.read_csv("data/processed/f1_cleaned.csv")

degradation_rates = {}

for (track, driver), group in df.groupby(["circuitId", "driverId"]):
    if len(group) > 5:
        base_time = group["milliseconds"].min()
        rate = np.polyfit(group["tire_age_laps"], np.log(group["milliseconds"] / base_time), 1)[0]
        degradation_rates[f"{track}_{driver}"] = round(rate, 6)

with open("models/tire_degradation.json", "w") as f:
    json.dump(degradation_rates, f)

print(f"Tire degradation model saved! {len(degradation_rates)} drivers tracked.")
