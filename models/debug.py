import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/f1_cleaned.csv")
drivers = pd.read_csv("data/raw/drivers.csv", usecols=["driverId", "forename", "surname"])
circuits = pd.read_csv("data/raw/circuits.csv", usecols=["circuitId", "name"])
races = pd.read_csv("data/raw/races.csv", usecols=["raceId", "year", "circuitId"])
results = pd.read_csv("data/raw/results.csv", usecols=["raceId", "driverId", "constructorId", "grid", "positionOrder", "milliseconds"])

print("Processed Dataset Columns:", df.columns.tolist())
print("\nResults Dataset Columns:", results.columns.tolist())

df.columns = df.columns.str.strip().str.lower()
races.columns = races.columns.str.strip().str.lower()
results.columns = results.columns.str.strip().str.lower()
drivers.columns = drivers.columns.str.strip().str.lower()
circuits.columns = circuits.columns.str.strip().str.lower()

print("\nBefore Merge:")
print("DF Shape:", df.shape)
print("Races Shape:", races.shape)
print("Results Shape:", results.shape)

df = df.merge(races, on="raceid", how="left")
df = df.merge(results, on=["raceid", "driverid"], how="left")

print("\nAfter Merge:")
print("DF Shape:", df.shape)
print("\nMilliseconds column null percentage:", df["milliseconds"].isnull().mean() * 100)
print("\nSample data rows:", df[["year", "raceid", "driverid", "milliseconds"]].head())