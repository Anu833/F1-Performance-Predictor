import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

df = pd.read_csv("data/processed/f1_cleaned.csv")

lap_time_model = joblib.load("models/lap_time_model.pkl")
lap_time_model_2025 = joblib.load("models/lap_time_model_2025.pkl")

driver_map = {row['driverId']: f"{row['forename']} {row['surname']}" for _, row in df.drop_duplicates(subset=['driverId']).iterrows()}
track_map = {row['circuitId']: row['circuit_name'] for _, row in df.drop_duplicates(subset=['circuitId']).iterrows()}

# Streamlit UI 
st.set_page_config(page_title="F1® Performance Predictor", page_icon="🏎️", layout="wide")

def generate_2025_data(df):
    df_2024 = df[df["year"] == 2024].copy()
    
    df_2025 = df_2024.copy()
    df_2025["year"] = 2025
    
    performance_factors = {
        "constructor_improvement": np.random.uniform(0.98, 1.02),
        "driver_adaptation": np.random.uniform(0.99, 1.01),
        "track_evolution": np.random.uniform(0.98, 1.02)
    }
    
    df_2025["milliseconds"] *= (
        performance_factors["constructor_improvement"] * 
        performance_factors["driver_adaptation"] * 
        performance_factors["track_evolution"]
    )
    
    return df_2025

def main():
    years = sorted(df["year"].unique())
    selected_year = st.sidebar.selectbox("Select Year", list(years) + [2025], index=len(years) - 1)

    if selected_year == 2025:
        filtered_df = generate_2025_data(df)
        model_to_use = lap_time_model_2025
    else:
        filtered_df = df[df["year"] == selected_year].copy()
        model_to_use = lap_time_model

    available_tracks = filtered_df["circuitId"].unique()
    available_drivers = filtered_df["driverId"].unique()

    selected_track = st.sidebar.selectbox("Select Track", options=available_tracks, format_func=lambda x: track_map.get(x, f"Track {x}"))
    selected_driver = st.sidebar.selectbox("Select Driver", options=available_drivers, format_func=lambda x: driver_map.get(x, f"Driver {x}"))

    features = ["circuitId", "driverId", "constructorId", "grid_scaled", "position_scaled", "tire_age_laps", "track_temp"]

    missing_features = [col for col in features if col not in filtered_df.columns]
    if missing_features:
        for col in missing_features:
            filtered_df[col] = 0  # Default values to avoid missing errors

    track_driver_data = filtered_df[(filtered_df["circuitId"] == selected_track) & (filtered_df["driverId"] == selected_driver)].copy()

    if track_driver_data.empty:
        st.error(f"No data found for Driver {driver_map.get(selected_driver)} at Track {track_map.get(selected_track)}")
        return

    input_row = track_driver_data.iloc[0]
    
    if input_row[features].isna().any():
        st.error("Missing feature data for prediction.")
        return


    X_input = input_row[features].values.reshape(1, -1)
    print(f"X_input shape: {X_input.shape}")

    
    predicted_lap_time_ms = model_to_use.predict(X_input)[0]
    predicted_lap_time_min = predicted_lap_time_ms / 60000  # Convert ms to minutes

    track_data = filtered_df[filtered_df["circuitId"] == selected_track]
    fastest_lap_min, lap_time_delta = None, None

    if "milliseconds" in track_data.columns and not track_data["milliseconds"].isna().all():
        fastest_lap_ms = track_data["milliseconds"].min()
        fastest_lap_min = fastest_lap_ms / 60000
        lap_time_delta = predicted_lap_time_min - fastest_lap_min

    st.markdown("## Race Insights")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric("Predicted Lap Time", f"{predicted_lap_time_min:.3f} min")
        if fastest_lap_min is not None:
            st.metric("Fastest Lap This Year", f"{fastest_lap_min:.3f} min")
            st.metric("Lap Time Delta", f"{lap_time_delta:.3f} min")
        else:
            st.warning("No historical fastest lap data available.")

    st.markdown("## Performance Trends")
    col4, col5 = st.columns([1, 2])

    with col4:
        tire_laps = np.arange(1, 30)
        degradation_rate = 0.02
        lap_times_min = (predicted_lap_time_ms * (1 + degradation_rate) ** tire_laps) / 60000
        
        fig_tire_deg = px.line(
            x=tire_laps, 
            y=lap_times_min, 
            labels={"x": "Tire Age (Laps)", "y": "Lap Time (min)"},
            title="Tire Degradation Curve"
        )
        st.plotly_chart(fig_tire_deg)

    with col5:
        if "milliseconds" in df.columns and not df["milliseconds"].isna().all():
            track_history = df[df["circuitId"] == selected_track].groupby("year")["milliseconds"].mean() / 60000
            if not track_history.empty:
                fig_lap_trends = px.line(
                    x=track_history.index, 
                    y=track_history.values, 
                    labels={"x": "Year", "y": "Avg Lap Time (min)"},
                    title="Historical Lap Time Trends"
                )
                st.plotly_chart(fig_lap_trends)
            else:
                st.warning("No historical lap time data available.")
        else:
            st.warning("No historical lap time data available.")

if __name__ == "__main__":
    main()
