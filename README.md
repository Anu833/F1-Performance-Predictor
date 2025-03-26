# F1 Performance Predictor

## Overview
The **F1 Performance Predictor** is a data-driven tool designed to predict Formula 1 race performance, including lap times, tire degradation, and historical analysis. The application leverages **machine learning models (LightGBM)** to estimate lap times based on past race data and project performance for the **2025 season**.

## Features
- **Lap Time Prediction:** Predict lap times based on historical race data.
- **2025 Performance Projection:** Extrapolate lap times for the upcoming F1 season.
- **Tire Degradation Modeling:** Analyse how lap times evolve with tire wear.
- **Interactive Streamlit UI:** Select year, track, and driver dynamically.
- **Historical Performance Insights:** Compare predicted and actual lap times.
- **Lightweight & Optimized:** Efficient data processing for quick predictions.

## Project Structure
```
.
├── Dockerfile                
├── data/
│   ├── processed/            
│   │   └── f1_cleaned.csv
│   └── raw/                  
│       ├── circuits.csv
│       ├── drivers.csv
│       ├── lap_times.csv
│       ├── pit_stops.csv
│       ├── races.csv
│       ├── results.csv
│       ├── seasons.csv
│       └── other files...
├── frontend/
│   ├── app.py                
├── models/
│   ├── lap_time_model.pkl    
│   ├── lap_time_model_2025.pkl  
│   ├── tire_degradation.json  
│   ├── train.py               
│   ├── train_2025.py          
│   └── tire_degradation.py    
├── tests/                     
├── .gitignore                 
├── requirements.txt           
└── README.md                  
```

## Installation
### Prerequisites
- Python 3.8+
- pip
- [Streamlit](https://streamlit.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Anu833/F1-Performance-Predictor.git
   cd F1-Performance-Predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run frontend/app.py
   ```

## Usage
- **Select a year** (2018-2025) from the sidebar.
- **Choose a track and driver** to get lap time predictions.
- **View performance trends**, including fastest lap comparisons.
- **Analyze tire degradation** with dynamic plots.

## Model Training
- **train.py**: Trains the lap time model using historical F1 data.
- **train_2025.py**: Projects performance for the 2025 season.
- **tire_degradation.py**: Estimates how lap times change over tire life.

## Handling Large Files
- A `.gitignore` file is used to exclude large and unnecessary files:
  ```bash
  echo "data/processed/f1_cleaned.csv" >> .gitignore
  git add .gitignore
  git commit -m "Added .gitignore to exclude large files"
  git push origin main
  ```

## Future Improvements
- **Expand feature set** with weather and fuel consumption models.
- **Enhance UI** with real-time race simulations.

## Contributors
- [Anu833](https://github.com/Anu833)

## License
MIT License

