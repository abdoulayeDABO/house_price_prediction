import pandas as pd
from utils import *
from pathlib import Path

models_dir = Path("models")
filename = models_dir / "trained_model.pkl"

# Load model
loaded_model = load_model(filename)

# Test data
new_data = pd.DataFrame({
    "longitude":          [-122.23, -118.35, -119.77, -117.03, -121.50],
    "latitude":           [ 37.88,   34.05,   36.73,   32.72,   38.52],
    "housing_median_age": [ 41.0,    25.0,    30.0,    15.0,    20.0 ],
    "total_rooms":        [ 880.0,  2500.0,  1500.0,  3000.0,  1200.0],
    "total_bedrooms":     [ 129.0,   500.0,   300.0,   600.0,   250.0],
    "population":         [ 322.0,  1200.0,   700.0,  1500.0,   600.0],
    "households":         [ 126.0,   480.0,   280.0,   550.0,   230.0],
    "median_income":      [  8.33,    4.50,    3.20,    6.10,    2.80],
    "ocean_proximity":    ["NEAR BAY", "NEAR OCEAN", "INLAND", "<1H OCEAN", "INLAND"],
})

predictions = loaded_model.predict(new_data)

for i, pred in enumerate(predictions):
    print(f"House {i+1}: ${pred:,.0f}")
