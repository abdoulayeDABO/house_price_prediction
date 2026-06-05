import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from utils import *
from pathlib import Path

# Import dataset
print("Loading dataset ...")
housing_data = pd.read_csv('./housing.csv')
X = housing_data.drop("median_house_value", axis=1)
y = housing_data["median_house_value"]

# Separate column types
categorical_cols = ["ocean_proximity"]
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# Create preprocessor
print("Creating pipeline ...")
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
])

pipeline = make_pipeline(preprocessor, RandomForestRegressor(random_state=42))

# Fit on ALL data
print("Fitting pipeline ...")
pipeline.fit(X, y)

# Save model
print("Saving ...")
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)
filename = models_dir / "trained_model.pkl"
save_model(pipeline, filename)
print("Done.")


