import pickle
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SKPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename):
    with open(filename, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def train_and_save_model():
        df = pd.read_csv(os.path.join(BASE_DIR, "housing.csv"))
        X = df.drop(columns=["median_house_value"])
        y = df["median_house_value"]

        numeric_features = ["longitude", "latitude", "housing_median_age",
                            "total_rooms", "total_bedrooms", "population",
                            "households", "median_income"]
        categorical_features = ["ocean_proximity"]

        preprocessor = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ])

        model = SKPipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X, y)
        save_model(model, MODEL_PATH)
    
