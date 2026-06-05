import streamlit as st
import pandas as pd
import os
import joblib
from utils import load_model, save_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")

# Config page
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline as SKPipeline

        os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

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

    return load_model(MODEL_PATH)

model = get_model()

# Titre
st.title("California House Price Predictor")
st.markdown("Remplis les informations pour estimer le prix d'une maison.")

# Formulaire
st.subheader("Informations sur la maison")
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23, format="%.4f")
    latitude = st.number_input("Latitude", value=37.88, format="%.4f")
    housing_median_age = st.slider("Âge médian du quartier", 1, 52, 20)
    total_rooms = st.number_input("Nombre total de pièces", value=880, min_value=1)
    total_bedrooms = st.number_input("Nombre total de chambres", value=129, min_value=1)

with col2:
    population = st.number_input("Population", value=322, min_value=1)
    households = st.number_input("Nombre de ménages", value=126, min_value=1)
    median_income = st.slider("Revenu médian (en dizaines de milliers $)", 0.5, 15.0, 5.0)
    ocean_proximity = st.selectbox(
        "Proximité de l'océan",
        ["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN", "ISLAND"]
    )

# Prédiction
if st.button("Estimer le prix", use_container_width=True):
    input_data = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity,
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"### 💰 Prix estimé : ${prediction:,.0f}")

    st.divider()
    col3, col4, col5 = st.columns(3)
    col3.metric("Prix au m²", f"${prediction/total_rooms:,.0f}")
    col4.metric("Revenu médian", f"${median_income*10_000:,.0f}")
    col5.metric("Âge du quartier", f"{housing_median_age} ans")