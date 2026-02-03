import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


DATA_PATH = "Dataset/Features.csv"

EMISSION_MODEL_PATH = "Models_Trained/Emission_Sklearn.pkl"
EMISSION_SCALER_PATH = "Models_Trained/Emission_Scaler.pkl"

CARBON_PRICE_MODEL_PATH = "Models_Trained/CarbonPrice_Sklearn.pkl"
CARBON_PRICE_SCALER_PATH = "Models_Trained/CarbonPrice_Scaler.pkl"

RF_BUYSELL_PATH = "Models_Trained/RF_BuySell_Predictor.pkl"
RF_COMPLIANCE_PATH = "Models_Trained/RF_Compliance_Cost.pkl"
RF_COST_SAVINGS_PATH = "Models_Trained/RF_Cost_Savings.pkl"
RF_OPTIMIZATION_PATH = "Models_Trained/RF_Optimization_Scenario.pkl"


def build_fuel_one_hot(fuel_type):
    return (
        1.0 if fuel_type == "Mixed Fuel" else 0.0,
        1.0 if fuel_type == "Natural Gas" else 0.0,
        1.0 if fuel_type == "Renewable" else 0.0,
    )


def build_industry_one_hot(industry_type):
    return (
        1.0 if industry_type == "Energy" else 0.0,
        1.0 if industry_type == "Manufacturing" else 0.0,
        1.0 if industry_type == "Steel" else 0.0,
    )


def build_optimization_one_hot(scenario):
    return (
        1.0 if scenario == "Low Demand" else 0.0,
        1.0 if scenario == "Price Surge" else 0.0,
    )


def month_cyclical(month):
    angle = 2 * np.pi * (month / 12)
    return np.sin(angle), np.cos(angle)


def train_emission_model():
    df = pd.read_csv(DATA_PATH)

    X_cols = [
        "Energy_Demand_MWh",
        "Emission_Allowance_tCO2",
        "Carbon_Price_USD_per_t",
        "Emission_Intensity",
        "Fuel_Type_Mixed Fuel",
        "Fuel_Type_Natural Gas",
        "Fuel_Type_Renewable",
        "Industry_Type_Energy",
        "Industry_Type_Manufacturing",
        "Industry_Type_Steel",
    ]

    X = df[X_cols].astype(float).values
    y = df["Emission_Produced_tCO2"].values

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, EMISSION_MODEL_PATH)
    joblib.dump(scaler, EMISSION_SCALER_PATH)

    return model, scaler


def train_carbon_price_model():
    df = pd.read_csv(DATA_PATH)

    X_cols = [
        "Energy_Demand_MWh",
        "Emission_Produced_tCO2",
        "Emission_Allowance_tCO2",
        "Fuel_Type_Mixed Fuel",
        "Fuel_Type_Natural Gas",
        "Fuel_Type_Renewable",
        "Industry_Type_Energy",
        "Industry_Type_Manufacturing",
        "Industry_Type_Steel",
    ]

    X = df[X_cols].astype(float).values
    y = df["Carbon_Price_USD_per_t"].values

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, CARBON_PRICE_MODEL_PATH)
    joblib.dump(scaler, CARBON_PRICE_SCALER_PATH)

    return model, scaler


def load_model_and_scaler(model_path, scaler_path, train_fn):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return train_fn()


def load_model_if_exists(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


st.set_page_config(page_title="Carbon Credit Predictor", layout="centered")
st.title("Carbon Credit Predictor")

st.write("Enter inputs once and predict all models.")

emission_model, emission_scaler = load_model_and_scaler(
    EMISSION_MODEL_PATH, EMISSION_SCALER_PATH, train_emission_model
)
carbon_price_model, carbon_price_scaler = load_model_and_scaler(
    CARBON_PRICE_MODEL_PATH, CARBON_PRICE_SCALER_PATH, train_carbon_price_model
)

rf_buysell = load_model_if_exists(RF_BUYSELL_PATH)
rf_compliance = load_model_if_exists(RF_COMPLIANCE_PATH)
rf_cost_savings = load_model_if_exists(RF_COST_SAVINGS_PATH)
rf_optimization = load_model_if_exists(RF_OPTIMIZATION_PATH)

# Input fields for the features
st.subheader("Energy and Emission Parameters")
energy_demand = st.number_input("Energy Demand (MWh)", 0.0, 10000.0, 1000.0)
emission_allowance = st.number_input("Emission Allowance (tCO2)", 0.0, 10000.0, 500.0)
carbon_price = st.number_input("Carbon Price (USD per t)", 0.0, 200.0, 50.0)
emission_intensity = st.number_input("Emission Intensity", 0.0, 5.0, 0.5)
month = st.number_input("Month (1-12)", 1, 12, 1)

st.subheader("Fuel Type")
fuel_type = st.selectbox("Fuel Type", ["Coal", "Mixed Fuel", "Natural Gas", "Renewable"])

st.subheader("Industry Type")
industry_type = st.selectbox("Industry Type", ["Construction", "Energy", "Manufacturing", "Steel"])

st.subheader("Optimization Scenario")
optimization_scenario = st.selectbox(
    "Scenario", ["Normal", "Low Demand", "Price Surge"]
)

def emission_category(emission):
    if emission <= 500:
        return "Low Emission"
    elif emission <= 1000:
        return "Moderate Emission"
    elif emission <= 2000:
        return "High Emission"
    else:
        return "Very High Emission"

if st.button("Predict All Models"):
    # Create one-hot encoded features
    fuel_type_mixed = 1.0 if fuel_type == "Mixed Fuel" else 0.0
    fuel_type_natural_gas = 1.0 if fuel_type == "Natural Gas" else 0.0
    fuel_type_renewable = 1.0 if fuel_type == "Renewable" else 0.0
    
    industry_type_energy = 1.0 if industry_type == "Energy" else 0.0
    industry_type_manufacturing = 1.0 if industry_type == "Manufacturing" else 0.0
    industry_type_steel = 1.0 if industry_type == "Steel" else 0.0
    
    # Prepare input data
    input_data = np.array([[
        energy_demand,
        emission_allowance,
        carbon_price,
        emission_intensity,
        fuel_type_mixed,
        fuel_type_natural_gas,
        fuel_type_renewable,
        industry_type_energy,
        industry_type_manufacturing,
        industry_type_steel
    ]])
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    predicted_emission = prediction[0]
    
    st.success(f"Predicted Carbon Emission: {predicted_emission:.2f} tCO2")
    category = emission_category(predicted_emission)
    st.info(f"Emission Category: {category}")