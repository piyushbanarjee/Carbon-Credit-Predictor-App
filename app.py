import streamlit as st
import numpy as np
import joblib
import torch

st.set_page_config(page_title="Carbon Emission Prediction App", layout="centered")
st.title("Carbon Emission Prediction")

st.write("Predict the Carbon Emission (tCO2) based on input parameters.")

# Load the pre-trained model and scaler
model = joblib.load("Models_Trained/MLP_Emission.pkl")
scaler = joblib.load("Models_Trained/MLP_Emission_Scalar.pkl")

# Input fields for the features
st.subheader("Energy and Emission Parameters")
energy_demand = st.number_input("Energy Demand (MWh)", 0.0, 10000.0, 1000.0)
emission_allowance = st.number_input("Emission Allowance (tCO2)", 0.0, 10000.0, 500.0)
carbon_price = st.number_input("Carbon Price (USD per t)", 0.0, 200.0, 50.0)
emission_intensity = st.number_input("Emission Intensity", 0.0, 5.0, 0.5)

st.subheader("Fuel Type")
fuel_type = st.selectbox("Fuel Type", ["Coal", "Mixed Fuel", "Natural Gas", "Renewable"])

st.subheader("Industry Type")
industry_type = st.selectbox("Industry Type", ["Construction", "Energy", "Manufacturing", "Steel"])

def emission_category(emission):
    if emission <= 500:
        return "Low Emission"
    elif emission <= 1000:
        return "Moderate Emission"
    elif emission <= 2000:
        return "High Emission"
    else:
        return "Very High Emission"

if st.button("Predict Carbon Emission"):
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
    
    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_emission = prediction.item()
    
    st.success(f"Predicted Carbon Emission: {predicted_emission:.2f} tCO2")
    category = emission_category(predicted_emission)
    st.info(f"Emission Category: {category}")