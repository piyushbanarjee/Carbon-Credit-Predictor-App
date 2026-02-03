# Model Documentation

This document describes the purpose, inputs, and outputs of each trained model in the project.

## 1. Emission Prediction Model (`MLP1.py`)
*   **Type**: Multi-Layer Perceptron (MLP) Regressor
*   **Goal**: Predict the total carbon emissions produced (`Emission_Produced_tCO2`).
*   **Target**: `Emission_Produced_tCO2`
*   **Inputs (Features)**:
    *   `Energy_Demand_MWh`
    *   `Emission_Allowance_tCO2`
    *   `Carbon_Price_USD_per_t`
    *   `Emission_Intensity`
    *   `Fuel_Type` (One-Hot: Mixed Fuel, Natural Gas, Renewable)
    *   `Industry_Type` (One-Hot: Energy, Manufacturing, Steel)
*   **Output**: Predicted total emissions (tCO2).

## 2. Carbon Price Prediction Model (`MPLR2.py`)
*   **Type**: Multi-Layer Perceptron (MLP) Regressor
*   **Goal**: Forecast the Carbon Price (`Carbon_Price_USD_per_t`).
*   **Target**: `Carbon_Price_USD_per_t`
*   **Inputs (Features)**:
    *   `Energy_Demand_MWh`
    *   `Emission_Produced_tCO2`
    *   `Emission_Allowance_tCO2`
    *   `Industry_Type` (One-Hot)
    *   `Fuel_Type` (One-Hot)
*   **Output**: Predicted Carbon Price (USD/t).

## 3. Buy/Sell Classification Model (`RFC1.py`)
*   **Type**: Random Forest Classifier
*   **Goal**: Recommend whether to Buy or Sell carbon credits (`Transaction_Type`).
*   **Target**: `Transaction_Type` (Binary/Categorical)
*   **Inputs (Features)**:
    *   `Energy_Demand_MWh`
    *   `Emission_Produced_tCO2`
    *   `Emission_Allowance_tCO2`
    *   `Carbon_Price_USD_per_t`
    *   `Net_Position` (Engineered: Allowance - Produced)
    *   `Net_Position_Price_interaction` (Engineered: Net_Position * Price)
    *   `Fuel_Type` (One-Hot)
*   **Output**: Transaction Recommendation (Buy/Sell).

## 4. Optimization Scenario Classifier (`RFC2.py`)
*   **Type**: Random Forest Classifier
*   **Goal**: Classify the market environment into scenarios to aid strategic decision-making.
*   **Target**: `Optimization_Scenario` (Classes: Normal, Low Demand, Price Surge)
*   **Inputs (Features)**:
    *   `Energy_Demand_MWh`
    *   `month` (Cyclical Encoding: `month_sin`, `month_cos`)
    *   `Fuel_Type` (One-Hot)
    *   `Industry_Type` (One-Hot)
*   **Output**: Market Scenario Class (Normal, Low Demand, Price Surge).

## 5. Compliance Cost & Savings Model (`RFR1.py`)
This script trains two separate Random Forest Regressors.

### A. Compliance Cost Predictor
*   **Type**: Random Forest Regressor
*   **Goal**: Estimate the total financial liability.
*   **Target**: `Compliance_Cost_USD`
*   **Inputs (Features)**:
    *   `Energy_Demand_MWh`
    *   `Emission_Produced_tCO2`
    *   `Carbon_Price_USD_per_t`
    *   `Industry_Type` (One-Hot)
*   **Output**: Estimated Compliance Cost (USD).

### B. Carbon Cost Savings Predictor
*   **Type**: Random Forest Regressor
*   **Goal**: Quantify potential savings from optimization.
*   **Target**: `Carbon_Cost_Savings_USD`
*   **Inputs (Features)**:
    *   `Energy_Demand_MWh`
    *   `Emission_Produced_tCO2`
    *   `Optimization_Scenario` (One-Hot: Low Demand, Price Surge)
    *   `Fuel_Type` (One-Hot)
*   **Output**: Estimated Cost Savings (USD).
