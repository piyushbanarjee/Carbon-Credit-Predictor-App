import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def train_evaluate_model(X, y, model_name, save_path):
    print(f"\n--- Training {model_name} ---")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    # Save Model
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
    return model

def main():
    print("Loading data...")
    try:
        df = pd.read_csv("../Dataset/Features.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("Dataset/Features.csv")
        except FileNotFoundError:
            print("Error: Dataset not found.")
            return

    # ==========================================
    # Model 1: Compliance Cost Predictor
    # ==========================================
    target_compliance = "Compliance_Cost_USD"
    
    # Features for Compliance Cost
    # Industry, Energy, Emissions, Price
    features_compliance = [
        "Energy_Demand_MWh",
        "Emission_Produced_tCO2",
        "Carbon_Price_USD_per_t"
    ]
    features_compliance.extend([col for col in df.columns if "Industry_Type" in col])
    
    if target_compliance in df.columns:
        print(f"\nTraining for Target: {target_compliance}")
        print(f"Features: {features_compliance}")
        
        X_comp = df[features_compliance].values
        y_comp = df[target_compliance].values
        
        train_evaluate_model(
            X_comp, 
            y_comp, 
            "Compliance Cost Model", 
            "../Models_Trained/RF_Compliance_Cost.pkl"
        )
    else:
        print(f"\nTarget {target_compliance} not found in dataset.")

    # ==========================================
    # Model 2: Carbon Cost Savings Predictor
    # ==========================================
    target_savings = "Carbon_Cost_Savings_USD"
    
    # Features for Savings
    # Optimization Scenarios, Fuel Type, Energy, Emissions
    features_savings = [
        "Energy_Demand_MWh",
        "Emission_Produced_tCO2"
    ]
    features_savings.extend([col for col in df.columns if "Optimization_Scenario" in col])
    features_savings.extend([col for col in df.columns if "Fuel_Type" in col])
    
    if target_savings in df.columns:
        print(f"\nTraining for Target: {target_savings}")
        print(f"Features: {features_savings}")
        
        X_save = df[features_savings].values
        y_save = df[target_savings].values
        
        train_evaluate_model(
            X_save, 
            y_save, 
            "Cost Savings Model", 
            "../Models_Trained/RF_Cost_Savings.pkl"
        )
    else:
        print(f"\nTarget {target_savings} not found in dataset.")

if __name__ == "__main__":
    main()
