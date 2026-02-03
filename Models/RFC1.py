import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("Loading data...")
    try:
        df = pd.read_csv("../Dataset/Features.csv")
    except FileNotFoundError:
        df = pd.read_csv("Dataset/Features.csv")

    # Feature Engineering: Net Position
    # Net Position = Allowance - Produced. 
    # Positive = Surplus (Likely Sell). Negative = Deficit (Likely Buy).
    # Since features might be scaled, we subtract raw columns if they are scaled or just simple subtraction.
    # Looking at data snippet, both seem to be scaled (small float values). 
    # Subtraction is still valid linear operation.
    df['Net_Position'] = df['Emission_Allowance_tCO2'] - df['Emission_Produced_tCO2']
    
    # Feature Engineering: Price Interaction
    # Economic incentive: Do we buy when price is low? Sell when high?
    df['Net_Position_Price_ interaction'] = df['Net_Position'] * df['Carbon_Price_USD_per_t']

    feature_cols = [
        "Energy_Demand_MWh",
        "Emission_Produced_tCO2",
        "Emission_Allowance_tCO2",
        "Carbon_Price_USD_per_t",
        "Net_Position",
        "Net_Position_Price_ interaction"
    ]
    
    # Add all Fuel_Type columns
    fuel_cols = [col for col in df.columns if "Fuel_Type" in col]
    feature_cols.extend(fuel_cols)
    
    print(f"Using features: {feature_cols}")
    
    X = df[feature_cols]
    
    # Diagnostic: Correlation Analysis
    if "Transaction_Type" in df.columns:
        corr_matrix = df[feature_cols + ['Transaction_Type']].corr()
        print("\nCorrelation with Transaction_Type:")
        print(corr_matrix['Transaction_Type'].sort_values(ascending=False))
        
        # Check alignment with Target_Trade_Action if available
        if "Target_Trade_Action" in df.columns:
             print("\nCorrelation with Target_Trade_Action:")
             print(df[feature_cols + ['Target_Trade_Action']].corr()['Target_Trade_Action'].sort_values(ascending=False))

    
    y = df["Transaction_Type"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train Random Forest
    print("\nTraining Random Forest Classifier...")
    # Increased estimators and depth to capture complex interactions
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    
    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature Importances:")
    for i in range(len(feature_cols)):
        print(f"{feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")

    # Save Model
    joblib.dump(rf_model, "../Models_Trained/RF_BuySell_Predictor.pkl")
    print("\nModel saved to ../Models_Trained/RF_BuySell_Predictor.pkl")

if __name__ == "__main__":
    main()
