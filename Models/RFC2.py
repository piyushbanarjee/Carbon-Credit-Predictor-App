import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def add_cyclical_features(df, col_name, period):
    df[f'{col_name}_sin'] = np.sin(2 * np.pi * df[col_name] / period)
    df[f'{col_name}_cos'] = np.cos(2 * np.pi * df[col_name] / period)
    return df

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

    # Construct Target
    # 0 = Normal
    # 1 = Low Demand
    # 2 = Price Surge
    targets = np.zeros(len(df), dtype=int)
    if 'Optimization_Scenario_Low_Demand' in df.columns:
        targets[df['Optimization_Scenario_Low_Demand'] == 1] = 1
    if 'Optimization_Scenario_Price_Surge' in df.columns:
        targets[df['Optimization_Scenario_Price_Surge'] == 1] = 2
        
    y = targets
    
    print("\nClass Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts)))
    # {0: 'Normal', 1: 'Low Demand', 2: 'Price Surge'}

    # Feature Engineering
    # 1. Cyclical Month
    if 'month' in df.columns:
        df = add_cyclical_features(df, 'month', 12)
        
    # 2. Select Features
    feature_cols = ["Energy_Demand_MWh"]
    if 'month_sin' in df.columns:
        feature_cols.extend(['month_sin', 'month_cos'])
    else:
        feature_cols.append('month')
        
    fuel_cols = [col for col in df.columns if "Fuel_Type" in col]
    industry_cols = [col for col in df.columns if "Industry_Type" in col]
    
    feature_cols.extend(fuel_cols)
    feature_cols.extend(industry_cols)
    
    print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
    
    X = df[feature_cols].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    print("\nTraining Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # Evaluation
    print("\nEvaluating model...")
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Low Demand', 'Price Surge']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature Importance
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\nFeature Importances:")
    for i in range(len(feature_cols)):
        print(f"{feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
        
    # Save Model
    joblib.dump(rf_model, "../Models_Trained/RF_Optimization_Scenario.pkl")
    print("\nModel saved to ../Models_Trained/RF_Optimization_Scenario.pkl")

if __name__ == "__main__":
    main()
