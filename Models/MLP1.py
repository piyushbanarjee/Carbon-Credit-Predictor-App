import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import joblib


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for regression tasks.
    Optimized architecture with batch normalization and dropout for better generalization.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super().__init__()
        layers = []
        
        # Build hidden layers
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def main():

    # Load data
    df = pd.read_csv("../Dataset/Features.csv")
    
    """Train MLP model for emission prediction."""
    # Feature columns for prediction
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
        "Industry_Type_Steel"
    ]
    
    # Prepare data
    X = df[X_cols].astype(float).values
    y = df["Emission_Produced_tCO2"].values.reshape(-1, 1)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Model configuration
    input_features = X.shape[1]
    hidden_layers = [256, 128, 64]  # Optimized deeper architecture
    output_features = 1
    
    # Initialize model
    model = MLP(
        input_dim=input_features,
        hidden_dims=hidden_layers,
        output_dim=output_features,
        dropout_rate=0.3
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, verbose=True
    )
    
    # Training
    epochs = 5000
    best_loss = float('inf')
    patience_counter = 0
    patience = 200
    
    print("Starting training...")
    for epoch in range(epochs):
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "../Models_Trained/MLP_best.pt")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs, y_test)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
    
    # Final evaluation
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        print(f"\nFinal Validation Loss: {val_loss.item():.6f}")
    
    # Save model and scaler
    joblib.dump(model, "../Models_Trained/MLP_Emission.pkl")
    joblib.dump(scaler, "../Models_Trained/MLP_Emission_Scalar.pkl")
    print("Model and scaler saved successfully!")


if __name__ == "__main__":
    main()