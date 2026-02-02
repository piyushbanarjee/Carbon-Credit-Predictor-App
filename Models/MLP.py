import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import joblib

df = pd.read_csv("Dataset/Features.csv")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
inputFeatures = 10
hiddenLayer = 64
output = 1
model = MLP(
    input_dim= inputFeatures,
    hidden_dim= hiddenLayer,
    output_dim= output
)

# For Regression
criterion = nn.MSELoss()
# For Classification
# criterion = nn.BCEWithLogitsLoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)

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

X = df[X_cols].astype(float).values
y = df["Emission_Produced_tCO2"].values.reshape(-1, 1)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

epochs = 5000

for epoch in range(epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs,y)

    #  Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

joblib(model, "..\Models_Trained\MLP.pkl")