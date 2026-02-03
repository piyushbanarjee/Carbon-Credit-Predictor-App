import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# ==================== Configuration ====================
# Data
DATA_PATH = "../Dataset/Features.csv"
FEATURE_COLS = [
    "Energy_Demand_MWh", "Emission_Allowance_tCO2", "Carbon_Price_USD_per_t",
    "Emission_Intensity", "Fuel_Type_Mixed Fuel", "Fuel_Type_Natural Gas",
    "Fuel_Type_Renewable", "Industry_Type_Energy", "Industry_Type_Manufacturing",
    "Industry_Type_Steel"
]
TARGET_COL = "Emission_Produced_tCO2"

# Model architecture
HIDDEN_LAYERS = [256, 128, 64]
DROPOUT_RATE = 0.3

# Training hyperparameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 5000
EARLY_STOP_PATIENCE = 200
LR_SCHEDULER_FACTOR = 0.5

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ==================== Load & Prepare Data ====================
df = pd.read_csv(DATA_PATH)
X = df[FEATURE_COLS].astype(float).values
y = df[TARGET_COL].values.reshape(-1, 1)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ==================== Build Model ====================
input_dim = X.shape[1]

model = nn.Sequential(
    # Layer 1: input_dim -> 256
    nn.Linear(input_dim, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(),
    nn.Dropout(DROPOUT_RATE),
    
    # Layer 2: 256 -> 128
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.LeakyReLU(),
    nn.Dropout(DROPOUT_RATE),
    
    # Layer 3: 128 -> 64
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.LeakyReLU(),
    nn.Dropout(DROPOUT_RATE),
    
    # Output layer: 64 -> 1
    nn.Linear(64, 1)
)

# ==================== Training Setup ====================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_SCHEDULER_FACTOR, 
    patience=EARLY_STOP_PATIENCE, verbose=True
)

# ==================== Training Loop ====================
print("Starting training...")
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    
    # Early stopping
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
        torch.save(model.state_dict(), "../Models_Trained/MLP_best.pt")
    else:
        patience_counter += 1
    
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    if (epoch + 1) % 100 == 0:
        with torch.no_grad():
            val_loss = criterion(model(X_test), y_test)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train: {loss.item():.6f}, Val: {val_loss.item():.6f}")

# ==================== Final Evaluation & Save ====================
with torch.no_grad():
    final_val_loss = criterion(model(X_test), y_test)
    print(f"\nFinal Validation Loss: {final_val_loss.item():.6f}")

# joblib.dump(model, "../Models_Trained/MLP_Emission.pkl")
# joblib.dump(scaler, "../Models_Trained/MLP_Emission_Scalar.pkl")
# print("Model and scaler saved successfully!")