import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
df = pd.read_csv("RawData.csv")

# Creating new feature
df["Emission_Intensity"] = (
    df["Emission_Produced_tCO2"]/df["Energy_Demand_MWh"]
)
df = df.sort_values(["Company_ID", "Date"])

# Data Normalization
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols ]

# Apply z-score normalization (StandardScaler) to numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Alternative: Use minimax normalization (uncomment to use instead)
# scaler = MinMaxScaler()
# df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print(f"Normalized {len(numerical_cols)} numerical features: {numerical_cols}")

# One hot encoding
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Industry_Type", "Optimization_Scenario"],
    drop_first= True
)
df[df.select_dtypes(include="bool").columns] = (
    df.select_dtypes(include="bool").astype(int)
)

# Binary Encoding
df["Transaction_Type"] = df["Transaction_Type"].map({"Sell": 1, "Buy": 0}) 
df["Verification_Status"] = df["Verification_Status"].map({"Disputed": 0, "Verified": 1}) 

# Making dates more useful
df["Date"] = pd.to_datetime(df["Date"])

df["year"] = df["Date"].dt.year
df["month"] = df["Date"].dt.month
df["day"] = df["Date"].dt.day

df.drop(columns=["Date"], inplace=True)

df.to_csv("Features.csv", index=False)