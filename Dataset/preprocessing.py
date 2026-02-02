import pandas as pd
df = pd.read_csv("originalDataset.csv")

# Creating new feature
df["Emission_Intensity"] = (
    df["Emission_Produced_tCO2"]/df["Energy_Demand_MWh"]
)
df = df.sort_values(["Company_ID", "Date"])

df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Industry_Type"],
    drop_first= True
)
df[df.select_dtypes(include="bool").columns] = (
    df.select_dtypes(include="bool").astype(int)
)

df.to_csv("carbon_trading_dataset_features.csv", index=False)