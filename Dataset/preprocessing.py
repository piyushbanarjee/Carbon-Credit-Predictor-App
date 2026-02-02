import pandas as pd
df = pd.read_csv("RawData.csv")

# Creating new feature
df["Emission_Intensity"] = (
    df["Emission_Produced_tCO2"]/df["Energy_Demand_MWh"]
)
df = df.sort_values(["Company_ID", "Date"])

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
#  making company ids more useful
freq = df["Company_ID"].value_counts()
df["company_freq"] = df["Company_ID"].map(freq)
df.drop(columns=["Company_ID"], inplace=True)

df.to_csv("Dataset\Features2.csv", index=False)