import pandas as pd
import numpy as np

df = pd.read_csv("uncleandata.csv")

print("before cleaning:")
print(df.shape)
print(df.info())
print(df.describe())
print(df.isna().sum())


df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("after cleaning:")
print(df.shape)
print(df.info())
print(df.describe())
print(df.isna().sum())

df["Savings/Investments"] = np.where(df["Monthly Income"]<df["Savings/Investments"],df["Monthly Income"]*0.30,df["Savings/Investments"])

df["Rent/Housing"] = np.where(df["Rent/Housing"]>0.6*df["Monthly Income"],df["Monthly Income"]*0.4,df["Rent/Housing"])

cleaneddf = df.to_csv("cleaneddata.csv")




