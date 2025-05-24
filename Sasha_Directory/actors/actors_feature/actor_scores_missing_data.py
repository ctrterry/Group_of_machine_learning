import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# load the filtered actor scores CSV
df = pd.read_csv('C:/Users/sasaa/OneDrive/Documents/GOLANG/src/MyVault/NOTES/UC-Davis/S25/ECS171/PROJECT/data/actor_scores_filtered.csv')

# check for missing values in each column
missing_data = df.isnull().sum()
print("Number of missing values in each column:")
print(missing_data)

# calculate the percentage of missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
print("\nPercentage of missing values in each column:")
print(missing_percentage)

# cisplay rows with any missing values
missing_rows = df[df.isnull().any(axis=1)]
print("\nSample of rows with missing values (first 5 rows):")
print(missing_rows.head())
