
import pandas as pd
import numpy as np

# STEP 1: Load dataset
data = pd.read_csv(
    "dataset/AirQualityUCI.csv",
    sep=";",
    decimal=","
)

print("Original dataset shape:", data.shape)

# STEP 2: Remove completely empty columns
data.dropna(axis=1, how="all", inplace=True)

# STEP 3: Replace missing value indicator
data.replace(-200, np.nan, inplace=True)

# STEP 4: Remove duplicate rows
data.drop_duplicates(inplace=True)

# STEP 5: Convert pollutant columns to numeric
for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")

# STEP 6: Check missing values
print("\nMissing values before cleaning:")
print(data.isnull().sum())

# STEP 7: Fill missing values with median
for column in data.columns:
    if data[column].isnull().any():
        median_value = data[column].median()

        if pd.notna(median_value):
            data[column] = data[column].fillna(median_value)

# STEP 8: Remove columns that are still completely empty
data.dropna(axis=1, how="all", inplace=True)

# STEP 9: Save cleaned dataset
data.to_csv("dataset/cleaned_air_quality.csv", index=False)

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nCleaned dataset shape:", data.shape)

print("\nDATA PREPROCESSING COMPLETED!")
