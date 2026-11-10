import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load dataset
data = pd.read_csv(
    "dataset/AirQualityUCI.csv",
    sep=";",
    decimal=","
)

# Step 2: Remove completely empty columns
data.dropna(axis=1, how="all", inplace=True)

# Step 3: Replace -200 with NaN
# In this dataset, -200 represents a missing measurement.
data.replace(-200, np.nan, inplace=True)

# Step 4: Display first five rows
print("\nFIRST FIVE ROWS:")
print(data.head())

# Step 5: Display dataset dimensions
print("\nDATASET SHAPE:")
print(data.shape)

# Step 6: Display column names
print("\nCOLUMN NAMES:")
print(data.columns.tolist())

# Step 7: Check missing values
print("\nMISSING VALUES:")
print(data.isnull().sum())

# Step 8: Display statistical information
print("\nSTATISTICAL SUMMARY:")
print(data.describe())

# Step 9: Check data types
print("\nDATA TYPES:")
print(data.dtypes)

# Step 10: Plot CO concentration distribution
if "CO(GT)" in data.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data["CO(GT)"].dropna(), bins=30, kde=True)
    plt.title("Distribution of Carbon Monoxide (CO)")
    plt.xlabel("CO Concentration")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("dataset/co_distribution.png")
    plt.show()

# Step 11: Correlation heatmap
numeric_data = data.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_data.corr(),
    cmap="coolwarm",
    annot=False
)
plt.title("Correlation Between Air Quality Parameters")
plt.tight_layout()
plt.savefig("dataset/correlation_heatmap.png")
plt.show()

print("\nDATA ANALYSIS COMPLETED!")