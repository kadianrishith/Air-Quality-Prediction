import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load original dataset
data = pd.read_csv(
    "dataset/AirQualityUCI.csv",
    sep=";",
    decimal=","
)

# STEP 2: Remove empty columns and replace missing indicators
data.dropna(axis=1, how="all", inplace=True)
data.replace(-200, np.nan, inplace=True)

# STEP 3: Convert measurement columns to numeric
for column in data.columns:
    if column not in ["Date", "Time"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

# STEP 4: Select target
target = "CO(GT)"

# STEP 5: Select input features
features = [
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "T",
    "RH",
    "AH"
]

# STEP 6: Keep rows where target is available
data = data.dropna(subset=[target]).copy()

# STEP 7: Create feature matrix and target
X = data[features]
y = data[target]

# STEP 8: Chronological train-test split
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# STEP 9: Create machine learning pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# STEP 10: Train model
print("Training machine learning model...")
model.fit(X_train, y_train)

# STEP 11: Make predictions
predictions = model.predict(X_test)

# STEP 12: Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nMODEL EVALUATION RESULTS")
print("------------------------")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2 Score:", round(r2, 3))

# STEP 13: Save trained model
joblib.dump(model, "models/air_quality_model.pkl")

print("\nModel saved successfully!")
# STEP 14: Create output folder
os.makedirs("outputs", exist_ok=True)

# STEP 15: Compare actual and predicted values
plt.figure(figsize=(8, 5))

plt.scatter(y_test, predictions, alpha=0.5, color="blue")

plt.xlabel("Actual CO Concentration")
plt.ylabel("Predicted CO Concentration")

plt.title("Actual vs Predicted CO Concentration")

# Perfect prediction reference line
minimum = min(y_test.min(), predictions.min())
maximum = max(y_test.max(), predictions.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    color="red",
    linestyle="--"
)

plt.tight_layout()
plt.savefig("outputs/actual_vs_predicted.png")
plt.close()


# STEP 16: Residual error visualization
residuals = y_test - predictions

plt.figure(figsize=(8, 5))

plt.scatter(predictions, residuals, alpha=0.5)

plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted CO Concentration")
plt.ylabel("Prediction Error")

plt.title("Residual Error Plot")

plt.tight_layout()
plt.savefig("outputs/residual_plot.png")
plt.close()

print("\nEvaluation graphs saved successfully!")