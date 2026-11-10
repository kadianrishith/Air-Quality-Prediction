
import streamlit as st
import joblib
import pandas as pd
import os

# Page configuration
st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌍",
    layout="centered"
)

# Load trained model
MODEL_PATH = "models/air_quality_model.pkl"

st.title("🌍 Air Quality Prediction System")

st.write(
    "Enter the air pollution and weather measurements "
    "to estimate Carbon Monoxide (CO) concentration."
)

if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Enter Environmental Parameters")

# Input fields
pt08 = st.number_input(
    "PT08.S1(CO) - Sensor Reading",
    min_value=0.0,
    value=1000.0
)

nmhc = st.number_input(
    "NMHC(GT) - Non-methane Hydrocarbons",
    min_value=0.0,
    value=150.0
)

c6h6 = st.number_input(
    "C6H6(GT) - Benzene",
    min_value=0.0,
    value=10.0
)

nox = st.number_input(
    "NOx(GT) - Nitrogen Oxides",
    min_value=0.0,
    value=150.0
)

no2 = st.number_input(
    "NO2(GT) - Nitrogen Dioxide",
    min_value=0.0,
    value=100.0
)

temperature = st.number_input(
    "Temperature (°C)",
    value=20.0
)

humidity = st.number_input(
    "Relative Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0
)

absolute_humidity = st.number_input(
    "Absolute Humidity",
    min_value=0.0,
    value=1.0
)

# Prediction button
if st.button("Predict Air Quality", type="primary"):

    input_data = pd.DataFrame([{
        "PT08.S1(CO)": pt08,
        "NMHC(GT)": nmhc,
        "C6H6(GT)": c6h6,
        "NOx(GT)": nox,
        "NO2(GT)": no2,
        "T": temperature,
        "RH": humidity,
        "AH": absolute_humidity
    }])

    prediction = model.predict(input_data)[0]

    st.success("Prediction completed!")

    st.metric(
        label="Estimated CO Concentration",
        value=f"{prediction:.3f} mg/m³"
    )

    st.info(
        "This is an estimated CO concentration, "
        "not an official AQI classification or safety assessment."
    )

st.divider()

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)
