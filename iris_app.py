# Streamlit app code
import streamlit as st
import numpy as np
import joblib
import os

# Check if the model file exists
if os.path.exists('lr_tuned.joblib'):
    # Load the logistic regression model
    lr_tuned = joblib.load('lr_tuned.joblib')
else:
    lr_tuned = None

# Check if the scaler file exists
if os.path.exists('scaler.joblib'):
    # Load the scaler
    scaler = joblib.load('scaler.joblib')
else:
    scaler = None

# Title
st.title("🌸 Iris Flower Species Prediction 🌸")

# Description
st.markdown("""
This application predicts the species of an Iris flower based on its measurements. 
Simply input the flower's features below and click **Predict** to see the result.
""")

# Input fields for user input
st.header("Input Measurements")
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

# Predict button
if st.button("Predict"):
    if lr_tuned is None or scaler is None:
        st.error("Model or scaler file is missing. Please ensure all required files are available.")
    else:
        # Prepare input data
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = lr_tuned.predict(input_data_scaled)
        prediction_proba = lr_tuned.predict_proba(input_data_scaled)
        confidence = np.max(prediction_proba) * 100

        # Display result
        st.success(f"🌟 The predicted species is: **{prediction[0]}** 🌟")
        st.info(f"Confidence: {confidence:.2f}%")