import streamlit as st
import numpy as np
import pickle

# Load the logistic regression model
with open('lr_tuned.pkl', 'rb') as f:
    lr_tuned = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Title with a flower image
st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Iris_versicolor_3.jpg", width=700)
st.title("🌸 Iris Flower Species Prediction 🌸")

# Sidebar for user input
st.sidebar.header("Input Features")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", width=200)
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

# Predict button
if st.button("🌼 Predict 🌼"):
    # Prepare input data
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = lr_tuned.predict(input_data_scaled)

    # Display result with a flower image
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Iris_setosa_2.jpg", width=400)
    st.write(f"🌺 The predicted species is: **{prediction[0]}** 🌺")
