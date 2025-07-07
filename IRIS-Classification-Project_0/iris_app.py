import streamlit as st
import numpy as np
import pickle
import os

# ------------------------------------------
# Load the logistic regression model safely
# ------------------------------------------
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "lr_tuned.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Could not find the model file 'lr_tuned.pkl'. Please ensure it's in the same directory.")
    st.stop()

try:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Could not find the scaler file 'scaler.pkl'. Please ensure it's in the same directory.")
    st.stop()

# ------------------------------------------
# App Title and Flower Image
# ------------------------------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/4/49/Iris_versicolor_3.jpg", width=700)
st.title("🌸 Iris Flower Species Prediction 🌸")

# ------------------------------------------
# Sidebar Inputs
# ------------------------------------------
st.sidebar.header("Input Features")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg", width=200)

sepal_length = st.sidebar.number_input(
    "Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.8, step=0.1
)
sepal_width = st.sidebar.number_input(
    "Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1
)
petal_length = st.sidebar.number_input(
    "Petal Length (cm)", min_value=0.0, max_value=10.0, value=4.0, step=0.1
)
petal_width = st.sidebar.number_input(
    "Petal Width (cm)", min_value=0.0, max_value=10.0, value=1.2, step=0.1
)


# ------------------------------------------
# Predict Button Logic
# ------------------------------------------
if st.button("🌼 Predict 🌼"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    try:
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        # Display result
        st.image("https://upload.wikimedia.org/wikipedia/commons/a/a7/Iris_setosa_2.jpg", width=400)
        st.success(f"🌺 The predicted species is: **{prediction[0]}** 🌺")

    except Exception as e:
        st.error(f"❌ An error occurred during prediction: {str(e)}")
