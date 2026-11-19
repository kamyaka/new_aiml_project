
import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_wine

# Load saved model and scaler
model = joblib.load('svm_wine_model.pkl')
scaler = joblib.load('scaler_wine.pkl')

# Load Wine dataset information
wine = load_wine()
class_names = wine.target_names
feature_names = wine.feature_names

# Streamlit App UI
st.title("🍷 Wine Classification Prediction using SVM")

st.write(
    "Enter the wine chemical measurements below "
    "to predict the wine class."
)

# Sidebar information
st.sidebar.header("📘 About the Project")

st.sidebar.info(
    "This app uses a Support Vector Machine (SVM) "
    "model trained on the Wine dataset. "
    "It predicts whether the wine belongs to "
    "Class 0, Class 1, or Class 2."
)

st.sidebar.write("🍇 Make your prediction below!")

# User input fields
st.header("Enter Wine Measurements")

default_values = [
    13.0, 2.3, 2.4, 18.0, 100.0,
    2.5, 2.0, 0.3, 1.5, 5.0,
    1.0, 3.0, 1000.0
]

input_values = []

for feature, default in zip(feature_names, default_values):

    value = st.number_input(
        feature.replace("_", " ").capitalize(),
        min_value=0.0,
        value=float(default),
        format="%.4f"
    )

    input_values.append(value)

# Prediction
if st.button("🔍 Predict"):

    input_data = np.array([input_values])

    # Feature scaling
    scaled_input = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(scaled_input)[0]

    st.success(
        f"🍷 Predicted Wine Class: "
        f"**{class_names[prediction].capitalize()}**"
    )
