import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- LOAD UPDATED MODELS ---
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Updated model files not found. Please upload the new .pkl files.")

st.title("Diabetes Prediction System")
st.write("Real-time Risk Assessment using 5 Key Health Indicators")

# --- USER INPUT (Simplified) ---
glucose = st.number_input("Glucose Level", value=120)
bp = st.number_input("Blood Pressure", value=70)
bmi = st.number_input("BMI", value=25.0)
insulin = st.number_input("Insulin Level", value=80)
age = st.number_input("Age", value=33)

# --- PREDICTION LOGIC ---
if st.button("Predict Diabetes Risk"):
    # Create array with exactly 5 features in the correct order
    features = np.array([[glucose, bp, bmi, insulin, age]])
    
    # Transform using the updated scaler
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 1:
        st.error("High Risk: Potential likelihood of diabetes detected.")
    else:
        st.success("Low Risk: No immediate indicators of diabetes found.")


