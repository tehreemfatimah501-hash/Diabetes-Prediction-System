import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the trained model and scaler 
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Diabetes Prediction System")
st.write("Enter patient data for a real-time risk assessment.") [cite: 12]

# User Input Fields [cite: 33, 111]
glucose = st.number_input("Glucose Level")
bp = st.number_input("Blood Pressure")
bmi = st.number_input("BMI")
insulin = st.number_input("Insulin Level")
age = st.number_input("Age", min_value=1, max_value=120)

# Prediction Logic [cite: 115]
if st.button("Predict Risk"):
    features = np.array([[glucose, bp, bmi, insulin, age]]) # Add all 8 features here
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)[0][1]
    
    if prediction[0] == 1:
        st.error(f"High Risk of Diabetes (Confidence: {probability:.2%})")
    else:
        st.success(f"Low Risk of Diabetes (Confidence: {1-probability:.2%})")

st.info("Disclaimer: Educational purposes only. Not medical advice.") [cite: 156]
