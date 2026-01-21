import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved models
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model files not found. Ensure .pkl files are in the repository.")

st.title("Diabetes Prediction System")
st.write("Machine Learning Based Risk Assessment")

# Input fields matching the PIMA dataset features
with st.container():
    pregnancies = st.number_input("Pregnancies", value=0)
    glucose = st.number_input("Glucose Level", value=120)
    blood_pressure = st.number_input("Blood Pressure", value=70)
    skin_thickness = st.number_input("Skin Thickness", value=20)
    insulin = st.number_input("Insulin Level", value=80)
    bmi = st.number_input("BMI", value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
    age = st.number_input("Age", value=33)

if st.button("Predict"):
    # Create feature array in the exact order:
    features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                          insulin, bmi, dpf, age]])
    
    # Apply Scaling
    features_scaled = scaler.transform(features)
    
    # Prediction
    prediction = model.predict(features_scaled)
    
    if prediction[0] == 1:
        st.error("High Risk: Result indicates a likelihood of diabetes.")
    else:
        st.success("Low Risk: Result indicates no diabetes.")

st.caption("Developed by Tehreem Fatima | SS-CASE-IT")
