# app.py
import streamlit as st
import numpy as np
import pickle

# --- Load Ensemble Model & Scaler ---
try:
    model = pickle.load(open('diabetes_ensemble_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or scaler .pkl files not found!")

st.title("Diabetes Prediction System (Ensemble)")

st.write("""
Enter your medical details below to predict the likelihood of diabetes.
""")

# --- User Inputs ---
pregnancies = st.number_input("Number of Pregnancies", value=0, min_value=0)
glucose = st.number_input("Glucose Level", value=120)
bp = st.number_input("Blood Pressure", value=70)
skin = st.number_input("Skin Thickness", value=20)
insulin = st.number_input("Insulin Level", value=80)
bmi = st.number_input("BMI", value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
age = st.number_input("Age", value=33)

# --- Prediction ---
if st.button("Predict Diabetes Risk"):
    # Arrange features in correct order
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled)[0][1]  # probability of diabetes

    if prediction[0] == 1:
        st.error(f"High Risk: Potential likelihood of diabetes detected (Probability: {proba:.2f})")
    else:
        st.success(f"Low Risk: No immediate indicators of diabetes found (Probability: {proba:.2f})")
