import streamlit as st # [cite: 24]
import pandas as pd # [cite: 40]
import numpy as np
import pickle # [cite: 142]

# --- LOAD ASSETS ---
# Ensure these files are in your GitHub repository root [cite: 148, 154]
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb')) # [cite: 142]
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except Exception as e:
    st.error("Model files not found. Please upload .pkl files.")

# --- UI SETUP ---
st.title("Diabetes Prediction System") # [cite: 1]
st.write("Machine Learning Based Risk Assessment") # [cite: 2]

# --- USER INPUT ---
# Using sliders to collect medical parameters [cite: 111]
glucose = st.number_input("Glucose Level", value=120) # [cite: 34]
bp = st.number_input("Blood Pressure", value=70) # [cite: 35]
bmi = st.number_input("BMI", value=25.0) # [cite: 36]
insulin = st.number_input("Insulin Level", value=80) # [cite: 36]
age = st.number_input("Age", value=33) # [cite: 37]

# --- PREDICTION LOGIC ---
# This section must be indented correctly to avoid SyntaxError 
if st.button("Predict Diabetes Risk"): # 
    # 1. Arrange inputs into a 2D array [cite: 41]
    features = np.array([[glucose, bp, bmi, insulin, age]]) 
    
    # 2. Scale features using your saved scaler [cite: 63]
    features_scaled = scaler.transform(features)
    
    # 3. Generate prediction [cite: 10]
    prediction = model.predict(features_scaled) # [cite: 113]
    
    # 4. Display results [cite: 115]
    st.subheader("Results:")
    if prediction[0] == 1:
        st.error("High Risk: The system predicts a likelihood of diabetes.") # [cite: 38]
    else:
        st.success("Low Risk: The system predicts no diabetes.") # [cite: 38]

# --- DISCLAIMER ---
st.info("Ethical Disclaimer: Educational purposes only. Not for medical diagnosis.") # [cite: 156]
