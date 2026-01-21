import streamlit as st  # Fixes the NameError 
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Diabetes Prediction System", layout="wide") [cite: 1]

# --- LOAD TRAINED MODELS ---
# These files must be in your GitHub repository [cite: 142, 148]
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb')) [cite: 142]
    scaler = pickle.load(open('scaler.pkl', 'rb')) [cite: 63]
except FileNotFoundError:
    st.error("Error: Trained model (.pkl) or Scaler not found. Please upload them to GitHub.") [cite: 148]

# --- SIDEBAR: USER INPUT --- [cite: 111]
st.sidebar.header("Patient Medical Parameters") [cite: 10]
def get_user_input():
    # Indicators based on your project requirements [cite: 33, 34, 35, 36, 37]
    glucose = st.sidebar.slider('Glucose Level', 0, 200, 120) [cite: 34]
    bp = st.sidebar.slider('Blood Pressure', 0, 130, 70) [cite: 35]
    bmi = st.sidebar.slider('BMI', 0.0, 70.0, 25.0) [cite: 36]
    insulin = st.sidebar.slider('Insulin Level', 0, 850, 80) [cite: 36]
    age = st.sidebar.slider('Age', 1, 120, 33) [cite: 37]
    
    # Create a feature array (Order must match your training data)
    features = pd.DataFrame({
        'Glucose': [glucose],
        'BloodPressure': [bp],
        'BMI': [bmi],
        'Insulin': [insulin],
        'Age': [age]
    })
    return features

input_data = get_user_input()

# --- MAIN INTERFACE ---
st.title("Diabetes Prediction System") [cite: 1]
st.write("Machine Learning Based Risk Assessment & Predictive Analytics") [cite: 2]
st.write("Enter patient data for a real-time risk assessment.") [cite: 10, 24]

# --- PREDICTION LOGIC ---
if st.button("Predict Diabetes Risk"): [cite: 18]
    # Apply feature scaling as defined in your preprocessing phase [cite: 55, 62]
    scaled_data = scaler.transform(input_data) [cite: 63]
    
    # Generate prediction (0 or 1) [cite: 38]
    prediction = model.predict(scaled_data) [cite: 115]
    probability = model.predict_proba(scaled_data) [cite: 115]
    
    st.subheader("Analysis Results")
    if prediction[0] == 1:
        st.error(f"Prediction: Diabetes Detected (High Risk)") [cite: 38]
    else:
        st.success(f"Prediction: No Diabetes (Low Risk)") [cite: 38]
    
    st.write(f"**Model Confidence:** {np.max(probability)*100:.2f}%") [cite: 115]

# --- INTERACTIVE ANALYTICS --- [cite: 21, 117]
st.divider()
st.subheader("Interactive Data Visualization") [cite: 117]

# Load dataset for visualization tools [cite: 30, 40]
try:
    df = pd.read_csv('diabetes.csv') [cite: 31, 40]
    
    # Correlation Heatmap [cite: 122]
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', ax=ax) [cite: 123, 126]
    st.pyplot(fig) [cite: 136]
    
except FileNotFoundError:
    st.info("Upload 'diabetes.csv' to view interactive correlation maps.") [cite: 136]

# --- ETHICAL DISCLAIMER --- [cite: 155]
st.warning("Educational purposes only. This does not replace professional medical diagnosis.") [cite: 156, 157]
