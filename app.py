import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Diabetes Prediction System", layout="wide")

# --- LOAD ASSETS ---
# Ensure these .pkl files are in your GitHub repository
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or Scaler file not found. Please ensure .pkl files are uploaded.")

# --- SIDEBAR: USER INPUT ---
st.sidebar.header("Patient Health Indicators")
st.sidebar.write("Adjust the values below to assess risk.") [cite: 111]

def user_input_features():
    glucose = st.sidebar.slider('Glucose Level', 0, 200, 120) [cite: 34]
    bp = st.sidebar.slider('Blood Pressure', 0, 130, 70) [cite: 35]
    bmi = st.sidebar.slider('BMI', 0.0, 70.0, 25.0) [cite: 36]
    insulin = st.sidebar.slider('Insulin Level', 0, 850, 80) [cite: 36]
    age = st.sidebar.slider('Age', 1, 120, 33) [cite: 37]
    
    # Matching the feature set expected by your model
    data = {'Glucose': glucose,
            'BloodPressure': bp,
            'BMI': bmi,
            'Insulin': insulin,
            'Age': age}
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- MAIN PANEL ---
st.title("Diabetes Prediction System") [cite: 1]
st.write("Machine Learning Based Risk Assessment & Predictive Analytics") [cite: 2]

# Display User Input
st.subheader("Current Patient Indicators")
st.write(input_df)

# --- PREDICTION LOGIC ---
if st.button("Run Diagnostic Analysis"): [cite: 13, 115]
    # Scaling the input
    input_scaled = scaler.transform(input_df)
    
    # Prediction
    prediction = model.predict(input_scaled) [cite: 38]
    prediction_proba = model.predict_proba(input_scaled)
    
    st.subheader("Analysis Outcome")
    if prediction[0] == 1:
        st.error(f"Prediction: High Likelihood of Diabetes")
    else:
        st.success(f"Prediction: Low Likelihood of Diabetes")
        
    st.write(f"**Confidence Score:** {np.max(prediction_proba)*100:.2f}%") [cite: 115]

# --- INTERACTIVE ANALYTICS ---
st.divider()
st.subheader("Interactive Health Indicator Analysis") [cite: 21, 117]

# Load dataset for visualization (ensure diabetes.csv is in repo)
try:
    df = pd.read_csv('diabetes.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Feature Correlation Heatmap**") [cite: 122]
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
    with col2:
        st.write("**Glucose Distribution vs Outcome**") [cite: 120]
        fig2, ax2 = plt.subplots()
        sns.histplot(data=df, x='Glucose', hue='Outcome', kde=True, ax=ax2)
        st.pyplot(fig2)
        
except Exception as e:
    st.warning("Upload 'diabetes.csv' to enable interactive data visualizations.")

# --- FOOTER ---
st.info("**Ethical Disclaimer:** This system is for educational purposes only and does not replace professional medical diagnosis.") [cite: 156]
st.caption(f"Submitted By: Tehreem Fatima | Instructor: Mr. Zeeshan Aslam | SS-CASE-IT") [cite: 4, 6]
