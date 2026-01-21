import streamlit as st [cite: 24]
import pandas as pd [cite: 40]
import numpy as np
import pickle [cite: 142]

# --- LOAD MODELS ---
try:
    model = pickle.load(open('diabetes_model.pkl', 'rb')) [cite: 142]
    scaler = pickle.load(open('scaler.pkl', 'rb')) [cite: 142]
except FileNotFoundError:
    st.error("Model files (.pkl) not found. Please upload them to your repository.") [cite: 148]

st.title("Diabetes Prediction System") [cite: 1]
st.write("Real-time Web Application for Risk Assessment") [cite: 23, 24]

# --- USER INPUT ---
# NOTE: Ensure you include EVERY feature used during model training.
# If your PIMA dataset used 8 features, you must collect all 8 here.
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        preg = st.number_input("Pregnancies", value=0)
        glucose = st.number_input("Glucose Level", value=120) [cite: 34]
        bp = st.number_input("Blood Pressure", value=70) [cite: 35]
        skinthickness = st.number_input("Skin Thickness", value=20)
    with col2:
        insulin = st.number_input("Insulin Level", value=80) [cite: 36]
        bmi = st.number_input("BMI", value=25.0) [cite: 36]
        dpf = st.number_input("Diabetes Pedigree Function", value=0.5)
        age = st.number_input("Age", value=33) [cite: 37]
    
    submit = st.form_submit_state = st.form_submit_button("Predict Risk")

# --- PREDICTION LOGIC ---
if submit:
    # 1. Create the feature list in the EXACT order of your training CSV [cite: 31, 33]
    # Example order for PIMA: Preg, Gluc, BP, Skin, Ins, BMI, DPF, Age
    features = np.array([[preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]])
    
    try:
        # 2. Scale the features [cite: 55, 62]
        features_scaled = scaler.transform(features) [cite: 63]
        
        # 3. Predict [cite: 18, 115]
        prediction = model.predict(features_scaled) [cite: 73, 77, 79]
        probability = model.predict_proba(features_scaled) [cite: 115]
        
        # 4. Display Results [cite: 22, 110]
        st.subheader("Diagnostic Results")
        if prediction[0] == 1: [cite: 38]
            st.error(f"Prediction: Diabetes Likely (Confidence: {np.max(probability):.2%})") [cite: 39, 115]
        else:
            st.success(f"Prediction: No Diabetes Detected (Confidence: {np.max(probability):.2%})") [cite: 39, 115]
            
    except ValueError as e:
        st.error(f"Feature Mismatch: Your model expects a different number of inputs. {e}")

st.info("Ethical Disclaimer: This system is for educational purposes only.") [cite: 155, 156]
