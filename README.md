# Diabetes-Prediction-System
**Machine Learning Based Risk Assessment & Predictive Analytics** 

This repository contains a comprehensive healthcare application designed to predict the likelihood of diabetes in patients using machine learning. By leveraging patient medical data, the system provides early detection and proactive health management insights.

## Project Overview

Developed as part of the **Artificial Intelligence & Machine Learning** course at **SS-CASE-IT**. The project underscores the potential of AI in enhancing diagnostic capabilities and improving patient outcomes.

## Key Objectives

* 
**Accurate Risk Prediction**: Identifying diabetes risk through structured patient health records.


* 
**Comparative Analysis**: Evaluating multiple algorithms including Logistic Regression, Random Forest, and XGBoost.


* 
**Interactive Analytics**: Visualizing data distributions and model performance through dynamic dashboards.


* 
**Web Deployment**: Providing a real-time user interface via Streamlit.



## Dataset & Features

The system utilizes the **PIMA Indians Diabetes Dataset**. The primary features analyzed include:

* Glucose Level, Blood Pressure, BMI, Insulin Level, and Age.


* The target variable is a binary outcome () indicating the presence or absence of diabetes.



## Tech Stack & Workflow

1. Data Preprocessing 

* 
**Ingestion**: Local dataset ingestion using the Pandas library.


* 
**Cleaning**: Handling missing values via mean/median imputation and removing outliers.


* 
**Scaling**: Normalizing feature ranges using StandardScaler or MinMaxScaler.


* 
**Splitting**:  train-test split for robust evaluation.



2. Machine Learning Models 

* 
**Logistic Regression**: A baseline model for high interpretability.


* 
**Random Forest**: An ensemble method to handle non-linear patterns and reduce overfitting.


* 
**XGBoost**: Advanced boosting for high predictive performance.



3. Evaluation Metrics 

Models are assessed based on:

* Accuracy, Precision, Recall, and F1-Score.


* ROC-AUC Score and Confusion Matrix analysis.



## 🌐 Deployment

The final application is deployed live using **Streamlit**.

* 
**Saved Models**: Trained models are serialized as `.pkl` files for instant inference.


* 
**Visualization**: Interactive charts (ROC curves, Correlation Heatmaps) implemented with Plotly and Matplotlib.



## Ethical Disclaimer

This system is intended for **educational purposes only** and does not replace professional medical diagnosis or advice. It should not be used as the sole basis for healthcare decisions.

---

**Author:** Tehreem Fatima 

**Instructor:** Mr. Zeeshan Aslam 

**Date:** December 26, 2023 

---

Would you like me to generate a `LICENSE` file or a `requirements.txt` file for your repository as well?
