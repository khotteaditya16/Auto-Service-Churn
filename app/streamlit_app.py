import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from xgboost import plot_importance  # Just if you want to add later

st.title("Customer Churn Prediction")

# Load model and scaler once
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('logistic_best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Feature names (update with your actual feature names)
feature_names = list(pd.read_csv(r"E:\Projects\churn\data\processed\churn_data_cleaned.csv").drop(columns=['customer_id','days_since_churn','churned']).columns)

st.sidebar.header("Input Customer Features")

def user_input_features():
    data = {}
    for feat in feature_names:
        # Assuming numeric features, you can adjust here for other types
        data[feat] = st.sidebar.number_input(f"{feat}", value=0.0)
    return pd.DataFrame([data])

input_df = user_input_features()

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
churn_prob = model.predict_proba(input_scaled)[:, 1][0]
churn_pred = model.predict(input_scaled)[0]

st.subheader("Prediction Result")
st.write(f"Predicted churn probability: **{churn_prob:.3f}**")
st.write(f"Predicted churn class: **{churn_pred}**")

st.markdown("---")
st.write("You can extend this app with more metrics and visualizations.")

