# app/frontend.py

import streamlit as st
import pandas as pd
import requests
import json
import numpy as np

# --- CRITICAL: UPDATE THIS WITH YOUR ACI DNS URL ---
# Example: "http://klinikops-api-f206c78c.germanywestcentral.azurecontainer.io"
API_URL = "http://klinikops-api-3f8a5d5ce66b810b73d4390df453f587b2fc6f30.germanywestcentral.azurecontainer.io" 

# Schema reflecting the 26 features expected by the deployed model (PatientData Pydantic Model)
FEATURE_DEFAULTS = {
    'rcount': 0, 'gender': 'F', 'dialysis': '0', 'mcd': '0', 'ecodes': '0',
    'hmo': '0', 'health': '0', 'age': 55, 'eclaim': 1.0, 'pridx': 1,
    'sdimd': 1, 'procedure': '0', 'pcode': '0', 'zid': '0', 'plos': 1.0, 
    'clmds': 1, 'disch': 'B', 'orproc': '0', 'comorb': '0', 'diag': '0',
    'ipros': '0', 'DRG': '0', 'last': '0', 'PG': '0', 'payer': '0',
    'primaryphy': '0'
}

st.set_page_config(page_title="KlinikOps Prediction Service", layout="wide")

st.title("🏥 ClinicOps: Hospital Length of Stay Prediction")
st.markdown("Connects to the FastAPI model deployed on Azure Container Instances (ACI).")

# --- API Health Check ---
try:
    health_response = requests.get(f"{API_URL}/health")
    health_data = health_response.json()
    model_loaded = health_data.get("model_loaded", False)
    api_version = health_data.get('api_version', 'N/A')
    
    if model_loaded:
        st.success(f"✅ API Status: Online and Model Loaded (API Version: {api_version})")
    else:
        # Warning for the current model loading issue
        st.warning(f"⚠️ API Status: Online, but **Model Failed to Load** (model_loaded: false). Predictions will result in an error.")
        st.info("The issue is related to the model loading from Azure Blob Storage, but the CI/CD pipeline is fully functional.")

except requests.exceptions.ConnectionError:
    st.error(f"❌ Could not connect to the API. Please check the URL ({API_URL}) and ACI status.")
    st.stop()


# --- Prediction Form ---
st.header("Enter Patient Features (26 Features)")
input_data = {}

# Split the form into two columns
cols = st.columns(2)
features_list = list(FEATURE_DEFAULTS.keys())

# Column 1: Demographic & Initial Data
with cols[0]:
    st.subheader("Demographic & Initial Data")
    for i in range(0, len(features_list) // 2):
        feature = features_list[i]
        default_val = FEATURE_DEFAULTS[feature]
        
        if isinstance(default_val, int):
            input_data[feature] = st.number_input(f"{feature.capitalize()}", value=default_val, key=feature, step=1)
        elif isinstance(default_val, float):
            input_data[feature] = st.number_input(f"{feature.capitalize()}", value=default_val, key=feature, format="%.2f")
        else:
            input_data[feature] = st.text_input(f"{feature.capitalize()} (Categorical)", value=default_val, key=feature)

# Column 2: Procedure & Codes
with cols[1]:
    st.subheader("Procedure & Diagnostic Codes")
    for i in range(len(features_list) // 2, len(features_list)):
        feature = features_list[i]
        default_val = FEATURE_DEFAULTS[feature]
        
        if isinstance(default_val, int):
            input_data[feature] = st.number_input(f"{feature.capitalize()}", value=default_val, key=feature, step=1)
        elif isinstance(default_val, float):
            input_data[feature] = st.number_input(f"{feature.capitalize()}", value=default_val, key=feature, format="%.2f")
        else:
            input_data[feature] = st.text_input(f"{feature.capitalize()} (Categorical)", value=default_val, key=feature)


if st.button("Get Prediction"):
    try:
        # Payload to be sent to the FastAPI service
        payload = input_data

        with st.spinner('Sending prediction request to the API...'):
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                prediction = response.json()
                st.balloons()
                st.success(f"**Predicted Length of Stay: {prediction['predicted_length_of_stay']} {prediction['unit']}**")
                
            elif response.status_code == 503:
                # Expected error due to model_loaded: false
                st.error("API Error (503 Service Unavailable): The model failed to load. Please check ACI logs for the exact MLflow error.")
            else:
                st.error(f"Unexpected Error Code from API: {response.status_code}. Detail: {response.text}")

    except Exception as e:
        st.error(f"An error occurred during prediction request: {e}")