import streamlit as st
import requests
import plotly.graph_objects as go

# --- Configuration ---
API_URL = st.secrets.get("API_URL", "http://<ACI_IP_OR_DNS>")  # Güncel deploy URL

# --- Feature Definitions ---
NUMERIC_FEATURES = {
    'hematocrit': {'default': 35.0, 'min': 0.0, 'max': 60.0, 'icon': '🩸'},
    'neutrophils': {'default': 50.0, 'min': 0.0, 'max': 100.0, 'icon': '⚪'},
    'sodium': {'default': 138.0, 'min': 100.0, 'max': 180.0, 'icon': '🧂'},
    'glucose': {'default': 100.0, 'min': 50.0, 'max': 500.0, 'icon': '🍬'},
    'bloodureanitro': {'default': 15.0, 'min': 0.0, 'max': 100.0, 'icon': '🫘'},
    'creatinine': {'default': 1.0, 'min': 0.0, 'max': 15.0, 'icon': '💧'},
    'bmi': {'default': 25.0, 'min': 10.0, 'max': 60.0, 'icon': '⚖️'},
    'pulse': {'default': 75.0, 'min': 40.0, 'max': 200.0, 'icon': '💓'},
    'respiration': {'default': 16.0, 'min': 8.0, 'max': 40.0, 'icon': '🫁'}
}

CATEGORICAL_FEATURES = {
    'rcount': {'default': '0', 'options': ['0','1','2','3','4','5+'], 'icon': '🔄'},
    'gender': {'default': 'F', 'options': ['M','F'], 'icon': '👤'},
    'dialysisrenalendstage': {'default': 'No', 'options': ['Yes','No'], 'icon': '🩺'},
    'asthma': {'default': 'No', 'options': ['Yes','No'], 'icon': '🫁'},
    'irondef': {'default': 'No', 'options': ['Yes','No'], 'icon': '🩸'},
    'pneum': {'default': 'No', 'options': ['Yes','No'], 'icon': '🦠'},
    'substancedependence': {'default': 'No', 'options': ['Yes','No'], 'icon': '💊'},
    'psychologicaldisordermajor': {'default': 'No', 'options': ['Yes','No'], 'icon': '🧠'},
    'depress': {'default': 'No', 'options': ['Yes','No'], 'icon': '😔'},
    'psychother': {'default': 'No', 'options': ['Yes','No'], 'icon': '🛋️'},
    'fibrosisandother': {'default': 'No', 'options': ['Yes','No'], 'icon': '🫁'},
    'malnutrition': {'default': 'No', 'options': ['Yes','No'], 'icon': '🍽️'},
    'hemo': {'default': 'No', 'options': ['Yes','No'], 'icon': '🩸'},
    'secondarydiagnosisnonicd9': {'default': 'No', 'options': ['Yes','No'], 'icon': '📋'},
    'discharged': {'default': 'B', 'options': ['A','B','C','D'], 'icon': '🚪'},
    'facid': {'default': '0', 'options': ['0','1','2','3','4'], 'icon': '🏥'}
}

st.set_page_config(page_title="ClinicOps - LoS Prediction", layout="wide", page_icon="🏥")
st.title("🏥 ClinicOps - Length of Stay Prediction")

input_data = {}

# --- Numeric sliders ---
cols = st.columns(3)
for idx, (f, cfg) in enumerate(NUMERIC_FEATURES.items()):
    with cols[idx % 3]:
        input_data[f] = st.slider(f, min_value=cfg['min'], max_value=cfg['max'], value=cfg['default'], step=0.1)

# --- Categorical selects ---
cols = st.columns(4)
for idx, (f, cfg) in enumerate(CATEGORICAL_FEATURES.items()):
    with cols[idx % 4]:
        input_data[f] = st.selectbox(f, options=cfg['options'], index=cfg['options'].index(cfg['default']))

# --- Predict ---
if st.button("🚀 Predict LoS"):
    payload = {k: float(v) if k in NUMERIC_FEATURES else str(v) for k, v in input_data.items()}
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.ok:
            los = response.json()['predicted_length_of_stay']
            st.success(f"Predicted Length of Stay: {los} days")
        else:
            st.error(f"Prediction failed: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")
