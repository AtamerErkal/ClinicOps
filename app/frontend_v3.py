# app/frontend_modern_clinicops.py
import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime
import os

# --- Configuration ---
API_URL = os.getenv("CLINICOPS_API_URL", "http://localhost:8000")

# --- Feature Definitions ---
NUMERIC_FEATURES = {
    'hematocrit': {'default': 35.0, 'min': 0.0, 'max': 60.0, 'icon': '🩸', 'unit': '%', 
                   'desc': "Red blood cell volume in blood (30-50%)."},
    'neutrophils': {'default': 50.0, 'min': 0.0, 'max': 100.0, 'icon': '⚪', 'unit': '%',
                    'desc': "Percentage of neutrophils (infection indicator, 40-75%)."},
    'sodium': {'default': 138.0, 'min': 100.0, 'max': 180.0, 'icon': '🧂', 'unit': 'mEq/L',
               'desc': "Sodium level (135-145 mEq/L)."},
    'glucose': {'default': 100.0, 'min': 50.0, 'max': 500.0, 'icon': '🍬', 'unit': 'mg/dL',
                'desc': "Blood glucose level (70-120 mg/dL)."},
    'bloodureanitro': {'default': 15.0, 'min': 0.0, 'max': 100.0, 'icon': '🫘', 'unit': 'mg/dL',
                       'desc': "Blood Urea Nitrogen (kidney function indicator)."},
    'creatinine': {'default': 1.0, 'min': 0.0, 'max': 15.0, 'icon': '💧', 'unit': 'mg/dL',
                   'desc': "Creatinine level (kidney function indicator)."},
    'bmi': {'default': 25.0, 'min': 10.0, 'max': 60.0, 'icon': '⚖️', 'unit': 'kg/m²',
            'desc': "Body Mass Index."},
    'pulse': {'default': 75.0, 'min': 40.0, 'max': 200.0, 'icon': '💓', 'unit': 'bpm',
              'desc': "Heart rate (beats per minute)."},
    'respiration': {'default': 16.0, 'min': 8.0, 'max': 40.0, 'icon': '🫁', 'unit': '/min',
                    'desc': "Respiration rate (breaths per minute)."}
}

CATEGORICAL_FEATURES = {
    'rcount': {'default': '0', 'options': ['0','1','2','3','4','5+'], 'icon': '🔄', 'desc': "Number of prior hospital visits."},
    'gender': {'default': 'F', 'options': ['M','F'], 'icon': '👤', 'desc': "Patient gender."},
    'dialysisrenalendstage': {'default': 'No', 'options': ['Yes','No'], 'icon': '🩺', 'desc': "History of dialysis or end-stage renal disease."},
    'asthma': {'default': 'No', 'options': ['Yes','No'], 'icon': '🫁', 'desc': "History of asthma."},
    'irondef': {'default': 'No', 'options': ['Yes','No'], 'icon': '🩸', 'desc': "History of iron deficiency."},
    'pneum': {'default': 'No', 'options': ['Yes','No'], 'icon': '🦠', 'desc': "History of pneumonia."},
    'substancedependence': {'default': 'No', 'options': ['Yes','No'], 'icon': '💊', 'desc': "History of substance dependence."},
    'psychologicaldisordermajor': {'default': 'No', 'options': ['Yes','No'], 'icon': '🧠', 'desc': "History of major psychological disorder."},
    'depress': {'default': 'No', 'options': ['Yes','No'], 'icon': '😔', 'desc': "History of depression."},
    'psychother': {'default': 'No', 'options': ['Yes','No'], 'icon': '🛋️', 'desc': "History of psychotherapy."},
    'fibrosisandother': {'default': 'No', 'options': ['Yes','No'], 'icon': '🫁', 'desc': "History of lung fibrosis or other lung disease."},
    'malnutrition': {'default': 'No', 'options': ['Yes','No'], 'icon': '🍽️', 'desc': "History of malnutrition."},
    'hemo': {'default': 'No', 'options': ['Yes','No'], 'icon': '🩸', 'desc': "History of hemorrhoids."},
    'secondarydiagnosisnonicd9': {'default': 'No', 'options': ['Yes','No'], 'icon': '📋', 'desc': "Secondary diagnosis (non-ICD9)."},
    'discharged': {'default': 'B', 'options': ['A','B','C','D'], 'icon': '🚪', 'desc': "Discharge status."},
    'facid': {'default': '0', 'options': ['0','1','2','3','4'], 'icon': '🏥', 'desc': "Facility ID (hospital/clinic)."}
}


# --- Streamlit Page ---
st.set_page_config(page_title="ClinicOps - LoS Prediction", layout="wide", page_icon="🏥")

st.markdown("""
# 🏥 ClinicOps
**AI-Powered Length of Stay Prediction System**

ClinicOps predicts the expected hospital stay length for patients based on laboratory, vitals, demographics, and medical history.  
This is a full **MLOps project**: model training, deployment, and inference are handled automatically via MLflow and Azure.
""")

# --- User Inputs ---
st.markdown("## Enter Patient Data")

input_data = {}

# Numeric Inputs
st.markdown("### 🔬 Laboratory Values & Vitals")
cols = st.columns(3)
for idx, (feature, cfg) in enumerate(NUMERIC_FEATURES.items()):
    with cols[idx % 3]:
        st.write(f"{cfg['icon']} **{feature.replace('_',' ').title()}**")
        st.caption(cfg['desc'])
        input_data[feature] = st.slider(
            "Value",
            min_value=cfg['min'],
            max_value=cfg['max'],
            value=cfg['default'],
            step=0.1,
            key=f"num_{feature}"
        )

# Categorical Inputs
st.markdown("### 📋 Medical History & Demographics")
cols = st.columns(4)
for idx, (feature, cfg) in enumerate(CATEGORICAL_FEATURES.items()):
    with cols[idx % 4]:
        st.write(f"{cfg['icon']} **{feature.replace('_',' ').title()}**")
        st.caption(cfg['desc'])
        input_data[feature] = st.selectbox(
            "Select",
            options=cfg['options'],
            index=cfg['options'].index(cfg['default']),
            key=f"cat_{feature}"
        )

# --- Prediction ---
st.markdown("### 🎯 Predicted Length of Stay")
if st.button("🚀 Predict LoS"):
    payload = {k: float(v) if k in NUMERIC_FEATURES else str(v) for k,v in input_data.items()}
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            los = result['predicted_length_of_stay']

            st.success(f"Predicted Length of Stay: **{los} days**")

            # Risk interpretation
            risk = "Low" if los<3 else "Medium" if los<7 else "High"
            st.metric("Complexity Risk", risk)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=los,
                domain={'x':[0,1],'y':[0,1]},
                title={'text':'Length of Stay (Days)'},
                gauge={'axis':{'range':[0,20]},
                       'bar':{'color':'#667eea'},
                       'steps':[{'range':[0,3],'color':'#38ef7d'},
                               {'range':[3,7],'color':'#ffd89b'},
                               {'range':[7,20],'color':'#ff6b6b'}],
                       'threshold':{'line':{'color':'red','width':4},'value':10}}
            ))
            fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=20))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.error(f"Prediction failed: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Analytics ---
st.markdown("### 📊 Patient Data Overview")
col1, col2 = st.columns(2)

# Numeric overview
with col1:
    lab_values = {k.replace('_',' ').title(): v for k,v in input_data.items() if k in NUMERIC_FEATURES}
    fig = go.Figure([go.Bar(x=list(lab_values.keys()), y=list(lab_values.values()),
                            marker_color='rgba(102,126,234,0.7)', text=list(lab_values.values()), textposition='auto')])
    fig.update_layout(title="Lab Values", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Categorical overview
with col2:
    conditions = sum([1 for k,v in input_data.items() if k in CATEGORICAL_FEATURES and v=='Yes'])
    fig = go.Figure([go.Pie(labels=['Active Conditions','No Conditions'], 
                            values=[conditions,len(CATEGORICAL_FEATURES)-conditions],
                            hole=0.3, marker_colors=['#ff6b6b','#38ef7d'])]))
    fig.update_layout(title="Medical Conditions Overview", height=400)
    st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray;">
🏥 ClinicOps | MLOps Healthcare System | Powered by MLflow & Azure  
© 2025 ClinicOps. All rights reserved.
</div>
""", unsafe_allow_html=True)