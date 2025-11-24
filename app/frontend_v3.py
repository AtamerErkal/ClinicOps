import streamlit as st
import requests
import plotly.graph_objects as go
import os

API_URL = os.getenv("CLINICOPS_API_URL", "http://localhost:8000")

NUMERIC_FEATURES = {
    'hematocrit': {'default': 35.0, 'min': 0.0, 'max': 60.0, 'icon': '🩸', 'unit': '%', 
                   'desc': "Red blood cell volume (30-50%)"},
    'neutrophils': {'default': 50.0, 'min': 0.0, 'max': 100.0, 'icon': '⚪', 'unit': '%',
                    'desc': "Neutrophils percentage (40-75%)"},
    'sodium': {'default': 138.0, 'min': 100.0, 'max': 180.0, 'icon': '🧂', 'unit': 'mEq/L',
               'desc': "Sodium level (135-145)"},
    'glucose': {'default': 100.0, 'min': 50.0, 'max': 500.0, 'icon': '🬬', 'unit': 'mg/dL',
                'desc': "Blood glucose (70-120)"},
    'bloodureanitro': {'default': 15.0, 'min': 0.0, 'max': 100.0, 'icon': '🫘', 'unit': 'mg/dL',
                       'desc': "Blood Urea Nitrogen"},
    'creatinine': {'default': 1.0, 'min': 0.0, 'max': 15.0, 'icon': '💧', 'unit': 'mg/dL',
                   'desc': "Creatinine level"},
    'bmi': {'default': 25.0, 'min': 10.0, 'max': 60.0, 'icon': '⚖️', 'unit': 'kg/m²',
            'desc': "Body Mass Index"},
    'pulse': {'default': 75.0, 'min': 40.0, 'max': 200.0, 'icon': '💓', 'unit': 'bpm',
              'desc': "Heart rate"},
    'respiration': {'default': 16.0, 'min': 8.0, 'max': 40.0, 'icon': '🫁', 'unit': '/min',
                    'desc': "Respiration rate"}
}

CATEGORICAL_FEATURES = {
    'rcount': {'default': '0', 'options': ['0','1','2','3','4','5+'], 'icon': '🔄', 'desc': "Prior visits"},
    'gender': {'default': 'F', 'options': ['M','F'], 'icon': '👤', 'desc': "Gender"},
    'dialysisrenalendstage': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🩺', 'desc': "Dialysis/ESRD"},
    'asthma': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🫁', 'desc': "Asthma"},
    'irondef': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🩸', 'desc': "Iron deficiency"},
    'pneum': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🦠', 'desc': "Pneumonia"},
    'substancedependence': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '💊', 'desc': "Substance dependence"},
    'psychologicaldisordermajor': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🧠', 'desc': "Psych disorder"},
    'depress': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '😔', 'desc': "Depression"},
    'psychother': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🛋️', 'desc': "Psychotherapy"},
    'fibrosisandother': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🫁', 'desc': "Lung fibrosis"},
    'malnutrition': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🍽️', 'desc': "Malnutrition"},
    'hemo': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '🩸', 'desc': "Hemorrhoids"},
    'secondarydiagnosisnonicd9': {'default': 0, 'options': [0, 1], 'labels': ['No', 'Yes'], 'icon': '📋', 'desc': "Secondary diagnosis"},
    'facid': {'default': 0, 'options': [0, 1, 2, 3, 4], 'labels': ['0', '1', '2', '3', '4'], 'icon': '🏥', 'desc': "Facility ID"}
}

st.set_page_config(page_title="ClinicOps - LoS Prediction", layout="wide", page_icon="🏥")

st.markdown("""
# 🏥 ClinicOps
**AI-Powered Length of Stay Prediction**

Predicts hospital stay length using MLflow + Azure MLOps pipeline.
""")

st.markdown("## Patient Data Input")
input_data = {}

# Numeric inputs
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

# Categorical inputs
st.markdown("### 📋 Medical History")
cols = st.columns(4)
for idx, (feature, cfg) in enumerate(CATEGORICAL_FEATURES.items()):
    with cols[idx % 4]:
        st.write(f"{cfg['icon']} **{feature.replace('_',' ').title()}**")
        st.caption(cfg['desc'])
        
        # Use labels if available, otherwise use options directly
        if 'labels' in cfg:
            display_options = cfg['labels']
            selected = st.selectbox(
                "Select",
                options=display_options,
                index=display_options.index(cfg['labels'][cfg['default']]),
                key=f"cat_{feature}"
            )
            # Map back to 0/1
            input_data[feature] = cfg['options'][display_options.index(selected)]
        else:
            # String options (rcount, gender)
            input_data[feature] = st.selectbox(
                "Select",
                options=cfg['options'],
                index=cfg['options'].index(cfg['default']),
                key=f"cat_{feature}"
            )

# Prediction
st.markdown("### 🎯 Prediction")
if st.button("🚀 Predict Length of Stay"):
    # Ensure correct types
    payload = {}
    for k, v in input_data.items():
        if k in NUMERIC_FEATURES:
            payload[k] = float(v)
        elif k in ['rcount', 'gender']:
            payload[k] = str(v)
        else:
            payload[k] = int(v)
    
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            los = result['predicted_length_of_stay']
            
            st.success(f"**Predicted Length of Stay: {los} days**")
            
            risk = "Low" if los < 3 else "Medium" if los < 7 else "High"
            st.metric("Risk Level", risk)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=los,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': 'Length of Stay (Days)'},
                gauge={
                    'axis': {'range': [0, 20]},
                    'bar': {'color': '#667eea'},
                    'steps': [
                        {'range': [0, 3], 'color': '#38ef7d'},
                        {'range': [3, 7], 'color': '#ffd89b'},
                        {'range': [7, 20], 'color': '#ff6b6b'}
                    ],
                    'threshold': {'line': {'color': 'red', 'width': 4}, 'value': 10}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Prediction failed: {response.status_code}")
            st.json(response.json())
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Analytics
st.markdown("### 📊 Data Overview")
col1, col2 = st.columns(2)

with col1:
    lab_values = {k.replace('_', ' ').title(): v for k, v in input_data.items() if k in NUMERIC_FEATURES}
    fig = go.Figure([go.Bar(
        x=list(lab_values.keys()),
        y=list(lab_values.values()),
        marker_color='rgba(102,126,234,0.7)',
        text=list(lab_values.values()),
        textposition='auto'
    )])
    fig.update_layout(title="Lab Values", height=400, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Count active conditions (value = 1)
    conditions = sum([1 for k, v in input_data.items() 
                     if k not in NUMERIC_FEATURES and k not in ['rcount', 'gender', 'facid'] 
                     and v == 1])
    total_conditions = len([k for k in CATEGORICAL_FEATURES.keys() 
                           if k not in ['rcount', 'gender', 'facid']])
    
    fig = go.Figure([go.Pie(
        labels=['Active Conditions', 'No Conditions'],
        values=[conditions, total_conditions - conditions],
        hole=0.3,
        marker_colors=['#ff6b6b', '#38ef7d']
    )])
    fig.update_layout(title="Medical Conditions", height=400)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray;">
🏥 ClinicOps | MLOps Healthcare System | Powered by MLflow & Azure
</div>
""", unsafe_allow_html=True)