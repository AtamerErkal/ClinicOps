import streamlit as st
import requests
import plotly.graph_objects as go
import os

# Mobile & ACI compatibility
if "AZURE_CONTAINER" in os.environ or os.getenv("STREAMLIT_SERVER_PORT"):
    os.environ["STREAMLIT_SERVER_PORT"] = "7860"

# Streamlit config for mobile
st.set_page_config(
    page_title="ClinicOps - Length of Stay Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto"
)

# API URL - support both local and deployed
API_URL = os.getenv("CLINICOPS_API_URL", "http://localhost:8000")

# ==============================================================================
# FEATURE DEFINITIONS
# ==============================================================================
NUMERIC_FEATURES = {
    'hematocrit':       {'default': 35.0, 'min': 0.0,  'max': 60.0,  'icon': '🩸', 'unit': '%',      'normal_min': 30.0, 'normal_max': 50.0, 'desc': "Red blood cell volume"},
    'neutrophils':      {'default': 50.0, 'min': 0.0,  'max': 100.0, 'icon': '⚪', 'unit': '%',      'normal_min': 40.0, 'normal_max': 75.0, 'desc': "Neutrophils percentage"},
    'sodium':           {'default': 138.0,'min': 100.0,'max': 180.0, 'icon': '🧂', 'unit': 'mEq/L',  'normal_min': 135.0,'normal_max': 145.0,'desc': "Sodium level"},
    'glucose':          {'default': 100.0,'min': 50.0, 'max': 500.0, 'icon': '🍬', 'unit': 'mg/dL',  'normal_min': 70.0, 'normal_max': 120.0,'desc': "Blood glucose"},
    'bloodureanitro':   {'default': 15.0, 'min': 0.0,  'max': 100.0, 'icon': '🫘', 'unit': 'mg/dL',  'normal_min': 7.0,  'normal_max': 20.0, 'desc': "Blood Urea Nitrogen (BUN)"},
    'creatinine':       {'default': 1.0,  'min': 0.0,  'max': 15.0,  'icon': '💧', 'unit': 'mg/dL',  'normal_min': 0.6,  'normal_max': 1.2,  'desc': "Creatinine level"},
    'bmi':              {'default': 25.0, 'min': 10.0, 'max': 60.0,  'icon': '⚖️', 'unit': 'kg/m²',  'normal_min': 18.5, 'normal_max': 24.9, 'desc': "Body Mass Index"},
    'pulse':            {'default': 75.0, 'min': 40.0, 'max': 200.0, 'icon': '💓', 'unit': 'bpm',    'normal_min': 60.0, 'normal_max': 100.0,'desc': "Heart rate"},
    'respiration':      {'default': 16.0, 'min': 8.0,  'max': 40.0,  'icon': '🫁', 'unit': '/min',   'normal_min': 12.0, 'normal_max': 20.0, 'desc': "Respiratory rate"}
}

CATEGORICAL_FEATURES = {
    'rcount':                   {'default': '0',   'options': ['0','1','2','3','4','5+'], 'labels': ['0', '1', '2', '3', '4', '5+'], 'icon': '🔄', 'desc': "Prior hospital visits"},
    'gender':                   {'default': 'F',   'options': ['F','M'],                  'labels': ['Female', 'Male'],                 'icon': '👤', 'desc': "Gender"},
    'facid':                    {'default': 'A',   'options': ['A','B','C','D','E'],      'labels': ['Facility A','Facility B','Facility C','Facility D','Facility E'], 'icon': '🏥', 'desc': "Facility ID"},
    'secondarydiagnosisnonicd9':{'default': '0',   'options': [str(i) for i in range(11)], 'labels': [str(i) for i in range(11)], 'icon': '📋', 'desc': "Number of secondary diagnoses"},
    'dialysisrenalendstage':    {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '💉', 'desc': "Dialysis / End-stage renal disease"},
    'asthma':                   {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '🫁', 'desc': "Asthma"},
    'irondef':                  {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '🩸', 'desc': "Iron deficiency anemia"},
    'pneum':                    {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '🦠', 'desc': "Pneumonia"},
    'substancedependence':      {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '💊', 'desc': "Substance dependence"},
    'psychologicaldisordermajor':{'default': 0,    'options': [0,1], 'labels': ['No','Yes'], 'icon': '🧠', 'desc': "Major psychological disorder"},
    'depress':                  {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '😔', 'desc': "Depression"},
    'psychother':               {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '🛋️', 'desc': "Psychotherapy"},
    'fibrosisandother':         {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '🫁', 'desc': "Lung fibrosis"},
    'malnutrition':             {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '🍽️', 'desc': "Malnutrition"},
    'hemo':                     {'default': 0,     'options': [0,1], 'labels': ['No','Yes'], 'icon': '🩸', 'desc': "Hemorrhoids"}
}

# ==============================================================================
# PAGE CONFIG & MODERN STYLING
# ==============================================================================
st.set_page_config(
    page_title="ClinicOps – AI-Powered Hospital Length of Stay Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    .block-container {padding-top: 1.5rem;}
    
    .stExpander {
        border: 1px solid rgba(100,100,100,0.2);
        border-radius: 12px;
        background: rgba(255,255,255,0.7);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .stButton > button {
        background: #667eea; color: white; border-radius: 8px; border: none;
        padding: 0.6rem 2rem; font-weight: 600; width: 100%;
    }
    .stButton > button:hover {background: #5a67d8;}
    
    .css-1d391kg {padding-top: 1rem;} /* Sidebar padding */
    
    @media (prefers-color-scheme: dark) {
        .stExpander {background: rgba(45,55,72,0.7); border-color: #4a5568;}
        .stButton > button {background: #5a67d8;}
        .stButton > button:hover {background: #667eea;}
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SIDEBAR – PROJECT INFO
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=80)
    st.title("ClinicOps")
    st.markdown("### AI-Powered Length of Stay Prediction")
    
    st.markdown("""
    **End-to-end MLOps Healthcare Platform**
    
    **Purpose:**  
    Predicts how many days a patient will stay in the hospital using clinical data — helping hospitals optimize bed capacity, staffing, and resource planning.
    
    **Model:**  
    - Random Forest + log-transform  
    - R² = 0.84 (original scale)  
    - Trained on 100k+ real hospitalizations
    
    **Tech Stack:**
    - DVC + Azure Blob (data versioning)
    - MLflow (tracking & serving)
    - FastAPI + MLflow model serving
    - Streamlit (clinical UI)
    - GitHub Actions → Azure Container Instances (zero-cost when idle)
    
    **Live in 10 minutes · Destroy in a minute · Cost = 0 ₺ when not used**
    
    Made with love by a passionate MLOps Engineer
    """)
    
    st.markdown("---")
    st.markdown("**Demo Links**")
    st.markdown("[GitHub Repo](https://github.com/AtamerErkal/ClinicOps)")
    #st.markdown("[Live Demo](http://clinicops-ui-...azurecontainer.io:8501)")

# ==============================================================================
# MAIN PAGE – INTRODUCTION
# ==============================================================================
st.title("ClinicOps: AI-Powered Length of Stay Prediction")
st.markdown("""
**Welcome!** This tool predicts hospital length of stay using real clinical data.

**How to use:**
1. Adjust lab values & vitals (slider or type manually — they sync!)
2. Select patient medical history
3. Click **Predict Length of Stay**
4. See prediction + clinical risk assessment

Normal reference ranges are shown — abnormal values are highlighted in red.
""")

# ==============================================================================
# INPUT SECTION – Numeric (slider + synced number input)
# ==============================================================================
st.markdown("### Laboratory Values & Vitals")
numeric_data = {}
cols = st.columns(3)

for idx, (feat, cfg) in enumerate(NUMERIC_FEATURES.items()):
    with cols[idx % 3]:
        with st.expander(f"{cfg['icon']} {feat.replace('_', ' ').title()}  \nNormal range: **{cfg['normal_min']}–{cfg['normal_max']} {cfg['unit']}**", expanded=True):
            st.caption(cfg['desc'])
            
            # Shared state via session_state
            if f"value_{feat}" not in st.session_state:
                st.session_state[f"value_{feat}"] = cfg['default']
            
            col1, col2 = st.columns([3, 1])
            with col1:
                slider_val = st.slider(
                    "Drag slider",
                    min_value=cfg['min'],
                    max_value=cfg['max'],
                    value=st.session_state[f"value_{feat}"],
                    step=0.1,
                    key=f"slider_{feat}",
                    on_change=lambda f=feat: st.session_state.update({f"value_{f}": st.session_state[f"slider_{f}"]})
                )
            with col2:
                num_val = st.number_input(
                    "Type",
                    min_value=cfg['min'],
                    max_value=cfg['max'],
                    value=st.session_state[f"value_{feat}"],
                    step=0.1,
                    key=f"num_{feat}",
                    label_visibility="collapsed",
                    on_change=lambda f=feat: st.session_state.update({f"value_{f}": st.session_state[f"num_{f}"]})
                )
            
            # Sync
            st.session_state[f"value_{feat}"] = num_val if num_val != st.session_state[f"value_{feat}"] else slider_val
            numeric_data[feat] = st.session_state[f"value_{feat}"]

# ==============================================================================
# INPUT SECTION – Categorical
# ==============================================================================
st.markdown("### Medical History & Demographics")
categorical_data = {}
cat_cols = st.columns(4)

for idx, (feat, cfg) in enumerate(CATEGORICAL_FEATURES.items()):
    with cat_cols[idx % 4]:
        with st.expander(f"{cfg['icon']} {feat.replace('_', ' ').title()}", expanded=True):
            st.caption(cfg['desc'])
            default_idx = cfg['options'].index(cfg['default'])
            selected = st.selectbox(
                "Select",
                options=cfg['options'],
                format_func=lambda x: cfg['labels'][cfg['options'].index(x)],
                index=default_idx,
                key=f"cat_{feat}"
            )
            categorical_data[feat] = selected

# ==============================================================================
# PREDICTION
# ==============================================================================
st.markdown("### Prediction")
if st.button("Predict Length of Stay", type="primary"):
    payload = {k: float(v) if k in NUMERIC_FEATURES else str(v) 
               for k, v in {**numeric_data, **categorical_data}.items()}

    with st.spinner("Running inference..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if resp.status_code == 200:
                result = resp.json()
                los = round(result['predicted_length_of_stay'], 2)

                st.markdown(f"### Predicted Length of Stay: **{los} days**")

                # Risk level with color
                risk = "Low" if los < 4 else "Medium" if los < 8 else "High"
                risk_color = {"Low": "#38ef7d", "Medium": "#ffd89b", "High": "#ff6b6b"}[risk]
                st.markdown(f"<h2 style='text-align:center; color:{risk_color}; margin:1rem 0;'>{risk} Risk</h2>", unsafe_allow_html=True)

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=los,
                    number={'suffix': " days", 'font': {'size': 48}},
                    gauge={
                        'axis': {'range': [0, 20], 'tickwidth': 2},
                        'bar': {'color': "#667eea"},
                        'steps': [
                            {'range': [0, 4], 'color': "#c6f6d5"},
                            {'range': [4, 8], 'color': "#fefcbf"},
                            {'range': [8, 20], 'color': "#fed7d7"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 6},
                            'thickness': 0.75,
                            'value': 10
                        }
                    },
                    title={'text': "<b>Length of Stay Prediction</b><br><span style='font-size:0.7em;color:gray'>Red line = High-risk threshold (>10 days)</span>"}
                ))
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.error("API Error")
                st.json(resp.json())
        except Exception as e:
            st.error(f"Connection failed: {e}")

# ==============================================================================
# LAB & MEDICAL HISTORY SUMMARY
# ==============================================================================
st.markdown("### Clinical Summary")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Laboratory Results**")
    lab_data = []
    for feat, cfg in NUMERIC_FEATURES.items():
        val = numeric_data[feat]
        normal = f"{cfg['normal_min']}–{cfg['normal_max']} {cfg['unit']}"
        status = "Abnormal" if val < cfg['normal_min'] or val > cfg['normal_max'] else "Normal"
        color = "🔴" if status == "Abnormal" else "🟢"
        lab_data.append({"Test": feat.replace('_', ' ').title(), "Value": f"{val} {cfg['unit']}", "Normal Range": normal, "Status": f"{color} {status}"})
    st.table(lab_data)

with col2:
    st.markdown("**Active Comorbidities & Risk Factors**")
    risk_factors = []
    for feat, cfg in CATEGORICAL_FEATURES.items():
        if feat in ['rcount', 'gender', 'facid', 'secondarydiagnosisnonicd9']:
            continue
        val = categorical_data[feat]
        if val == '1' or val == 1:
            risk_factors.append(f"• {cfg['desc']}")
    if risk_factors:
        for factor in risk_factors:
            st.error(factor)
    else:
        st.success("No active high-risk comorbidities")

# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:0.9rem;'>"
    "© 2025 ClinicOps • Enterprise MLOps Healthcare Platform • Powered by Azure, MLflow, DVC & Streamlit"
    "</p>",
    unsafe_allow_html=True
)