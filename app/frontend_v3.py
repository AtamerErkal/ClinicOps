import streamlit as st
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.graph_objects as go
import os

# Mobil & ACI uyumlu port
if "AZURE_CONTAINER" in os.environ or os.getenv("STREAMLIT_SERVER_PORT"):
    os.environ["STREAMLIT_SERVER_PORT"] = "7860"

# Streamlit config – Mobil için centered layout
st.set_page_config(
    page_title="ClinicOps – AI-Powered Hospital Length of Stay Prediction",
    page_icon="🏥",
    layout="centered",  # wide → centered (mobilde overflow yok)
    initial_sidebar_state="auto",
    menu_items=None  # Menü gizle (temiz UI)
)

# API URL
API_URL = os.getenv("CLINICOPS_API_URL", "http://localhost:8000")

# Mobil CSS – Büyük slider, buton, responsive
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    .block-container {padding-top: 1rem; max-width: 100vw;}
    
    /* Mobil slider büyüt */
    [data-testid="stSlider"] {
        padding: 1rem 0;
    }
    [data-testid="stSlider"] .thumb {
        width: 40px !important; height: 40px !important; 
        background: #667eea !important;
    }
    [data-testid="stSlider"] input[type=range] {
        height: 8px !important;
    }
    
    /* Büyük predict butonu */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white; border-radius: 12px; border: none;
        padding: 1rem 2rem; font-weight: 600; font-size: 1.2rem;
        height: 4rem; width: 100%; margin: 1rem 0;
    }
    .stButton > button:hover {background: linear-gradient(90deg, #5a67d8 0%, #6b46c1 100%);}
    
    /* Expander mobil */
    .stExpander {border-radius: 12px; margin-bottom: 1rem;}
    
    /* Dark mode uyumlu */
    @media (prefers-color-scheme: dark) {
        .stExpander {background: rgba(45,55,72,0.8); border-color: #4a5568;}
    }
    
    /* Header padding azalt */
    .css-1d391kg {padding-top: 0.5rem;}
</style>
""", unsafe_allow_html=True)

# Feature definitions (mevcut – kısaltılmış)
NUMERIC_FEATURES = {
    'hematocrit': {'default': 35.0, 'min': 0.0, 'max': 60.0, 'icon': '🩸', 'unit': '%', 'normal_min': 30.0, 'normal_max': 50.0, 'desc': "Red blood cell volume"},
    'neutrophils': {'default': 50.0, 'min': 0.0, 'max': 100.0, 'icon': '⚪', 'unit': '%', 'normal_min': 40.0, 'normal_max': 75.0, 'desc': "Neutrophils percentage"},
    'sodium': {'default': 138.0, 'min': 100.0, 'max': 180.0, 'icon': '🧂', 'unit': 'mEq/L', 'normal_min': 135.0, 'normal_max': 145.0, 'desc': "Sodium level"},
    'glucose': {'default': 100.0, 'min': 50.0, 'max': 500.0, 'icon': '🍬', 'unit': 'mg/dL', 'normal_min': 70.0, 'normal_max': 120.0, 'desc': "Blood glucose"},
    'bloodureanitro': {'default': 15.0, 'min': 0.0, 'max': 100.0, 'icon': '🫘', 'unit': 'mg/dL', 'normal_min': 7.0, 'normal_max': 20.0, 'desc': "Blood Urea Nitrogen (BUN)"},
    'creatinine': {'default': 1.0, 'min': 0.0, 'max': 15.0, 'icon': '💧', 'unit': 'mg/dL', 'normal_min': 0.6, 'normal_max': 1.2, 'desc': "Creatinine level"},
    'bmi': {'default': 25.0, 'min': 10.0, 'max': 60.0, 'icon': '⚖️', 'unit': 'kg/m²', 'normal_min': 18.5, 'normal_max': 24.9, 'desc': "Body Mass Index"},
    'pulse': {'default': 75.0, 'min': 40.0, 'max': 200.0, 'icon': '💓', 'unit': 'bpm', 'normal_min': 60.0, 'normal_max': 100.0, 'desc': "Heart rate"},
    'respiration': {'default': 16.0, 'min': 8.0, 'max': 40.0, 'icon': '🫁', 'unit': '/min', 'normal_min': 12.0, 'normal_max': 20.0, 'desc': "Respiratory rate"}
}

CATEGORICAL_FEATURES = {
    'rcount': {'default': '0', 'options': ['0','1','2','3','4','5+'], 'labels': ['0', '1', '2', '3', '4', '5+'], 'icon': '🔄', 'desc': "Prior hospital visits"},
    'gender': {'default': 'F', 'options': ['F','M'], 'labels': ['Female', 'Male'], 'icon': '👤', 'desc': "Gender"},
    'facid': {'default': 'A', 'options': ['A','B','C','D','E'], 'labels': ['Facility A','B','C','D','E'], 'icon': '🏥', 'desc': "Facility ID"},
    'secondarydiagnosisnonicd9': {'default': '0', 'options': [str(i) for i in range(11)], 'labels': [str(i) for i in range(11)], 'icon': '📋', 'desc': "Number of secondary diagnoses"},
    'dialysisrenalendstage': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '💉', 'desc': "Dialysis / End-stage renal disease"},
    'asthma': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🫁', 'desc': "Asthma"},
    'irondef': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🩸', 'desc': "Iron deficiency anemia"},
    'pneum': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🦠', 'desc': "Pneumonia"},
    'substancedependence': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '💊', 'desc': "Substance dependence"},
    'psychologicaldisordermajor': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🧠', 'desc': "Major psychological disorder"},
    'depress': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '😔', 'desc': "Depression"},
    'psychother': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🛋️', 'desc': "Psychotherapy"},
    'fibrosisandother': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🫁', 'desc': "Lung fibrosis"},
    'malnutrition': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🍽️', 'desc': "Malnutrition"},
    'hemo': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🩸', 'desc': "Hemorrhoids"}
}

# Sidebar (mevcut)
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/hospital.png", width=80)
    st.title("ClinicOps")
    st.markdown("### AI-Powered Length of Stay Prediction")
    st.markdown("**End-to-end MLOps Healthcare Platform**")
    st.markdown("- Random Forest (R²=0.918)")
    st.markdown("- Azure ACI Deployed")
    st.markdown("- Mobile Ready")

# Numeric Inputs (mevcut, ama session_state sync ile)
st.markdown("### Vital Signs & Labs")
numeric_data = {}
for feat, cfg in NUMERIC_FEATURES.items():
    with st.expander(f"{cfg['icon']} {feat.replace('_', ' ').title()}", expanded=False):
        st.caption(cfg['desc'])
        if f"value_{feat}" not in st.session_state:
            st.session_state[f"value_{feat}"] = cfg['default']
        
        col1, col2 = st.columns([3, 1])
        with col1:
            slider_val = st.slider(
                label=" ",  # Boş label (mobil temiz)
                min_value=cfg['min'], max_value=cfg['max'],
                value=st.session_state[f"value_{feat}"], step=0.1,
                key=f"slider_{feat}"
            )
        with col2:
            num_val = st.number_input(
                label=" ", min_value=cfg['min'], max_value=cfg['max'],
                value=st.session_state[f"value_{feat}"], step=0.1,
                key=f"num_{feat}", label_visibility="collapsed"
            )
        
        st.session_state[f"value_{feat}"] = num_val if abs(num_val - st.session_state[f"value_{feat}"]) > 0.01 else slider_val
        numeric_data[feat] = st.session_state[f"value_{feat}"]

# Categorical (mevcut)
st.markdown("### Medical History & Demographics")
categorical_data = {}
cat_cols = st.columns(len(CATEGORICAL_FEATURES) // 3 + 1) if len(CATEGORICAL_FEATURES) > 3 else st.columns(4)
idx = 0
for feat, cfg in CATEGORICAL_FEATURES.items():
    with cat_cols[idx % len(cat_cols)]:
        with st.expander(f"{cfg['icon']} {feat.replace('_', ' ').title()}", expanded=False):
            st.caption(cfg['desc'])
            default_idx = cfg['options'].index(str(cfg['default']))
            selected = st.selectbox(
                label=" ", options=cfg['options'],
                format_func=lambda x: cfg['labels'][cfg['options'].index(str(x))],
                index=default_idx, key=f"cat_{feat}"
            )
            categorical_data[feat] = str(selected)
    idx += 1

# Prediction – Retry + Uzun Timeout (MOBİL İÇİN KRİTİK)
st.markdown("### Predict Length of Stay")
if st.button("🚀 Predict Now", type="primary", use_container_width=True):
    payload = {**numeric_data, **categorical_data}
    
    # Retry session (mobil yavaş bağlantı için)
    session = requests.Session()
    retry_strategy = Retry(total=5, backoff_factor=2, status_forcelist=[502, 503, 504, 104])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    with st.spinner(f"Analyzing with AI... (connecting to {API_URL.split('//')[1].split(':')[0]})"):
        try:
            resp = session.post(f"{API_URL}/predict", json=payload, timeout=45)  # 15 → 45sn
            if resp.status_code == 200:
                result = resp.json()
                los = round(result['predicted_length_of_stay'], 2)
                st.success(f"**Predicted: {los} days**")
                
                # Risk gauge (mevcut)
                risk = "Low" if los < 4 else "Medium" if los < 8 else "High"
                risk_color = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}[risk]
                st.markdown(f"<h3 style='text-align:center; color:{risk_color};'>Risk: {risk}</h3>", unsafe_allow_html=True)
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=los, delta={'reference': 5},
                    number={'suffix': " days", 'font': {'size': 42}},
                    gauge={'axis': {'range': [0, 20]}, 'bar': {'color': "#667eea"},
                           'steps': [{'range': [0,4],'color':'lightgreen'}, {'range':[4,8],'color':'yellow'}, {'range':[8,20],'color':'red'}],
                           'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 10}},
                    title={'text': "<b>Hospital Stay Prediction</b>"}  # Basit title
                ))
                fig.update_layout(height=400, margin=dict(l=20,r=20,t=50,b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Debug info (opsiyonel, mobil gizle)
                if st.checkbox("Show Debug"):
                    st.json(result.get('debug', {}))
                    
            else:
                st.error(f"API Error {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.ConnectionError as e:
            st.error(f"Connection failed (mobil ağı yavaş olabilir): {str(e)[:100]}")
            st.info("💡 Tip: WiFi'ye bağlan veya sayfayı yenile.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# Summary (mevcut – kısalt)
st.markdown("### Clinical Summary")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Labs")
    lab_data = [{"Test": k.replace('_',' ').title(), "Value": f"{v} {NUMERIC_FEATURES[k]['unit']}", 
                 "Status": "🟢 Normal" if NUMERIC_FEATURES[k]['normal_min'] <= v <= NUMERIC_FEATURES[k]['normal_max'] else "🔴 Abnormal"} 
                for k,v in numeric_data.items()]
    st.table(lab_data)
with col2:
    st.subheader("Risk Factors")
    risks = [k for k,v in categorical_data.items() if v == '1' and k in CATEGORICAL_FEATURES]
    if risks:
        for r in risks: st.warning(CATEGORICAL_FEATURES[r]['desc'])
    else: st.success("No high risks")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray; font-size:0.8rem;'>© 2025 ClinicOps • Mobile-Ready MLOps • Powered by Azure & Streamlit</p>", unsafe_allow_html=True)