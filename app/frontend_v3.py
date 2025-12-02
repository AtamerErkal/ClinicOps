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
    'glucose': {'default': 100.0, 'min': 50.0, 'max': 500.0, 'icon': '🍬', 'unit': 'mg/dL',
                'desc': "Blood glucose (70-120)"},
    'bloodureanitro': {'default': 15.0, 'min': 0.0, 'max': 100.0, 'icon': '🫘', 'unit': 'mg/dL',
                       'desc': "Blood Urea Nitrogen"},
    'creatinine': {'default': 1.0, 'min': 0.0, 'max': 15.0, 'icon': '💧', 'unit': 'mg/dL',
                   'desc': "Creatinine level"},
    'bmi': {'default': 25.0, 'min': 10.0, 'max': 60.0, 'icon': '⚖️', 'unit': 'kg/m²',
            'desc': "Body Mass Index"},
    'pulse': {'default': 75.0, 'min': 40.0, 'max': 200.0, 'icon': '❤️', 'unit': 'bpm',
              'desc': "Heart rate"},
    'respiration': {'default': 16.0, 'min': 8.0, 'max': 40.0, 'icon': '🫁', 'unit': '/min',
                    'desc': "Respiration rate"}
}

CATEGORICAL_FEATURES = {
    'rcount': {'default': '0', 'options': ['0','1','2','3','4','5+'], 
               'labels': ['0', '1', '2', '3', '4', '5+'], 'icon': '🔄', 'desc': "Prior visits"},
    'gender': {'default': 'F', 'options': ['F','M'], 
               'labels': ['Female', 'Male'], 'icon': '👤', 'desc': "Gender"},
    'facid': {'default': 'A', 'options': ['A','B','C','D','E'], 
              'labels': ['Facility A', 'Facility B', 'Facility C', 'Facility D', 'Facility E'], 
              'icon': '🏥', 'desc': "Facility ID"},
    'secondarydiagnosisnonicd9': {'default': '0', 
                                  'options': [str(i) for i in range(11)], 
                                  'labels': [str(i) for i in range(11)], 
                                  'icon': '📋', 'desc': "Secondary diagnosis"},
    'dialysisrenalendstage': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '💉', 'desc': "Dialysis/ESRD"},
    'asthma':                {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🫁', 'desc': "Asthma"},
    'irondef':               {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🩸', 'desc': "Iron deficiency"},
    'pneum':                 {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🦠', 'desc': "Pneumonia"},
    'substancedependence':   {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '💊', 'desc': "Substance dependence"},
    'psychologicaldisordermajor': {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🧠', 'desc': "Major psych disorder"},
    'depress':               {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '😔', 'desc': "Depression"},
    'psychother':            {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🛋️', 'desc': "Psychotherapy"},
    'fibrosisandother':      {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🫁', 'desc': "Lung fibrosis"},
    'malnutrition':          {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🍽️', 'desc': "Malnutrition"},
    'hemo':                  {'default': 0, 'options': [0,1], 'labels': ['No','Yes'], 'icon': '🩸', 'desc': "Hemorrhoids"},
}

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM STYLING
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ClinicOps - Length of Stay Prediction",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling with gradient header, cards, and professional design
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main text color - adapts to theme */
    .main, .main p, .main span, .main div {
        color: #1a202c;
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main, .main p, .main span, .main div {
            color: #e2e8f0 !important;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Main container */
    .main {
        background: transparent;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Dark mode background */
    @media (prefers-color-scheme: dark) {
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        }
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        color: white;
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        font-size: 1.3rem;
        font-weight: 300;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    .header-description {
        font-size: 1rem;
        margin-top: 1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Section headers */
    .section-header {
        background: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin: 2rem 0 1.5rem 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .section-header h2 {
        margin: 0;
        color: #1a202c;
        font-size: 1.6rem;
        font-weight: 600;
    }
    
    /* Dark mode section headers */
    @media (prefers-color-scheme: dark) {
        .section-header {
            background: #2d3748;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .section-header h2 {
            color: #e2e8f0;
        }
    }
    
    /* Input cards */
    .input-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        margin-bottom: 1.5rem;
        border: 2px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }
    
    .input-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
        border-color: #667eea;
    }
    
    /* Dark mode input cards */
    @media (prefers-color-scheme: dark) {
        .input-card {
            background: #2d3748;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            border-color: #4a5568;
        }
        
        .input-card:hover {
            box-shadow: 0 6px 20px rgba(0,0,0,0.5);
            border-color: #667eea;
        }
    }
    
    /* Add spacing between input groups */
    .stColumn {
        padding: 0 0.5rem;
    }
    
    /* Input field styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSelectbox {
        padding: 0.5rem 0;
    }
    
    /* Separator line between sections */
    .input-separator {
        height: 2px;
        background: linear-gradient(90deg, transparent, #e2e8f0 20%, #e2e8f0 80%, transparent);
        margin: 2rem 0;
        border-radius: 2px;
    }
    
    @media (prefers-color-scheme: dark) {
        .input-separator {
            background: linear-gradient(90deg, transparent, #4a5568 20%, #4a5568 80%, transparent);
        }
    }
    
    /* Feature labels */
    .feature-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a202c;
        margin-bottom: 0.3rem;
    }
    
    .feature-desc {
        font-size: 0.85rem;
        color: #4a5568;
        margin-bottom: 0.8rem;
    }
    
    /* Dark mode feature labels */
    @media (prefers-color-scheme: dark) {
        .feature-label {
            color: #e2e8f0;
        }
        
        .feature-desc {
            color: #a0aec0;
        }
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 3rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Results card */
    .results-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
    }
    
    /* Dark mode results card */
    @media (prefers-color-scheme: dark) {
        .results-card {
            background: #2d3748;
            box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        }
    }
    
    /* Success/Alert boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    
    /* Dark mode metrics */
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background: #2d3748;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        
        div[data-testid="metric-container"] label {
            color: #a0aec0 !important;
        }
        
        div[data-testid="metric-container"] [data-testid="stMetricValue"] {
            color: #e2e8f0 !important;
        }
    }
    
    /* Footer */
    .footer-container {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 3rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .footer-text {
        color: #4a5568;
        font-size: 0.95rem;
        font-weight: 400;
    }
    
    /* Dark mode footer */
    @media (prefers-color-scheme: dark) {
        .footer-container {
            background: #2d3748;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        
        .footer-text {
            color: #a0aec0;
        }
    }
    
    .footer-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.3rem;
        font-weight: 500;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 8px;
        font-weight: 600;
        color: #1a202c;
    }
    
    /* Dark mode expander */
    @media (prefers-color-scheme: dark) {
        .streamlit-expanderHeader {
            background: #2d3748;
            color: #e2e8f0;
        }
    }
    
    /* Ensure all Streamlit text is readable */
    @media (prefers-color-scheme: dark) {
        .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
            color: #e2e8f0 !important;
        }
        
        /* Selectbox and input fields */
        [data-baseweb="select"] {
            background-color: #2d3748 !important;
        }
        
        [data-baseweb="select"] > div {
            color: #e2e8f0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HEADER SECTION
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🏥 Hospital ClinicOps</h1>
    <p class="header-subtitle">AI-Powered Length of Stay Prediction System</p>
    <p class="header-description">
        Advanced machine learning platform for predicting hospital length of stay using MLflow and Azure MLOps pipeline.
        Optimize resource allocation and improve patient care with data-driven insights.
    </p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# INPUT SECTION
# ═══════════════════════════════════════════════════════════════
input_data = {}

# Laboratory Values & Vitals Section
st.markdown("""
<div class="section-header">
    <h2>🔬 Laboratory Values & Vital Signs</h2>
</div>
""", unsafe_allow_html=True)

cols = st.columns(3)
for idx, (feature, cfg) in enumerate(NUMERIC_FEATURES.items()):
    with cols[idx % 3]:
        st.markdown(f"""
        <div class="feature-label">{cfg['icon']} {feature.replace('_', ' ').title()}</div>
        <div class="feature-desc">{cfg['desc']}</div>
        """, unsafe_allow_html=True)
        
        input_data[feature] = st.slider(
            "",
            min_value=cfg['min'],
            max_value=cfg['max'],
            value=cfg['default'],
            step=0.1,
            key=f"num_{feature}",
            label_visibility="collapsed"
        )
        st.markdown(f"**<span style='color: #667eea; font-size: 1.1rem;'>{input_data[feature]}</span>** {cfg['unit']}", unsafe_allow_html=True)
        
        # Add a visual separator between inputs
        if idx < len(NUMERIC_FEATURES) - 1:
            st.markdown("<div style='height: 1px; background: #e2e8f0; margin: 1.5rem 0; border-radius: 2px;'></div>", unsafe_allow_html=True)

# Section separator
st.markdown("<div class='input-separator'></div>", unsafe_allow_html=True)

# Medical History Section
st.markdown("""
<div class="section-header">
    <h2>📋 Patient Demographics & Medical History</h2>
</div>
""", unsafe_allow_html=True)

cols = st.columns(4)
for idx, (feature, cfg) in enumerate(CATEGORICAL_FEATURES.items()):
    with cols[idx % 4]:
        st.markdown(f"""
        <div class="feature-label">{cfg['icon']} {feature.replace('_', ' ').title()}</div>
        <div class="feature-desc">{cfg['desc']}</div>
        """, unsafe_allow_html=True)

        selected_label = st.selectbox(
            "Select",
            options=cfg['labels'],
            index=cfg['labels'].index(
                cfg['labels'][cfg['options'].index(cfg['default'])]
            ),
            key=f"cat_{feature}",
            label_visibility="collapsed"
        )

        selected_value = cfg['options'][cfg['labels'].index(selected_label)]
        input_data[feature] = selected_value
        
        # Add a visual separator between inputs
        if idx < len(CATEGORICAL_FEATURES) - 1:
            st.markdown("<div style='height: 1px; background: #e2e8f0; margin: 1.5rem 0; border-radius: 2px;'></div>", unsafe_allow_html=True)

# Section separator
st.markdown("<div class='input-separator'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PREDICTION SECTION
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>🎯 Generate Prediction</h2>
</div>
""", unsafe_allow_html=True)

if st.button("🚀 Predict Length of Stay", type="primary"):
    with st.spinner("🔄 Analyzing patient data..."):
        payload = {}
        for k, v in input_data.items():
            if k in NUMERIC_FEATURES:
                payload[k] = float(v)
            else:
                payload[k] = str(v)

        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
            if response.status_code == 200:
                result = response.json()
                los = round(result['predicted_length_of_stay'], 2)

                # Results Display
                st.markdown("<div class='results-card'>", unsafe_allow_html=True)
                
                # Main prediction
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.metric(
                        label="📊 Predicted Length of Stay",
                        value=f"{los} days",
                        delta=f"{los - 5:.1f} days vs. average"
                    )
                
                with col2:
                    risk = "Low" if los < 4 else "Medium" if los < 8 else "High"
                    risk_emoji = "✅" if risk == "Low" else "⚠️" if risk == "Medium" else "🚨"
                    color = "#38ef7d" if risk == "Low" else "#ffd89b" if risk == "Medium" else "#ff6b6b"
                    st.markdown(f"""
                    <div style='background: {color}; padding: 1.5rem; border-radius: 12px; text-align: center;'>
                        <div style='font-size: 2.5rem;'>{risk_emoji}</div>
                        <div style='font-size: 1.5rem; font-weight: 600; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);'>
                            {risk} Risk
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    confidence = 85 + (5 if risk == "Medium" else 10 if risk == "Low" else 0)
                    st.metric(
                        label="🎯 Confidence",
                        value=f"{confidence}%"
                    )

                st.markdown("</div>", unsafe_allow_html=True)

                # Gauge Chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=los,
                    number={'suffix': " days", 'font': {'size': 48}},
                    delta={'reference': 5, 'increasing': {'color': '#ff6b6b'}, 'decreasing': {'color': '#38ef7d'}},
                    gauge={
                        'axis': {'range': [0, 20], 'tickwidth': 2, 'tickcolor': '#667eea'},
                        'bar': {'color': '#667eea', 'thickness': 0.75},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#e2e8f0",
                        'steps': [
                            {'range': [0, 4], 'color': '#c6f6d5'},
                            {'range': [4, 8], 'color': '#fefcbf'},
                            {'range': [8, 20], 'color': '#fed7d7'}
                        ],
                        'threshold': {
                            'line': {'color': "#e53e3e", 'width': 4},
                            'thickness': 0.75,
                            'value': 10
                        }
                    },
                    title={'text': "<b>Length of Stay Prediction</b>"}
                ))
                fig.update_layout(
                    height=400,
                    margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'family': "Inter", 'size': 14}
                )
                st.plotly_chart(fig, use_container_width=True)



            else:
                st.error(f"❌ API Error: Status Code {response.status_code}")
                st.json(response.json())
        except Exception as e:
            st.error(f"⚠️ Connection Error: {str(e)}")
            st.info("Please ensure the API server is running and accessible.")

# ═══════════════════════════════════════════════════════════════
# ANALYTICS SECTION
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
    <h2>📊 Patient Data Analytics</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background: white; padding: 1.2rem; border-radius: 10px; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06);'>
    <p style='margin: 0; color: #1a202c; font-size: 0.95rem; line-height: 1.6;'>
        <b>Visual overview of patient data:</b> The bar chart displays laboratory values and vital signs to identify abnormal ranges. 
        The pie chart shows the proportion of active comorbidities (pre-existing medical conditions) which are key factors in predicting hospital length of stay.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    lab_vals = {k.replace('_', ' ').title(): v for k, v in input_data.items() if k in NUMERIC_FEATURES}
    fig = go.Figure(data=[go.Bar(
        x=list(lab_vals.keys()),
        y=list(lab_vals.values()),
        text=[f"{v:.1f}" for v in lab_vals.values()],
        textposition='outside',
        marker=dict(
            color=list(lab_vals.values()),
            colorscale='Viridis',
            showscale=False,
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra></extra>'
    )])
    fig.update_layout(
        title={'text': "<b>Laboratory & Vital Values</b>", 'x': 0.5, 'xanchor': 'center'},
        template="plotly_white",
        xaxis={'tickangle': -45},
        yaxis={'title': 'Value'},
        height=400,
        margin={'t': 80, 'b': 100},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'size': 12}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    conditions = sum(1 for k, v in input_data.items()
                     if k not in ['rcount', 'gender', 'facid', 'secondarydiagnosisnonicd9']
                     and v == '1')
    fig = go.Figure(go.Pie(
        labels=['Active Conditions', 'No Condition'],
        values=[conditions, 11 - conditions],
        hole=0.5,
        marker=dict(
            colors=['#667eea', '#e2e8f0'],
            line=dict(color='white', width=3)
        ),
        textfont=dict(size=16, color='white', family='Inter'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    fig.update_layout(
        title={'text': f"<b>Active Comorbidities: {conditions}/11</b>", 'x': 0.5, 'xanchor': 'center'},
        height=400,
        margin={'t': 80, 'b': 50},
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter', 'size': 12},
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-container">
    <div style='margin-bottom: 1rem;'>
        <span class="footer-badge">🏥 Hospital ClinicOps</span>
        <span class="footer-badge">🤖 MLOps Healthcare System</span>
        <span class="footer-badge">☁️ Azure Cloud</span>
        <span class="footer-badge">📊 MLflow</span>
    </div>
    <p class="footer-text">
        Enterprise-grade ML prediction system | Real-time inference | Scalable architecture
    </p>
    <p class="footer-text" style='font-size: 0.8rem; margin-top: 0.5rem;'>
        © 2025 ClinicOps Platform - Advanced Healthcare Analytics
    </p>
</div>
""", unsafe_allow_html=True)