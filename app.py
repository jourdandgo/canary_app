
import streamlit as st
import pandas as pd
import numpy as np
import shap
import dice_ml
import matplotlib.pyplot as plt
import pickle
import os
import requests
import time
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime, timedelta

# --- Dashboard Presentation Layer ---
st.set_page_config(page_title="Canary Early Warning Dashboard", layout="wide", page_icon="🐔")

# --- 0. Gemini API Handler ---
apiKey = "" 

def call_gemini_with_retry(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": { "parts": [{ "text": "You are a Senior Poultry Veterinarian. Provide technical, clinical advice based on sensor data. Be concise, bold the key actions." }] }
    }
    for i in range(5):
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response.")
            time.sleep(2**i)
        except:
            time.sleep(2**i)
    return "Veterinary Advisor is currently offline. Please review DiCE recommendations."

# --- 1. Load Resources ---
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('X_train_data_retrained.pkl', 'rb') as f:
        X_train_data = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, X_train_data, label_encoder

model, X_train, label_encoder = load_artifacts()

# --- 2. Data Logic ---
@st.cache_data
def get_dashboard_data(_label_encoder, _expected_cols):
    df = pd.read_csv('broiler_health_noisy_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'Zone_ID']).reset_index(drop=True)

    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3).mean())
    df['Feed_Avg_3D'] = df.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(3).mean())
    df['Water_Avg_3D'] = df.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(3).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)

    # Re-identify target status
    df['Target_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    df_encoded = pd.get_dummies(df, columns=['Zone_ID'], drop_first=True, dtype=int)
    df_encoded = df_encoded.reindex(columns=['Date', 'Target_Status'] + list(_expected_cols), fill_value=0)

    df_hist = df_encoded.dropna(subset=['Target_Status']).copy()
    df_live = df_encoded[df_encoded['Target_Status'].isna()].copy()
    return df_hist, df_live

df_hist, df_live = get_dashboard_data(label_encoder, X_train.columns)

# --- 3. UI Header & Problem Definition ---
st.title('🐔 Automated Canary Early Warning Dashboard')
st.markdown("### Predicting Tomorrow's Risks Today")

with st.expander("📖 Project Narrative & Objective (Read First)", expanded=False):
    st.markdown("""
    **The Problem:** Commercial broiler farms often detect health issues (heat stress, respiratory disease) only after mortality begins. 
    By then, the financial loss is already catastrophic.
    
    **The Solution:** This AI agent acts as a 'Digital Canary.' It monitors **Lead Indicators** (Temperature trends, Feed/Water intake) 
    to forecast a zone's risk status **24 hours in advance**.
    
    **Decision Support:** - **SHAP (Root Cause):** Tells you *why* the AI is worried globally.
    - **DiCE (Action Plan):** Tells you the *minimal change* needed today to prevent a crisis tomorrow.
    - **Simulator (Human Insight):** Allows you to test your own barn management ideas.
    """)

# --- 4. Live Triage (Forecasting Tomorrow) ---
latest_date = df_live['Date'].max()
st.subheader(f"🌐 Fleet Forecast for Tomorrow: { (latest_date + timedelta(days=1)).strftime('%Y-%m-%d') }")
st.caption(f"Based on sensor telemetry from: {latest_date.strftime('%B %d, %Y')}")

X_live = df_live.drop(['Date', 'Target_Status'], axis=1)
probs_live = model.predict_proba(X_live)[:, 1]

# Risk Level Indicator
def get_risk_label(p):
    if p > 0.7: return "🚨 IMMEDIATE ACTION", "#ef4444"
    if p > 0.4: return "⚠️ PRECAUTIONARY", "#f59e0b"
    return "✅ STABLE", "#22c55e"

cols = st.columns(4)
zone_names = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
for i, zone in enumerate(zone_names):
    p = probs_live[i] if i < len(probs_live) else 0.0
    label, color = get_risk_label(p)
    with cols[i]:
        st.markdown(f"""
        <div style="padding:15px; border-radius:10px; border:2px solid {color}; background-color:rgba(0,0,0,0.05); text-align:center;">
            <h3 style="margin:0;">{zone}</h3>
            <p style="color:{color}; font-weight:bold; font-size:1.1em; margin:10px 0;">{label}</p>
            <p style="font-size:0.9em; margin:0;">Risk Score: {p:.1%}</p>
            <p style="font-size:0.75em; color:#94a3b8;">(Likelihood of crisis within 24h)</p>
        </div>
        """, unsafe_allow_html=True)

# --- 5. Intervention Simulator & Explainability ---
st.divider()
st.sidebar.header("🛠️ Intervention Simulator")
st.sidebar.markdown("Pick a zone and adjust sensors to see how your interventions change the **Tomorrow Forecast**.")

selected_zone = st.sidebar.selectbox("Select Zone:", zone_names)

zone_map = {"Zone_B": "Zone_ID_Zone_B", "Zone_C": "Zone_ID_Zone_C", "Zone_D": "Zone_ID_Zone_D"}
if selected_zone == "Zone_A":
    query_row = df_live[(df_live.filter(like="Zone_ID_").sum(axis=1) == 0)]
else:
    query_row = df_live[df_live[zone_map[selected_zone]] == 1]

if not query_row.empty:
    orig_input = X_live.loc[[query_row.index[0]]].copy()
    s_temp = st.sidebar.slider("Ambient Temp (°C)", 20.0, 45.0, float(orig_input['Max_Temperature_C'].iloc[0]))
    s_hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, float(orig_input['Avg_Humidity_Percent'].iloc[0]))
    s_water = st.sidebar.slider("Water Intake (ml)", 50.0, 600.0, float(orig_input['Avg_Water_Intake_ml'].iloc[0]))
    s_feed = st.sidebar.slider("Feed Intake (g)", 20.0, 400.0, float(orig_input['Avg_Feed_Intake_g'].iloc[0]))

    sim_row = orig_input.copy()
    sim_row['Max_Temperature_C'], sim_row['Avg_Humidity_Percent'] = s_temp, s_hum
    sim_row['Avg_Water_Intake_ml'], sim_row['Avg_Feed_Intake_g'] = s_water, s_feed
    
    sim_prob = model.predict_proba(sim_row.astype(float))[0][1]
    orig_prob = model.predict_proba(orig_input.astype(float))[0][1]
    
    col_sim, col_shap = st.columns([1, 1.5])
    with col_sim:
        st.subheader("Intervention Impact")
        st.write("Compare current vs simulated risk probability.")
        st.metric("Forecasted Risk Score", f"{sim_prob:.1%}", delta=f"{sim_prob - orig_prob:.1%}", delta_color="inverse")
        st.progress(sim_prob)
        st.caption("Lowering the score reduces the likelihood of mortality tomorrow.")

    with col_shap:
        st.subheader("📊 Global Risk Drivers (Root Cause)")
        @st.cache_data
        def get_shap_plot(_model, _X_train):
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer.shap_values(_X_train)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values[:, :, 1], _X_train, plot_type="bar", show=False)
            plt.tight_layout()
            return fig
        st.pyplot(get_shap_plot(model, X_train))

# --- 6. Prescriptive Intelligence ---
st.divider()
st.header("📋 Prescriptive Recommendations")
tab_dice, tab_vet = st.tabs(["🎯 AI Prescriptions (DiCE)", "👨‍⚕️ Clinical Advisor (Gemini)"])

with tab_dice:
    st.markdown("**Automated Action Plan:** Smallest changes needed today for a healthy tomorrow.")
    controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']
    X_hist_clean = df_hist.drop(['Date', 'Target_Status'], axis=1)
    y_hist_clean = label_encoder.transform(df_hist['Target_Status'])
    
    # BROADENING THE SEARCH to fix UserConfigValidationException
    try:
        dice_data = dice_ml.Data(
            dataframe=pd.concat([X_hist_clean.astype(float), pd.Series(y_hist_clean, name='Target_Status', index=X_hist_clean.index)], axis=1), 
            continuous_features=[c for c in X_hist_clean.columns if 'Zone_ID' not in c], 
            categorical_features=[c for c in X_hist_clean.columns if 'Zone_ID' in c], 
            outcome_name='Target_Status'
        )
        exp_dice = dice_ml.Dice(dice_data, dice_ml.Model(model=model, backend='sklearn'), method='random')
        with st.spinner("Calculating optimal recovery path..."):
            cf = exp_dice.generate_counterfactuals(orig_input.astype(float), total_CFs=2, desired_class=0, features_to_vary=controllable)
            st.dataframe(cf.cf_examples_list[0].final_cfs_df.style.highlight_max(axis=0, color="#1e3a1e"))
    except:
        st.warning("⚠️ DiCE could not find a mathematical counterfactual for this specific case. This happens when the risk is driven by non-controllable factors like bird age. Consult the Gemini Advisor below.")

with tab_vet:
    if st.button("Consult AI Veterinary Advisor"):
        with st.spinner("Analyzing biometric signatures..."):
            p = f"Zone {selected_zone}, Age {orig_input['Bird_Age_Days'].iloc[0]}. Risk: {sim_prob:.1%}. Sensors: {s_temp}C, {s_hum}% Hum, {s_water}ml Water, {s_feed}g Feed. Provide 3 immediate actions for the manager."
            st.markdown(call_gemini_with_retry(p))

st.caption("Master's Project | Automated Canary System | Powered by XAI & Gemini 1.5 Flash")
