
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

# --- Premium Presentation Config ---
st.set_page_config(page_title="Canary Early Warning Dashboard", layout="wide", page_icon="🐔")

# --- 0. AI Executive Advisor ---
apiKey = "" 

def call_gemini_with_retry(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": { "parts": [{ "text": "You are a Senior Poultry Operations Consultant. Provide executive-level briefs. Segregate advice into 'Quick Wins' (immediate actions) and 'Long-Term Solutions' (management strategy). Use bold text and bullet points." }] }
    }
    for i in range(5):
        try:
            response = requests.post(url, json=payload, timeout=12)
            if response.status_code == 200:
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response.")
            time.sleep(2**i)
        except:
            time.sleep(2**i)
    return "The Executive Advisor is currently unavailable. Review DiCE results for mathematical guidance."

# --- 1. Load Model Assets ---
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

# --- 2. Live Telemetry Logic ---
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

    df['Target_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    df_encoded = pd.get_dummies(df, columns=['Zone_ID'], drop_first=True, dtype=int)
    df_encoded = df_encoded.reindex(columns=['Date', 'Target_Status'] + list(_expected_cols), fill_value=0)

    df_hist = df_encoded.dropna(subset=['Target_Status']).copy()
    df_live = df_encoded[df_encoded['Target_Status'].isna()].copy()
    return df_hist, df_live

df_hist, df_live = get_dashboard_data(label_encoder, X_train.columns)

# --- 3. UI Narrative & Problem Statement ---
st.title('🐔 Automated Canary: Predictive Poultry Monitoring')
st.markdown("### Decision Support System for Broiler Health Management")

with st.expander("📖 Strategic Overview: Why use this tool?"):
    st.markdown("""
    **The Problem:** Commercial broiler farms often detect health issues (heat stress, disease) only after mortality spikes.
    **The Solution:** This AI monitors 'Lead Indicators' to forecast a zone's risk status **24 hours in advance**, protecting your flock population.
    
    **Understanding the Risk Score:**
    - **Risk 0-40% (Stable):** Normal operation. Population is thriving.
    - **Risk 41-70% (Precautionary):** Potential stress detected. Inspect ventilation/water systems.
    - **Risk 71-100% (Critical):** High probability of health crisis within 24 hours. Immediate barn-floor intervention required.
    """)

# --- 4. Fleet Health Forecast (Active Monitoring) ---
latest_date = df_live['Date'].max()
st.subheader(f"🌐 Fleet Health Forecast for Tomorrow: { (latest_date + timedelta(days=1)).strftime('%B %d, %Y') }")

X_live = df_live.drop(['Date', 'Target_Status'], axis=1)
probs_live = model.predict_proba(X_live)[:, 1]

cols = st.columns(4)
zone_names = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
for i, zone in enumerate(zone_names):
    p = probs_live[i] if i < len(probs_live) else 0.0
    birds = int(X_live.iloc[i]['Total_Alive_Birds'])
    color = "#ef4444" if p > 0.7 else ("#f59e0b" if p > 0.4 else "#22c55e")
    
    with cols[i]:
        st.markdown(f"""
        <div style="padding:15px; border-radius:12px; border:3px solid {color}; text-align:center; background-color:rgba(0,0,0,0.05);">
            <h3 style="margin:0;">{zone}</h3>
            <p style="font-size:1.4em; margin:10px 0; color:{color};"><b>{p:.1%} Risk</b></p>
            <hr style="margin:10px 0; border:0.5px solid #475569;">
            <p style="font-size:0.9em; margin:0;">Target Population: <b>{birds:,} birds</b></p>
        </div>
        """, unsafe_allow_html=True)

# --- 5. Intervention Simulator & Diagnostic Root Cause ---
st.divider()
st.sidebar.header("🛠️ Diagnostic Cockpit")
st.sidebar.markdown("Adjust sensors manually to simulate recovery.")

sel_zone_name = st.sidebar.selectbox("Select Zone for Deep Analysis:", zone_names)

zone_map = {"Zone_B": "Zone_ID_Zone_B", "Zone_C": "Zone_ID_Zone_C", "Zone_D": "Zone_ID_Zone_D"}
if sel_zone_name == "Zone_A":
    query_row = df_live[(df_live.filter(like="Zone_ID_").sum(axis=1) == 0)]
else:
    query_row = df_live[df_live[zone_map[sel_zone_name]] == 1]

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
        st.subheader("💡 Simulator Impact")
        st.metric("Forecasted Risk", f"{sim_prob:.1%}", delta=f"{sim_prob - orig_prob:.1%}", delta_color="inverse")
        st.progress(sim_prob)
        st.write(f"Manually adjusting controllable factors helps identify the most effective intervention.")

    with col_shap:
        st.subheader(f"📊 Root Cause for {sel_zone_name}")
        # Waterfall SHAP for individual prediction
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(orig_input)
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values[1], orig_input, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)

# --- 6. Prescriptive Intelligence & AI Advisor ---
st.divider()
st.header("📋 Executive Prescriptive Advisor")
tab_advisor, tab_dice = st.tabs(["👨‍💼 Strategic Advisor (Gemini)", "🎯 Mathematical Action Plan (DiCE)"])

with tab_advisor:
    st.markdown("### Strategic Intervention Brief")
    if st.button("Request Executive Consultation"):
        with st.spinner("Generating strategic brief..."):
            birds = int(orig_input['Total_Alive_Birds'].iloc[0])
            prompt = f"""
            Executive Brief for {sel_zone_name}. 
            Population: {birds} birds. Current Risk Score: {sim_prob:.1%}.
            Current Metrics: Temp {s_temp}C, Hum {s_hum}%, Water {s_water}ml, Feed {s_feed}g.
            
            1. Summarize the Economic Risk of inaction.
            2. Provide 3 'Quick Wins' for immediate intervention on the barn floor.
            3. Provide 2 'Long-Term Solutions' to prevent this sensor pattern from re-occurring.
            """
            st.markdown(call_gemini_with_retry(prompt))

with tab_dice:
    st.markdown("**Automated Prescription:** Smallest changes today to ensure health tomorrow.")
    # Selector for high-risk zones
    high_risk_zones = [zone_names[i] for i, p in enumerate(probs_live) if p > 0.4]
    if high_risk_zones:
        focus = st.selectbox("Select Zone to solve mathematically:", high_risk_zones)
        if focus == "Zone_A":
            d_query = df_live[(df_live.filter(like="Zone_ID_").sum(axis=1) == 0)]
        else:
            d_query = df_live[df_live[zone_map[focus]] == 1]
        
        controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']
        try:
            X_hist_clean = df_hist.drop(['Date', 'Target_Status'], axis=1)
            dice_data = dice_ml.Data(dataframe=pd.concat([X_hist_clean.astype(float), pd.Series(label_encoder.transform(df_hist['Target_Status']), name='Target_Status', index=X_hist_clean.index)], axis=1), continuous_features=[c for c in X_hist_clean.columns if 'Zone_ID' not in c], categorical_features=[c for c in X_hist_clean.columns if 'Zone_ID' in c], outcome_name='Target_Status')
            exp_dice = dice_ml.Dice(dice_data, dice_ml.Model(model=model, backend='sklearn'), method='random')
            with st.spinner("Finding optimal path..."):
                cf = exp_dice.generate_counterfactuals(d_query.drop(['Date', 'Target_Status'], axis=1).astype(float), total_CFs=2, desired_class=0, features_to_vary=controllable)
                st.dataframe(cf.cf_examples_list[0].final_cfs_df)
        except:
            st.warning("Computational Limit: Could not find a mathematical flip. Consult the Strategic Advisor above.")
    else:
        st.success("All systems stable. No prescriptions required.")

st.caption("Master's Thesis Project | Canary Early Warning System | Predictive & Prescriptive AI")
