
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

st.set_page_config(page_title="Canary Early Warning Dashboard", layout="wide", page_icon="🐔")

# --- 0. AI Executive Advisor ---
apiKey = "" 

def call_gemini_with_retry(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": { "parts": [{ "text": "You are a Senior Poultry Operations Consultant. Provide executive-level briefs. Segregate advice into 'Quick Wins' and 'Long-Term Strategic Solutions'. Focus heavily on environmental restoration to fix consumption drops." }] }
    }
    for i in range(5):
        try:
            response = requests.post(url, json=payload, timeout=12)
            if response.status_code == 200:
                return response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response.")
            time.sleep(2**i)
        except:
            time.sleep(2**i)
    return "Advisor offline. Please use the simulator."

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
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3).mean())
    df['Feed_Avg_3D'] = df.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(3).mean())
    df['Water_Avg_3D'] = df.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(3).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)
    df['Target_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    df_encoded = pd.get_dummies(df, columns=['Zone_ID'], drop_first=True, dtype=int)
    df_encoded = df_encoded.reindex(columns=['Date', 'Target_Status'] + list(_expected_cols), fill_value=0)
    return df_encoded.dropna(subset=['Target_Status']).copy(), df_encoded[df_encoded['Target_Status'].isna()].copy()

df_hist, df_live = get_dashboard_data(label_encoder, X_train.columns)

# --- 3. UI Header ---
st.title('🐔 Automated Canary: Predictive Early Warning')

with st.expander("📖 Strategic Guide: Biological Lead Indicators", expanded=True):
    st.markdown("""
    **How to read this dashboard:**
    - **Risk Score:** The probability of a health crisis tomorrow.
    - **Consumption Logic:** Healthy birds eat and drink MORE. If the AI flags a risk, it is usually because it detects a **drop** in water or feed intake.
    - **Objective:** Intervene today to restore optimal consumption tomorrow.
    """)

# --- 4. Fleet Triage ---
latest_date = df_live['Date'].max()
st.subheader(f"🌐 Fleet Forecast for Tomorrow: { (latest_date + timedelta(days=1)).strftime('%Y-%m-%d') }")

X_live = df_live.drop(['Date', 'Target_Status'], axis=1)
probs_live = model.predict_proba(X_live)[:, 1]

cols = st.columns(4)
zone_names = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
for i, zone in enumerate(zone_names):
    p = probs_live[i]
    birds = int(X_live.iloc[i]['Total_Alive_Birds'])
    color = "#ef4444" if p > 0.7 else ("#f59e0b" if p > 0.4 else "#22c55e")
    with cols[i]:
        st.markdown(f"""
        <div style="padding:15px; border-radius:12px; border:3px solid {color}; text-align:center; background-color:rgba(0,0,0,0.05);">
            <h3 style="margin:0;">{zone}</h3>
            <p style="font-size:1.5em; margin:5px 0; color:{color};"><b>{p:.1%} Risk</b></p>
            <p style="font-size:0.8em;">Population: {birds:,} birds</p>
        </div>
        """, unsafe_allow_html=True)

# --- 5. Intervention Cockpit ---
st.divider()
st.sidebar.header("🛠️ Diagnostic Cockpit")
sel_zone = st.sidebar.selectbox("Select Zone:", zone_names)

zone_map = {"Zone_B": "Zone_ID_Zone_B", "Zone_C": "Zone_ID_Zone_C", "Zone_D": "Zone_ID_Zone_D"}
if sel_zone == "Zone_A": query_row = df_live[(df_live.filter(like="Zone_ID_").sum(axis=1) == 0)]
else: query_row = df_live[df_live[zone_map[sel_zone]] == 1]

if not query_row.empty:
    orig_in = X_live.loc[[query_row.index[0]]].copy()
    age = int(orig_in['Bird_Age_Days'].iloc[0])
    
    # CALCULATE BIOLOGICAL TARGETS FOR UI GUIDANCE
    target_f = (age * 4.8) + 20
    target_w = target_f * 2.1
    
    st.sidebar.markdown(f"**Age {age} Target Range:**")
    st.sidebar.caption(f"Feed: {target_f:.0f}g | Water: {target_w:.0f}ml")
    
    s_temp = st.sidebar.slider("Ambient Temp (°C)", 20.0, 45.0, float(orig_in['Max_Temperature_C'].iloc[0]))
    s_hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, float(orig_in['Avg_Humidity_Percent'].iloc[0]))
    s_water = st.sidebar.slider("Water Intake (ml)", 50.0, 600.0, float(orig_in['Avg_Water_Intake_ml'].iloc[0]))
    s_feed = st.sidebar.slider("Feed Intake (g)", 20.0, 400.0, float(orig_in['Avg_Feed_Intake_g'].iloc[0]))

    sim_row = orig_in.copy()
    sim_row['Max_Temperature_C'], sim_row['Avg_Humidity_Percent'] = s_temp, s_hum
    sim_row['Avg_Water_Intake_ml'], sim_row['Avg_Feed_Intake_g'] = s_water, s_feed
    
    sim_p = model.predict_proba(sim_row.astype(float))[0][1]
    orig_p = model.predict_proba(orig_in.astype(float))[0][1]

    c_sim, c_shap = st.columns([1, 1.5])
    with c_sim:
        st.subheader("💡 Impact Analysis")
        st.metric("Risk Score", f"{sim_p:.1%}", delta=f"{sim_p - orig_p:.1%}", delta_color="inverse")
        st.write("**Intervention Insight:**")
        if sim_p < orig_p: st.success("This change reduces biological stress.")
        else: st.warning("This change increases predicted flock risk.")

    with c_shap:
        st.subheader(f"📊 Root Cause Diagnosis: {sel_zone}")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(orig_in)
        if isinstance(shap_values, list): sv = shap_values[1]
        else: sv = shap_values
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, orig_in, plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)

    # --- 6. AI Advisor (Moved inside the IF block to prevent NameError) ---
    st.divider()
    st.header("📋 Executive Prescriptive Advisor")
    tab_advisor, tab_dice = st.tabs(["👨‍💼 Strategic Advisor (Gemini)", "🎯 Recovery Plan (DiCE)"])

    with tab_advisor:
        if st.button("Request Executive Brief"):
            with st.spinner("Consulting AI Specialist..."):
                birds = int(orig_in['Total_Alive_Birds'].iloc[0])
                p = f"""
                Executive Brief for {sel_zone}. 
                Population: {birds} birds at Age {age}.
                Predicted Risk: {sim_p:.1%}.
                Current Readings vs Targets:
                - Temp: {s_temp}C (Target: <28C)
                - Feed: {s_feed}g (Target: {target_f:.0f}g)
                - Water: {s_water}ml (Target: {target_w:.0f}ml)
                
                Identify the consumption gap. Provide 3 'Quick Wins' to restore intake and 2 'Long-Term Solutions'.
                """
                st.markdown(call_gemini_with_retry(p))

    with tab_dice:
        st.markdown("**Path to Recovery:** Smallest sensor changes needed to return to 'Healthy' status.")
        controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']
        try:
            X_hist = df_hist.drop(['Date', 'Target_Status'], axis=1)
            dice_data = dice_ml.Data(dataframe=pd.concat([X_hist.astype(float), pd.Series(label_encoder.transform(df_hist['Target_Status']), name='Target_Status', index=X_hist.index)], axis=1), continuous_features=[c for c in X_hist.columns if 'Zone_ID' not in c], categorical_features=[c for c in X_hist.columns if 'Zone_ID' in c], outcome_name='Target_Status')
            exp = dice_ml.Dice(dice_data, dice_ml.Model(model=model, backend='sklearn'), method='random')
            cf = exp.generate_counterfactuals(orig_in.astype(float), total_CFs=2, desired_class=0, features_to_vary=controllable)
            st.dataframe(cf.cf_examples_list[0].final_cfs_df)
        except:
            st.warning("Computational Limit: Please consult the Strategic Advisor.")
else:
    st.error("No valid data found for the selected zone. Please check sensor feeds.")

st.caption("Master's Thesis Project | Automated Canary System | Powered by XAI & Gemini 1.5 Flash")
