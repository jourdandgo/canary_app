
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

# --- Page Configuration ---
st.set_page_config(page_title="Canary Early Warning Dashboard", layout="wide", page_icon="🐔")

# --- 0. Gemini API Setup ---
apiKey = "" 

def call_gemini_with_retry(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": { "parts": [{ "text": "You are a Senior Poultry Veterinarian. Provide technical, actionable advice based on farm sensors and AI risk forecasts." }] }
    }
    for i in range(5):
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "Advisor unavailable.")
            time.sleep(2**i)
        except:
            time.sleep(2**i)
    return "AI Advisor busy."

# --- 1. Load Artifacts ---
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

# --- 2. Data Preprocessing ---
@st.cache_data
def get_dashboard_data(_label_encoder, expected_cols):
    df = pd.read_csv('broiler_health_noisy_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Zone_ID', 'Date']).reset_index(drop=True)

    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3).mean())
    df['Feed_Avg_3D'] = df.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(3).mean())
    df['Water_Avg_3D'] = df.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(3).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)

    df['Target_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    
    df_encoded = pd.get_dummies(df, columns=['Zone_ID'], drop_first=True, dtype=int)
    df_encoded = df_encoded.reindex(columns=['Date', 'Target_Status'] + list(expected_cols), fill_value=0)

    df_hist = df_encoded.dropna(subset=['Target_Status']).copy()
    df_live = df_encoded[df_encoded['Target_Status'].isna()].copy()
    return df_hist, df_live

df_hist, df_live = get_dashboard_data(label_encoder, X_train.columns)

# Metrics
train_size = int(len(df_hist) * 0.75)
X_test_eval = df_hist.drop(['Date', 'Target_Status'], axis=1).iloc[train_size:]
y_test_eval = label_encoder.transform(df_hist['Target_Status'].iloc[train_size:])
y_pred_eval = model.predict(X_test_eval)
acc = accuracy_score(y_test_eval, y_pred_eval)

# --- 3. UI Header ---
st.title('🐔 Automated Canary Dashboard')
with st.expander("ℹ️ Narrative: Predicting Tomorrow"):
    st.markdown("**Problem:** Late detection. **Solution:** Predicting tomorrow's risk from today's sensors.")

# --- 4. Live Forecast ---
latest_date = df_live['Date'].max()
st.subheader(f"🌐 Forecast for Tomorrow: { (latest_date + timedelta(days=1)).strftime('%Y-%m-%d') }")

X_live = df_live.drop(['Date', 'Target_Status'], axis=1)
probs_live = model.predict_proba(X_live)[:, 1]

cols = st.columns(4)
zone_names = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]
for i, zone in enumerate(zone_names):
    p = probs_live[i] if i < len(probs_live) else 0.0
    with cols[i]:
        color = "#ef4444" if p > 0.5 else "#22c55e"
        st.markdown(f"<div style='padding:15px; border-radius:10px; border:3px solid {color}; text-align:center;'><h3>{zone}</h3><p style='color:{color}; font-weight:bold;'>{'⚠️ AT RISK' if p > 0.5 else '✅ HEALTHY'}</p><p>Risk: {p:.1%}</p></div>", unsafe_allow_html=True)

# --- 5. Simulator ---
st.sidebar.header("🛠️ Simulator")
selected_zone_name = st.sidebar.selectbox("Select Zone:", zone_names)

zone_map = {"Zone_B": "Zone_ID_Zone_B", "Zone_C": "Zone_ID_Zone_C", "Zone_D": "Zone_ID_Zone_D"}
if selected_zone_name == "Zone_A":
    query_row = df_live[(df_live.filter(like="Zone_ID_").sum(axis=1) == 0)]
else:
    query_row = df_live[df_live[zone_map[selected_zone_name]] == 1]

if not query_row.empty:
    original_input = X_live.loc[[query_row.index[0]]].copy()
    s_temp = st.sidebar.slider("Temp (°C)", 20.0, 45.0, float(original_input['Max_Temperature_C'].iloc[0]))
    s_hum = st.sidebar.slider("Hum (%)", 30.0, 100.0, float(original_input['Avg_Humidity_Percent'].iloc[0]))
    s_water = st.sidebar.slider("Water (ml)", 50.0, 600.0, float(original_input['Avg_Water_Intake_ml'].iloc[0]))
    s_feed = st.sidebar.slider("Feed (g)", 20.0, 400.0, float(original_input['Avg_Feed_Intake_g'].iloc[0]))

    sim_row = original_input.copy()
    sim_row['Max_Temperature_C'], sim_row['Avg_Humidity_Percent'] = s_temp, s_hum
    sim_row['Avg_Water_Intake_ml'], sim_row['Avg_Feed_Intake_g'] = s_water, s_feed
    sim_prob = model.predict_proba(sim_row.astype(float))[0][1]
    
    col_sim, col_shap = st.columns([1, 1.5])
    with col_sim:
        st.subheader("Impact")
        st.metric("Forecast Risk", f"{sim_prob:.1%}", delta=f"{sim_prob - (model.predict_proba(original_input.astype(float))[0][1]):.1%}", delta_color="inverse")

    with col_shap:
        st.subheader("📊 Drivers (SHAP)")
        @st.cache_data
        def get_shap_fig(_model, _X_train):
            explainer = shap.TreeExplainer(_model)
            shap_values = explainer.shap_values(_X_train)
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values[:, :, 1], _X_train, plot_type="bar", show=False)
            plt.tight_layout()
            return fig
        st.pyplot(get_shap_fig(model, X_train))

# --- 6. DiCE & Gemini ---
st.divider()
tab_dice, tab_vet = st.tabs(["🎯 DiCE", "👨‍⚕️ Gemini"])
with tab_dice:
    controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']
    X_hist_clean = df_hist.drop(['Date', 'Target_Status'], axis=1)
    y_hist_clean = label_encoder.transform(df_hist['Target_Status'])
    dice_data = dice_ml.Data(dataframe=pd.concat([X_hist_clean.astype(float), pd.Series(y_hist_clean, name='Target_Status', index=X_hist_clean.index)], axis=1), continuous_features=[c for c in X_hist_clean.columns if 'Zone_ID' not in c], categorical_features=[c for c in X_hist_clean.columns if 'Zone_ID' in c], outcome_name='Target_Status')
    exp_dice = dice_ml.Dice(dice_data, dice_ml.Model(model=model, backend='sklearn'), method='random')
    cf = exp_dice.generate_counterfactuals(original_input.astype(float), total_CFs=2, desired_class=0, features_to_vary=controllable)
    st.dataframe(cf.cf_examples_list[0].final_cfs_df)

with tab_vet:
    if st.button("Consult"):
        st.markdown(call_gemini_with_retry(f"Zone {selected_zone_name}. Risk: {sim_prob:.1%}."))

st.caption("Master's Thesis Project | Canary Health System")
