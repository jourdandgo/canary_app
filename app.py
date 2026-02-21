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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="Automated Canary: Early Warning System", 
    layout="wide", 
    page_icon="🐔"
)

# --- 0. Gemini API Setup ---
apiKey = "" # Environment injects this at runtime

def call_gemini_with_retry(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [{"text": "You are a Senior Poultry Veterinarian and Data Scientist. Provide expert, structured interventions based on sensor data. Focus on biology and barn management."}]
        }
    }
    for i in range(5):
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No advice available.")
            time.sleep(2**i)
        except:
            time.sleep(2**i)
    return "The AI Advisor is temporarily offline. Refer to the Prescriptive Actions table below."

# --- 1. Load Model Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        with open('random_forest_model_retrained.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('X_train_data_retrained.pkl', 'rb') as f:
            X_train_data = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, X_train_data, label_encoder
    except FileNotFoundError as e:
        st.error(f"Missing artifact: {e.filename}. Ensure you run the training notebook first.")
        st.stop()

model, X_train, label_encoder = load_artifacts()

# --- 2. Data Preprocessing (Predictive Logic) ---
@st.cache_data
def get_processed_data(_label_encoder):
    if not os.path.exists('broiler_health_noisy_dataset.csv'):
        st.error("Dataset not found!")
        st.stop()
        
    df = pd.read_csv('broiler_health_noisy_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Zone_ID', 'Date']).reset_index(drop=True)

    # Feature Engineering (3-Day Rolling Windows for Early Warning)
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Feed_Avg_3D'] = df.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Water_Avg_3D'] = df.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)
    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek

    # Predict Tomorrow's Status
    df['Target_Label'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    df_clean = df.dropna().copy()

    # Remove leakage
    leakage = ['Health_Status', 'Daily_Mortality_Count', 'Daily_Mortality_Rate_Pct',
               'Panting_Percent', 'Huddling_Percent', 'Wing_Spreading_Percent', 'Droppings_Abnormality_Score']
    
    df_encoded = pd.get_dummies(df_clean, columns=['Zone_ID'], drop_first=True, dtype=int)
    X = df_encoded.drop(['Date', 'Target_Label'] + leakage, axis=1)
    y = _label_encoder.transform(df_clean['Target_Label'])

    return df_clean, X, y

df_clean, X_full, y_full = get_processed_data(label_encoder)

# Calculate Accuracy & F1 for the expander
train_size = int(len(X_full) * 0.75)
X_test = X_full.iloc[train_size:]
y_test = y_full[train_size:]
y_pred_test = model.predict(X_test)
acc = accuracy_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

# --- 3. UI Header & Problem Statement ---
st.title('🐔 Automated Canary: Early Warning System')

with st.expander("ℹ️ Project Narrative & Problem Statement"):
    st.markdown("""
    **The Problem:** Respiratory diseases and heat stress in broiler farms spread exponentially. By the time a human identifies a 'sick' bird, it's often too late.
    
    **The Solution:** This AI agent acts as a 'Digital Canary.' It monitors environmental trends (Temperature/Humidity) and biological consumption (Feed/Water) to predict **Tomorrow's Health Risk today**.
    
    **Academic Rigor (XAI):** - **SHAP** identifies the global environmental drivers of risk.
    - **DiCE** provides 'Counterfactual Explanations,' telling the farmer exactly what to change to return an 'At-Risk' zone to a 'Healthy' state.
    """)

# --- 4. Fleet Triage (Active Monitoring) ---
latest_date = df_clean['Date'].max()
st.subheader(f"🌐 Fleet Health Snapshot: {latest_date.strftime('%Y-%m-%d')}")

current_data = df_clean[df_clean['Date'] == latest_date]
X_curr = X_full.loc[current_data.index]
probs = model.predict_proba(X_curr)[:, 1]

# Zone Status Cards
c1, c2, c3, c4 = st.columns(4)
zone_cards = [c1, c2, c3, c4]
zones = sorted(current_data['Zone_ID'].unique())

for i, zone in enumerate(zones):
    p = probs[i]
    with zone_cards[i]:
        label = "⚠️ WARNING" if p > 0.5 else "✅ STABLE"
        color = "red" if p > 0.5 else "green"
        st.markdown(f"""
        <div style="padding:10px; border-radius:8px; border:2px solid {color}; text-align:center;">
            <h3>{zone}</h3>
            <p style="color:{color}; font-weight:bold;">{label}</p>
            <p>Risk: {p:.1%}</p>
        </div>
        """, unsafe_allow_html=True)

# --- 5. Global Root Cause Analysis (SHAP) ---
st.divider()
st.header("📊 Global Explainability: What Drives Risk?")
st.write("SHAP (SHapley Additive exPlanations) breaks down which features the model relies on across all zones.")

col_shap_text, col_shap_plot = st.columns([1, 1.5])
with col_shap_text:
    st.markdown("""
    **How to read this plot:**
    - **Features at the top** (like `Temp_Avg_3D`) have the highest predictive power.
    - If `Temp_Avg_3D` is high, it significantly pushes the model toward an **At-Risk** prediction.
    - Notice that **Biological Factors** (Water/Feed) often lag behind **Environmental Factors**, proving that environment is the leading indicator.
    """)
with col_shap_plot:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values[:, :, 1], X_train, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig)

# --- 6. Individual Diagnostic & Intervention Planner ---
st.divider()
st.header("🛠️ Diagnostic Cockpit & Prescriptive Planner")
st.write("Select a specific record below to understand why it was flagged and test interventions.")

# Record Selection
analysis_mode = st.radio("Mode:", ["Fix Current Risks", "Analyze Historical Anomalies"], horizontal=True)
if analysis_mode == "Fix Current Risks":
    target_records = df_clean[df_clean['Date'] == latest_date].copy()
else:
    target_records = df_clean.loc[X_test.index[y_pred_test == 1]].copy()

if not target_records.empty:
    target_records['Selector'] = target_records['Date'].dt.date.astype(str) + " | " + target_records['Zone_ID']
    selected = st.selectbox("Select Record:", target_records['Selector'])
    idx = target_records[target_records['Selector'] == selected].index[0]
    query_row = X_full.loc[[idx]].copy()
    
    # --- Simulator Sliders ---
    st.subheader("🧪 Interactive What-If Simulator")
    st.info("Adjust the 'Controllable' variables to see if you can manually flip the prediction to 'Healthy'.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        s_temp = st.slider("Max Temp (°C)", 20.0, 45.0, float(query_row['Max_Temperature_C'].iloc[0]))
        s_hum = st.slider("Humidity (%)", 30.0, 100.0, float(query_row['Avg_Humidity_Percent'].iloc[0]))
    with sim_col2:
        s_water = st.slider("Water Intake (ml)", 50.0, 600.0, float(query_row['Avg_Water_Intake_ml'].iloc[0]))
        s_feed = st.slider("Feed Intake (g)", 20.0, 400.0, float(query_row['Avg_Feed_Intake_g'].iloc[0]))

    # Run Simulated Prediction
    sim_row = query_row.copy()
    sim_row['Max_Temperature_C'] = s_temp
    sim_row['Avg_Humidity_Percent'] = s_hum
    sim_row['Avg_Water_Intake_ml'] = s_water
    sim_row['Avg_Feed_Intake_g'] = s_feed
    
    sim_prob = model.predict_proba(sim_row.astype(float))[0][1]
    orig_prob = model.predict_proba(query_row.astype(float))[0][1]
    
    # Simulation Results
    sc1, sc2 = st.columns(2)
    sc1.metric("Original Risk", f"{orig_prob:.1%}")
    sc2.metric("Simulated Risk", f"{sim_prob:.1%}", delta=f"{sim_prob-orig_prob:.1%}", delta_color="inverse")

    # --- DiCE + LLM Advisory ---
    st.subheader("📋 Prescriptive Action Plan")
    tab_dice, tab_llm = st.tabs(["🎯 AI Prescriptions (DiCE)", "👨‍⚕️ Veterinary Advisor (Gemini)"])
    
    with tab_dice:
        st.write("DiCE identifies the **minimal mathematical change** needed to reach 'Healthy' status.")
        controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']
        dice_df = pd.concat([X_full.astype(float), pd.Series(y_full, name='Status', index=X_full.index)], axis=1)
        d_data = dice_ml.Data(dataframe=dice_df, continuous_features=[c for c in X_full.columns if 'Zone_ID' not in c], categorical_features=[c for c in X_full.columns if 'Zone_ID' in c], outcome_name='Status')
        m_dice = dice_ml.Model(model=model, backend='sklearn')
        exp = dice_ml.Dice(d_data, m_dice, method='random')
        
        cf = exp.generate_counterfactuals(query_row.astype(float), total_CFs=2, desired_class=0, features_to_vary=controllable)
        st.dataframe(cf.cf_examples_list[0].final_cfs_df.style.highlight_max(axis=0, color="#1e3a1e"))

    with tab_llm:
        if st.button("Generate Expert Intervention Strategy"):
            with st.spinner("Analyzing biometric signatures..."):
                prompt = f"""
                Incident Context: Zone {target_records.loc[idx, 'Zone_ID']} at Age {query_row['Bird_Age_Days'].iloc[0]} days.
                Model Flags this as 'At Risk' (Prob: {orig_prob:.1%}).
                
                Sensor Data:
                - Temperature: {s_temp}°C
                - Humidity: {s_hum}%
                - Water/Feed: {s_water}ml / {s_feed}g
                
                Provide a structured plan:
                1. Immediate biological risk (heat stress vs respiratory).
                2. Immediate barn adjustments.
                3. Long-term preventative feed/water strategy.
                """
                st.markdown(call_gemini_with_retry(prompt))

# --- Footer & Technical Metrics ---
st.divider()
with st.expander("📚 Model Technical Performance (Fixed Metric Definitions)"):
    m1, m2 = st.columns(2)
    m1.metric("Accuracy (Test Set)", f"{acc:.2%}")
    m2.metric("F1-Score (Robustness)", f"{f1:.2%}")
    st.caption("Metrics evaluated on a 25% hold-out window to prevent temporal leakage.")

st.caption("Master's Thesis Project | Automated Canary System | Powered by XAI & Gemini 1.5 Flash")
