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

# Page Configuration for a Premium Analytics Feel
st.set_page_config(
    page_title="Canary Early Warning Dashboard", 
    layout="wide", 
    page_icon="🐔",
    initial_sidebar_state="expanded"
)

# --- 0. Gemini API Setup ---
apiKey = ""

def call_gemini_with_retry(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {
            "parts": [{"text": "You are a Senior Poultry Veterinarian and Bio-Security Specialist. Provide structured, technical, yet actionable farm management advice. Use bullet points and bold text for clarity."}]
        }
    }
    
    for i in range(5):
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No advice available.")
            elif response.status_code == 429:
                time.sleep(2**i)
            else:
                break
        except Exception:
            time.sleep(2**i)
    return "The AI Advisor is currently busy. Please review the DiCE prescriptive summary."

# --- 1. Load Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('X_train_data_retrained.pkl', 'rb') as f:
            X_train_data = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return model, X_train_data, label_encoder
    except FileNotFoundError as e:
        st.error(f"Missing artifact file: {e.filename}. Ensure .pkl files are in the repository.")
        st.stop()

model, X_train, label_encoder = load_artifacts()

# --- 2. Data Preprocessing Logic ---
@st.cache_data
def get_processed_data(_label_encoder):
    if not os.path.exists('broiler_health_noisy_dataset.csv'):
        st.error("Dataset not found! Please upload 'broiler_health_noisy_dataset.csv'.")
        st.stop()
        
    df = pd.read_csv('broiler_health_noisy_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Zone_ID', 'Date']).reset_index(drop=True)

    # Feature Engineering (Lags and Rolling Averages)
    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Feed_Avg_3D'] = df.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Water_Avg_3D'] = df.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Mortality_Rate_Avg_3D'] = df.groupby('Zone_ID')['Daily_Mortality_Rate_Pct'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)

    # Target: Predicting the health status for the NEXT day
    df['Target_Health_Status_Label'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    df_clean = df.dropna().copy()

    # Leakage Prevention
    leakage = ['Health_Status', 'Daily_Mortality_Count', 'Daily_Mortality_Rate_Pct',
               'Panting_Percent', 'Huddling_Percent', 'Wing_Spreading_Percent', 'Droppings_Abnormality_Score']
    
    df_encoded = pd.get_dummies(df_clean, columns=['Zone_ID'], drop_first=True, dtype=int)
    X = df_encoded.drop(['Date', 'Target_Health_Status_Label'] + leakage, axis=1)
    y = _label_encoder.transform(df_clean['Target_Health_Status_Label'])

    return df_clean, X, y

df_full_clean, X_full, y_full = get_processed_data(label_encoder)

# Metrics calculation
train_size = int(len(X_full) * 0.75)
X_test = X_full.iloc[train_size:]
y_test = y_full[train_size:]
y_pred = model.predict(X_test)
metrics = {"Accuracy": accuracy_score(y_test, y_pred), "F1": f1_score(y_test, y_pred)}

# --- 3. Dashboard Fleet Status & Triage ---
latest_date = df_full_clean['Date'].max()
st.title('🐔 Automated Canary Early Warning')
st.markdown(f"**Fleet Monitoring Active** | Data Window: {latest_date.strftime('%Y-%m-%d')} (Forecasting Tomorrow)")

# Top Row: Fleet Overview
fleet_data = df_full_clean[df_full_clean['Date'] == latest_date]
X_fleet = X_full.loc[fleet_data.index]
probs_fleet = model.predict_proba(X_fleet)[:, 1]

cols = st.columns(4)
zones = sorted(fleet_data['Zone_ID'].unique())
for i, zone in enumerate(zones):
    prob = probs_fleet[i]
    with cols[i]:
        status_color = "red" if prob > 0.5 else "green"
        status_text = "⚠️ WARNING" if prob > 0.5 else "✅ STABLE"
        st.markdown(
            f"""
            <div style="padding:15px; border-radius:10px; border:2px solid {status_color}; background-color:rgba(0,0,0,0.1);">
                <h4 style="margin:0;">{zone}</h4>
                <p style="color:{status_color}; font-weight:bold; margin-bottom:5px;">{status_text}</p>
                <small>Risk Prob: {prob:.1%}</small>
            </div>
            """, 
            unsafe_allow_html=True
        )

st.write("") # Spacer

# --- 4. Sidebar: Simulation Control ---
st.sidebar.header("🛠️ Intervention Simulator")
st.sidebar.info("Select a zone and adjust environmental sliders to simulate 'What-If' recovery scenarios.")

analysis_mode = st.radio("Record Filter:", ["Current Sensor Snapshot", "Historical Anomalies"], horizontal=True)

if analysis_mode == "Current Sensor Snapshot":
    display_df = df_full_clean[df_full_clean['Date'] == latest_date].copy()
else:
    # Filter for all At-Risk historical predictions in test set
    hist_idx = X_test.index[y_pred == 1]
    display_df = df_full_clean.loc[hist_idx].copy()

if not display_df.empty:
    display_df['Label'] = display_df['Date'].dt.date.astype(str) + " | " + display_df['Zone_ID']
    selected_incident = st.selectbox("Select Record for Deep Analysis:", display_df['Label'])
    query_idx = display_df[display_df['Label'] == selected_incident].index[0]
    original_row = X_full.loc[[query_idx]].copy()
    
    # --- Simulator Sliders ---
    st.sidebar.subheader("Adjust Controllable Inputs")
    sim_temp = st.sidebar.slider("Ambient Temp (°C)", 20.0, 45.0, float(original_row['Max_Temperature_C'].iloc[0]))
    sim_hum = st.sidebar.slider("House Humidity (%)", 30.0, 100.0, float(original_row['Avg_Humidity_Percent'].iloc[0]))
    sim_water = st.sidebar.slider("Water Consumption (ml)", 50.0, 600.0, float(original_row['Avg_Water_Intake_ml'].iloc[0]))
    sim_feed = st.sidebar.slider("Feed Consumption (g)", 20.0, 400.0, float(original_row['Avg_Feed_Intake_g'].iloc[0]))

    # Create simulated instance
    sim_instance = original_row.copy()
    sim_instance['Max_Temperature_C'] = sim_temp
    sim_instance['Avg_Humidity_Percent'] = sim_hum
    sim_instance['Avg_Water_Intake_ml'] = sim_water
    sim_instance['Avg_Feed_Intake_g'] = sim_feed
    
    sim_prob = model.predict_proba(sim_instance.astype(float))[0][1]
    sim_class = "At Risk" if sim_prob > 0.5 else "Healthy"
else:
    st.warning("No records found for selection.")
    st.stop()

# --- 5. Main Analysis Cockpit ---
col_stats, col_viz = st.columns([1, 1.5])

with col_stats:
    st.subheader("Predictive Outcome")
    orig_prob = model.predict_proba(original_row.astype(float))[0][1]
    
    # Comparison Gauges
    st.metric("Original Risk (T+1)", f"{orig_prob:.1%}")
    delta = sim_prob - orig_prob
    st.metric("Simulated Risk (Intervention)", f"{sim_prob:.1%}", delta=f"{delta:.1%}", delta_color="inverse")
    
    if sim_class == "Healthy" and orig_prob > 0.5:
        st.success("🎉 Intervention Successful: Changes return flock to Stable status.")
    elif sim_class == "At Risk":
        st.error("🚨 Critical: Simulated changes insufficient to mitigate risk.")

with col_viz:
    st.subheader("7-Day Zone Context")
    # Show history for the selected zone to see trends
    zone_id = fleet_data.loc[query_idx, 'Zone_ID'] if query_idx in fleet_data.index else df_full_clean.loc[query_idx, 'Zone_ID']
    zone_history = df_full_clean[df_full_clean['Zone_ID'] == zone_id].tail(7)
    
    # Quick visual trend
    st.line_chart(zone_history.set_index('Date')[['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml']])

# Intervention Insights Tabs
tab1, tab2, tab3 = st.tabs(["💡 AI Veterinary Consultant", "🎯 Prescriptive Actions (DiCE)", "📊 Model Diagnostics"])

with tab1:
    st.subheader("Consultation Brief")
    if st.button("Generate Expert Analysis"):
        with st.spinner("Processing bio-metric data..."):
            prompt = f"""
            Zone Context: {zone_id} | Bird Age: {original_row['Bird_Age_Days'].iloc[0]} days.
            Current Simulation State: {sim_class} (Risk Probability: {sim_prob:.1%}).
            
            Simulated Parameters:
            - Max Temp: {sim_temp}°C (Original: {original_row['Max_Temperature_C'].iloc[0]})
            - Humidity: {sim_hum}%
            - Water Intake: {sim_water}ml
            - Feed Intake: {sim_feed}g
            
            1. Evaluate if these parameters are within safe biological ranges for this age.
            2. Provide immediate 'Barn Floor' actions for the farm manager.
            3. Suggest a preventative adjustment to the ventilation or feeding schedule.
            """
            advice = call_gemini_with_retry(prompt)
            st.markdown(advice)

with tab2:
    st.subheader("Automated Recovery Paths")
    st.info("The AI mathematically searches for the smallest change needed to guarantee a 'Healthy' status tomorrow.")
    
    # DiCE Engine
    controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']
    dice_df = pd.concat([X_full.astype(float), pd.Series(y_full, name='Target_Health_Status', index=X_full.index)], axis=1)
    d_data = dice_ml.Data(dataframe=dice_df, 
                          continuous_features=[c for c in X_full.columns if 'Zone_ID' not in c],
                          categorical_features=[c for c in X_full.columns if 'Zone_ID' in c],
                          outcome_name='Target_Health_Status')
    m_dice = dice_ml.Model(model=model, backend='sklearn')
    exp_dice = dice_ml.Dice(d_data, m_dice, method='random')
    
    with st.spinner("Calculating recovery paths..."):
        cf = exp_dice.generate_counterfactuals(original_row.astype(float), total_CFs=2, desired_class=0, features_to_vary=controllable)
        res_df = cf.cf_examples_list[0].final_cfs_df
        st.dataframe(res_df.style.highlight_max(axis=0, color="#1b5e20"))

with tab3:
    # Model Global Context
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Model Precision", f"{metrics['Accuracy']:.2%}")
    c_m2.metric("F1 Performance", f"{metrics['F1']:.2%}")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, 1], X_train, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption("Automated Canary System | Master's Thesis Portfolio | Prescriptive AI for Sustainable Poultry Management")
