
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
        "systemInstruction": { "parts": [{ "text": "You are a pragmatic, business-focused Poultry Operations Consultant. Speak directly to the farm owner. Emphasize that fixing the environment (temp/humidity) restores appetite naturally. Segregate advice into 'Quick Wins' (actions for the next 4 hours) and 'Strategic Solutions' (long term). Be concise, use bullet points, and mention the financial risk." }] }
    }
    for i in range(5):
        try:
            response = requests.post(url, json=payload, timeout=12)
            if response.status_code == 200:
                return response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response.")
            time.sleep(2**i)
        except:
            time.sleep(2**i)
    return "The Advisor is temporarily offline."

# --- 1. Load Assets ---
@st.cache_resource
def load_artifacts():
    with open('random_forest_model_retrained.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('X_train_data_retrained.pkl', 'rb') as f:
        X_train_data = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, X_train_data, label_encoder

model, X_train, label_encoder = load_artifacts()

# --- 2. Live Data Processor ---
@st.cache_data
def get_dashboard_data(_label_encoder, _expected_cols):
    df = pd.read_csv('broiler_health_noisy_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'Zone_ID']).reset_index(drop=True)
    
    # Feature Engineering
    df['Water_Feed_Ratio'] = df['Avg_Water_Intake_ml'] / (df['Avg_Feed_Intake_g'] + 1e-5)
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)
    df['Target_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    
    df_dummies = pd.get_dummies(df, columns=['Zone_ID'], drop_first=True, dtype=int)
    model_df = df_dummies.reindex(columns=list(_expected_cols), fill_value=0)
    
    # Attach Metadata
    model_df['Date'] = df['Date'].values
    model_df['Zone_ID'] = df['Zone_ID'].values
    model_df['Health_Status'] = df['Health_Status'].values
    model_df['Target_Status'] = df['Target_Status'].values
    model_df['Total_Alive_Birds'] = df['Total_Alive_Birds'].values
    model_df['Bird_Age_Days'] = df['Bird_Age_Days'].values
        
    df_hist = model_df.dropna(subset=['Target_Status']).copy()
    df_live = model_df[model_df['Target_Status'].isna()].copy()
    
    return df_hist, df_live

df_hist, df_live = get_dashboard_data(label_encoder, X_train.columns)

# --- 3. Header & Narrative ---
st.title('🐔 Automated Canary: Farm Operations Center')
st.markdown("### Stop outbreaks today. Protect your profits tomorrow.")

with st.expander("📖 HOW TO USE THIS TOOL (Click to expand)", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Spot the Fire")
        st.write("Look at the **Executive Triage** board below. The AI monitors your barn sensors to find zones where birds are losing their appetite or suffering from cumulative heat stress.")
    with col2:
        st.markdown("#### 2. Test a Fix")
        st.write("Scroll down to the **Simulator**. Don't guess what to do. Use the sliders to test if turning on cooling pads or fans will successfully drop tomorrow's sickness risk.")
    with col3:
        st.markdown("#### 3. Get the Action Plan")
        st.write("Use the **AI Advisor** at the bottom to generate a step-by-step instruction list for your barn crew. Print it or message it to them immediately.")

# --- 4. Fleet Triage Area ---
latest_date = df_live['Date'].max()
st.subheader(f"🚨 Executive Triage - Risk Forecast for: { (latest_date + timedelta(days=1)).strftime('%B %d, %Y') }")
st.write("This shows you exactly where your money is at risk. A **90% Risk** means that based on today's heat and appetite, there is a 90% chance birds will start dying tomorrow.")

feature_cols = list(X_train.columns)
probs_live = model.predict_proba(df_live[feature_cols])[:, 1]

cols = st.columns(4)
zone_names = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]

zone_risk_map = {}
for i in range(len(df_live)):
    z_id = df_live.iloc[i]['Zone_ID']
    zone_risk_map[z_id] = {
        'prob': probs_live[i], 
        'birds': int(df_live.iloc[i]['Total_Alive_Birds']),
        'age': int(df_live.iloc[i]['Bird_Age_Days'])
    }

for i, zone in enumerate(zone_names):
    if zone in zone_risk_map:
        p = zone_risk_map[zone]['prob']
        birds = zone_risk_map[zone]['birds']
        age = zone_risk_map[zone]['age']
        val_at_risk = birds * 4.0 * p
        
        if p > 0.7:
            color, bg_color, label = "#ef4444", "rgba(239, 68, 68, 0.1)", "🚨 CRITICAL RISK"
        elif p > 0.4:
            color, bg_color, label = "#f59e0b", "rgba(245, 158, 11, 0.1)", "⚠️ WARNING"
        else:
            color, bg_color, label = "#22c55e", "rgba(34, 197, 94, 0.1)", "✅ STABLE"
    else:
        p, birds, val_at_risk, age, color, bg_color, label = 0.0, 0, 0, 0, "#94a3b8", "transparent", "NO DATA"
    
    with cols[i]:
        st.markdown(f"""
        <div style="padding:15px; border-radius:12px; border:3px solid {color}; text-align:center; background-color:{bg_color};">
            <h3 style="margin:0;">{zone}</h3>
            <p style="color:{color}; font-weight:bold; font-size:1.1em; margin:5px 0;">{label}</p>
            <p style="font-size:1.3em; margin:0;"><b>{p:.1%}</b> Sickness Risk</p>
            <hr style="margin:10px 0; border:0.5px solid #475569;">
            <p style="font-size:0.85em; color:#e2e8f0; margin:0;">Flock Age: <b>{age} Days</b></p>
            <p style="font-size:0.85em; color:#e2e8f0; margin:0;">Population: <b>{birds:,} birds</b></p>
            <p style="font-size:0.95em; color:#a3e635; margin-top:5px; font-weight:bold;">Money at Risk: <b>${val_at_risk:,.0f}</b></p>
        </div>
        """, unsafe_allow_html=True)

# --- 5. Interactive Simulator ---
st.divider()
st.header("🛠️ The Intervention Simulator")
st.write("Don't guess. Select a failing zone below, turn the environmental dials (simulating turning on fans or cooling pads), and see if it drops tomorrow's risk back to a safe level.")

sel_zone_name = st.selectbox("Select Zone to Manage:", zone_names)

q_row = df_live[df_live['Zone_ID'] == sel_zone_name]

if not q_row.empty:
    orig_input = q_row[feature_cols].copy()
    age = int(q_row['Bird_Age_Days'].iloc[0])
    birds_count = int(q_row['Total_Alive_Birds'].iloc[0])
    
    target_f = (age * 4.8) + 20
    target_w = target_f * 2.1
    
    col_sim_env, col_sim_bio = st.columns(2)
    
    with col_sim_env:
        st.markdown("#### 🌍 Step 1: Adjust Barn Controls")
        st.info("These are the physical levers you can pull right now. Lowering temperature usually relieves heat stress.")
        s_temp = st.slider("Target Ambient Temp (°C)", 20.0, 45.0, float(orig_input['Max_Temperature_C'].iloc[0]))
        s_hum = st.slider("Target Humidity (%)", 30.0, 100.0, float(orig_input['Avg_Humidity_Percent'].iloc[0]))
        
    with col_sim_bio:
        st.markdown("#### 🐔 Step 2: Biological Appetite")
        st.warning("You cannot force sick birds to eat. But fixing the temperature controls (Step 1) should naturally move them back toward these healthy targets.")
        st.write(f"**What they SHOULD be eating at Age {age}:** Feed: {target_f:.0f}g | Water: {target_w:.0f}ml")
        s_water = st.slider("Simulate Water Recovery (ml)", 50.0, 600.0, float(orig_input['Avg_Water_Intake_ml'].iloc[0]))
        s_feed = st.slider("Simulate Feed Recovery (g)", 20.0, 400.0, float(orig_input['Avg_Feed_Intake_g'].iloc[0]))

    # Calculate Simulation
    sim_row = orig_input.copy()
    sim_row['Max_Temperature_C'] = s_temp
    sim_row['Avg_Humidity_Percent'] = s_hum
    sim_row['Avg_Water_Intake_ml'] = s_water
    sim_row['Avg_Feed_Intake_g'] = s_feed
    sim_row['Water_Feed_Ratio'] = s_water / (s_feed + 1e-5)
    
    sim_prob = model.predict_proba(sim_row[feature_cols].astype(float))[0][1]
    orig_prob = model.predict_proba(orig_input[feature_cols].astype(float))[0][1]

    st.markdown("#### 📈 Step 3: See the Result")
    col_res1, col_res2 = st.columns([1, 1.5])
    with col_res1:
        st.metric("New Tomorrow's Risk Score", f"{sim_prob:.1%}", delta=f"{sim_prob - orig_prob:.1%}", delta_color="inverse")
        if sim_prob < 0.4: 
            st.success("✅ Good Job: These changes will stabilize the flock and protect your profits.")
        else: 
            st.error("🚨 Warning: This is not enough. You need to drop the temperature further or fix their appetite.")

    with col_res2:
        st.write("**What's driving the sickness? (Root Cause)**")
        st.caption("Look for the biggest RED bar below—that is the main problem. Red bars push the risk UP. Blue bars push the risk DOWN.")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(orig_input[feature_cols])
        if isinstance(shap_values, list): sv = shap_values[1]
        else: sv = shap_values
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.summary_plot(sv, orig_input[feature_cols], plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)

    # --- 6. Executive Advisory ---
    st.divider()
    st.header("📋 Automated Action Plans for your Crew")
    tab_advisor, tab_dice = st.tabs(["👨‍💼 Plain-English Instructions (AI)", "🎯 The 'Minimum Viable Fix' (Math)"])

    with tab_advisor:
        st.write("Click below to generate a simple, direct instruction manual you can hand to your barn workers right now.")
        if st.button("Generate Barn Instructions"):
            with st.spinner("Consulting Operations Expert..."):
                prompt = f"""
                Brief for {sel_zone_name}. Population: {birds_count} birds (Age {age}).
                Forecasted Sickness Probability: {sim_prob:.1%}.
                Current Environment: {s_temp}C, {s_hum}%.
                Intake Ratio: {orig_input['Water_Feed_Ratio'].iloc[0]:.2f} (Target ~2.1).
                
                Identify if the primary threat is Heat Stress (high W:F ratio) or Pathogen (low intake overall).
                Provide 3 highly actionable 'Quick Wins' for the barn crew (ventilation, cooling pads, water line flushing).
                """
                st.markdown(call_gemini_with_retry(prompt))

    with tab_dice:
        st.markdown("**Save Electricity: Find the Exact Minimum Cooling Needed**")
        st.write("Running tunnel ventilation and cooling pads at 100% is incredibly expensive. This algorithm calculates the *absolute minimum* drop in temperature or humidity needed to make the flock safe.")
        
        at_risk_list = [z for z, data in zone_risk_map.items() if data['prob'] > 0.4]
        if at_risk_list:
            focus = st.selectbox("Select critical zone to calculate minimum fix:", at_risk_list)
            d_query = df_live[df_live['Zone_ID'] == focus][feature_cols].astype(float)
            try:
                X_h = df_hist[feature_cols]
                y_h = label_encoder.transform(df_hist['Target_Status'])
                dice_data = dice_ml.Data(dataframe=pd.concat([X_h.astype(float), pd.Series(y_h, name='Target_Status', index=X_h.index)], axis=1), continuous_features=[c for c in X_h.columns if 'Zone_ID' not in c], categorical_features=[c for c in X_h.columns if 'Zone_ID' in c], outcome_name='Target_Status')
                exp = dice_ml.Dice(dice_data, dice_ml.Model(model=model, backend='sklearn'), method='random')
                with st.spinner("Crunching environmental math..."):
                    env_controllables = ['Max_Temperature_C', 'Avg_Humidity_Percent']
                    cf = exp.generate_counterfactuals(d_query, total_CFs=2, desired_class=0, features_to_vary=env_controllables)
                    st.dataframe(cf.cf_examples_list[0].final_cfs_df.style.highlight_max(axis=0, color="#1e3a1e"))
            except:
                st.warning("Could not calculate a safe environmental threshold. Please rely on the AI instructions.")
        else:
            st.success("All zones are currently stable. No emergency calculations needed.")
else:
    st.error("Missing Data: Please verify sensor connections.")

st.caption("Master's Thesis Project | Automated Canary System")
