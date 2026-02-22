
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

# --- Business UI Configuration ---
st.set_page_config(page_title="Canary Early Warning Dashboard", layout="wide", page_icon="🐔")

# --- 0. AI Executive Advisor ---
apiKey = "" 

def call_gemini_with_retry(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={apiKey}"
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": { "parts": [{ "text": "You are a Senior Poultry Operations Consultant. Provide executive-level briefs for farm owners. Use professional, friendly language. Segregate advice into 'Quick Wins' (immediate actions) and 'Long-Term Solutions' (strategy). Focus on protecting bird population and appetite." }] }
    }
    for i in range(5):
        try:
            response = requests.post(url, json=payload, timeout=12)
            if response.status_code == 200:
                return response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response.")
            time.sleep(2**i)
        except:
            time.sleep(2**i)
    return "The Executive Advisor is currently offline. Please review the mathematical recovery paths."

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

# --- 2. Live Data Processor ---
@st.cache_data
def get_dashboard_data(_label_encoder, _expected_cols):
    df = pd.read_csv('broiler_health_noisy_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Date', 'Zone_ID']).reset_index(drop=True)
    
    # Feature Engineering (Must match training)
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3).mean())
    df['Feed_Avg_3D'] = df.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(3).mean())
    df['Water_Avg_3D'] = df.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(3).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)
    df['Target_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    
    # Create dummy variables for Zone_ID
    df_dummies = pd.get_dummies(df, columns=['Zone_ID'], drop_first=True, dtype=int)
    
    # FIXED: Reindex ONLY the features used in training to prevent duplicates
    model_ready_features = df_dummies.reindex(columns=list(_expected_cols), fill_value=0)
    
    # Attach non-feature metadata back for display/filtering
    # Use copy to ensure we don't accidentally modify the feature set
    final_df = model_ready_features.copy()
    final_df['Date'] = df['Date'].values
    final_df['Zone_ID'] = df['Zone_ID'].values
    final_df['Health_Status'] = df['Health_Status'].values
    final_df['Target_Status'] = df['Target_Status'].values
    
    # Split into History (for DiCE background) and Live (for today's dashboard)
    df_hist = final_df.dropna(subset=['Target_Status']).copy()
    df_live = final_df[final_df['Target_Status'].isna()].copy()
    
    return df_hist, df_live

df_hist, df_live = get_dashboard_data(label_encoder, X_train.columns)

# --- 3. Header & Strategic Narrative ---
st.title('🐔 Automated Canary: Predictive Early Warning Dashboard')
st.markdown("### Protect Your Flock. Protect Your Profits.")

with st.expander("📖 STRATEGIC GUIDE: How to use this system to protect your farm", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**The Problem: We detect sickness too late.**")
        st.write("In broiler houses, respiratory and heat issues spread overnight. By the time a bird looks sick, the damage is done. Mortality is inevitable.")
        st.markdown("**What is the Pattern Match Score?**")
        st.write("This AI compares your current intake pattern (feed, water, temperature) against HISTORICAL cases. A 99% match means your situation mirrors cases that historically led to health transitions 99% of the time. It's NOT a mortality rate—it's a biometric signature match.")
        st.markdown("**How to read it:**")
        st.markdown("""
        - **≥99% Match**: Your birds match a state that historically precedes health challenges. NOT guaranteed crisis, but WATCH intake closely.
        - **80-99% Match**: High historical correlation. Stack management interventions.
        - **<80% Match**: Pattern less aligned with at-risk cases. But still validate intake targets below.
        """)
    with col2:
        st.markdown("**The Solution: Biological Lead Indicators**")
        st.write("This AI monitors sensors for **Tomorrow's Forecast**. It identifies drops in appetite and rising heat history *today* so you can fix the environment *before* mortality occurs.")
        st.markdown("**Your Action Plan:**")
        st.markdown("""
        1. **Check the Appetite Gap** (below): Are you 95%+ of biological targets?
        2. **If YES**: You're safe. Monitor the 3-day rolling trends.
        3. **If NO**: Adjust ventilation, feed delivery, water pressure TODAY.
        4. **Run What-If** (in diagnostic zone): See how closing the gap reduces pattern match.
        """)
        st.info("💡 **Biological Truth**: Birds eating 100%+ of age-based targets almost NEVER fail. An intake of 95%+ is safe margin.")

# --- 4. Fleet Triage Area ---
latest_date = df_live['Date'].max()
st.subheader(f"🌐 Fleet Health Forecast for Tomorrow: { (latest_date + timedelta(days=1)).strftime('%B %d, %Y') }")

# Select only features used in training for the model to ensure order match
feature_cols = list(X_train.columns)
X_live = df_live[feature_cols]
probs_live = model.predict_proba(X_live)[:, 1]

cols = st.columns(4)
zone_names = ["Zone_A", "Zone_B", "Zone_C", "Zone_D"]

# Correct mapping for triage cards WITH BIOLOGICAL TARGETS
zone_risk_map = {}
for i in range(len(df_live)):
    z_id = df_live.iloc[i]['Zone_ID']
    age = int(df_live.iloc[i]['Bird_Age_Days'])
    target_feed = (age * 4.8) + 20
    target_water = target_feed * 2.1
    actual_feed = float(df_live.iloc[i]['Avg_Feed_Intake_g'])
    actual_water = float(df_live.iloc[i]['Avg_Water_Intake_ml'])
    
    # Appetite Gap: how far below target are we?
    feed_gap_pct = (actual_feed - target_feed) / target_feed if target_feed > 0 else 0
    water_gap_pct = (actual_water - target_water) / target_water if target_water > 0 else 0
    
    zone_risk_map[z_id] = {
        'prob': probs_live[i],
        'birds': int(df_live.iloc[i]['Total_Alive_Birds']),
        'age': age,
        'feed_gap': feed_gap_pct,
        'water_gap': water_gap_pct,
        'actual_feed': actual_feed,
        'target_feed': target_feed,
        'actual_water': actual_water,
        'target_water': target_water
    }

for i, zone in enumerate(zone_names):
    if zone in zone_risk_map:
        data = zone_risk_map[zone]
        p = data['prob']
        birds = data['birds']
        val_at_risk = birds * 4.0 * p
        
        # NEW LOGIC: Risk stratification based on appetite gap + model probability
        # If intake is ABOVE target, lower the effective risk; if BELOW, amplify it
        appetite_severity = max(data['feed_gap'], data['water_gap'])  # More negative = worse
        
        # Adjusted risk: model prob + appetite gap penalty
        if appetite_severity < -0.1:  # More than 10% below target
            adjusted_risk_level = "CRITICAL"
            color = "#dc2626"
        elif appetite_severity < -0.05 or p > 0.85:  # 5-10% below or high model prob
            adjusted_risk_level = "HIGH RISK"
            color = "#ea580c"
        elif appetite_severity < 0 or p > 0.60:  # Slightly below or moderate prob
            adjusted_risk_level = "MONITOR"
            color = "#f59e0b"
        else:  # Meeting or exceeding targets
            adjusted_risk_level = "STABLE"
            color = "#10b981"
        
        icon = "🚨" if adjusted_risk_level == "CRITICAL" else ("⚠️" if "RISK" in adjusted_risk_level else ("📊" if "MONITOR" in adjusted_risk_level else "✅"))
    else:
        data = {}
        p, birds, val_at_risk, color, adjusted_risk_level, icon = 0.0, 0, 0, "#94a3b8", "NO DATA", "⚠️"
    
    with cols[i]:
        feed_str = f"{data.get('actual_feed', 0):.0f}g" if data else "N/A"
        target_feed_str = f"{data.get('target_feed', 0):.0f}g" if data else "N/A"
        
        st.markdown(f"""
        <div style="padding:15px; border-radius:12px; border:3px solid {color}; text-align:center; background-color:rgba(0,0,0,0.1);">
            <h3 style="margin:0;">{zone}</h3>
            <p style="color:{color}; font-weight:bold; font-size:0.95em; margin:5px 0;">{icon} {adjusted_risk_level}</p>
            <p style="font-size:1.2em; margin:3px 0;"><b>{p:.1%}</b> Pattern Match</p>
            <hr style="margin:8px 0; border:0.5px solid #475569;">
            <p style="font-size:0.8em; color:#cbd5e1; margin:3px 0;">Feed: {feed_str} / {target_feed_str}</p>
            <p style="font-size:0.8em; color:#cbd5e1; margin:0px 0;">Pop: <b>{birds:,}</b> birds</p>
            <p style="font-size:0.85em; color:#fef3c7; margin-top:5px;"><b>${val_at_risk:,.0f}</b> at risk</p>
        </div>
        """, unsafe_allow_html=True)

# --- 4.5 Appetite Gap Deep Dive (Business-Ready Insight) ---
st.divider()
st.subheader("📉 Appetite Gap Analysis: Actual vs. Biological Targets")
st.markdown("*This is the REAL diagnostics: how far are your birds from healthy intake targets?*")

gap_cols = st.columns(4)
for i, zone in enumerate(zone_names):
    if zone in zone_risk_map:
        data = zone_risk_map[zone]
        age = data['age']
        actual_feed = data['actual_feed']
        target_feed = data['target_feed']
        actual_water = data['actual_water']
        target_water = data['target_water']
        
        feed_pct_of_target = (actual_feed / target_feed) * 100 if target_feed > 0 else 0
        water_pct_of_target = (actual_water / target_water) * 100 if target_water > 0 else 0
        
        with gap_cols[i]:
            st.markdown(f"**{zone} (Age {age} days)**")
            
            # Feed intake
            st.markdown("**🌾 Feed Intake**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Actual", f"{actual_feed:.0f}g", f"{feed_pct_of_target:.0f}%")
            with col2:
                st.metric("Target", f"{target_feed:.0f}g", f"100%")
            
            # Water intake  
            st.markdown("**💧 Water Intake**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Actual", f"{actual_water:.0f}ml", f"{water_pct_of_target:.0f}%")
            with col2:
                st.metric("Target", f"{target_water:.0f}ml", f"100%")
            
            # Severity indicator
            gap_severity = min(feed_pct_of_target, water_pct_of_target)
            if gap_severity < 90:
                st.error("⛔ CRITICAL GAP: <90% of target")
            elif gap_severity < 95:
                st.warning("⚠️ SIGNIFICANT GAP: 90-95% of target")
            elif gap_severity < 100:
                st.info("📊 MINOR GAP: 95-100% of target")
            else:
                st.success("✅ ABOVE TARGET: >100%")

# --- 5. Interactive Diagnostic Area ---
st.divider()
st.sidebar.header("🛠️ Diagnostic Simulator")
st.sidebar.info("Adjust today's sensors manually to test recovery strategies.")
sel_zone_name = st.sidebar.selectbox("Select Zone for Analysis:", zone_names)

# Filter live data for the selected zone
q_row = df_live[df_live['Zone_ID'] == sel_zone_name]

if not q_row.empty:
    # Get only the features the model expects (in the correct order)
    orig_input = q_row[feature_cols].copy()
    age = int(q_row['Bird_Age_Days'].iloc[0])
    birds_count = int(q_row['Total_Alive_Birds'].iloc[0])
    
    # Health Targets
    target_f = (age * 4.8) + 20
    target_w = target_f * 2.1
    
    st.sidebar.markdown(f"**Age {age} Health Targets:**")
    st.sidebar.write(f"Feed: {target_f:.0f}g | Water: {target_w:.0f}ml")
    
    s_temp = st.sidebar.slider("Ambient Temp (°C)", 20.0, 45.0, float(orig_input['Max_Temperature_C'].iloc[0]))
    s_hum = st.sidebar.slider("Humidity (%)", 30.0, 100.0, float(orig_input['Avg_Humidity_Percent'].iloc[0]))
    s_water = st.sidebar.slider("Water Intake (ml)", 50.0, 600.0, float(orig_input['Avg_Water_Intake_ml'].iloc[0]))
    s_feed = st.sidebar.slider("Feed Intake (g)", 20.0, 400.0, float(orig_input['Avg_Feed_Intake_g'].iloc[0]))

    # Compute Simulation
    sim_row = orig_input.copy()
    sim_row['Max_Temperature_C'], sim_row['Avg_Humidity_Percent'] = s_temp, s_hum
    sim_row['Avg_Water_Intake_ml'], sim_row['Avg_Feed_Intake_g'] = s_water, s_feed
    
    # Prediction calls (ensure order matches training)
    sim_prob = model.predict_proba(sim_row[feature_cols].astype(float))[0][1]
    orig_prob = model.predict_proba(orig_input[feature_cols].astype(float))[0][1]

    col_sim, col_shap = st.columns([1, 1.5])
    with col_sim:
        st.subheader("💡 What-If Analysis")
        st.write("**Forecast risk if you intervene now:**")
        
        # Calculate appetite gap for this zone
        feed_gap = (s_feed - target_f) / target_f
        water_gap = (s_water - target_w) / target_w
        gap_health = max(feed_gap, water_gap)
        
        st.metric("Future Pattern Match", f"{sim_prob:.1%}", delta=f"{sim_prob - orig_prob:.1%}", delta_color="inverse")
        st.progress(sim_prob)
        
        # Actionable guidance
        if gap_health < -0.05:
            st.error(f"🔴 Intake is {abs(gap_health)*100:.0f}% BELOW target. This is driving high pattern match.")
            st.markdown(f"""
            **Action**: Move intake toward targets:
            - Feed target: {target_f:.0f}g (current: {s_feed:.0f}g)
            - Water target: {target_w:.0f}ml (current: {s_water:.0f}ml)
            """)
        elif gap_health < 0:
            st.warning(f"⚠️ Intake is {abs(gap_health)*100:.0f}% below target. Still in safe margin but monitor.")
        else:
            st.success(f"✅ Intake is {gap_health*100:.0f}% ABOVE target. Low risk of appetite-driven issues.")

    with col_shap:
        st.subheader(f"📊 Root Cause Diagnosis: {sel_zone_name}")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(orig_input[feature_cols])
        if isinstance(shap_values, list): sv = shap_values[1]
        else: sv = shap_values
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, orig_input[feature_cols], plot_type="bar", show=False)
        plt.tight_layout()
        st.pyplot(fig)

    # --- 6. Executive Advisory ---
    st.divider()
    st.header("📋 Executive Prescriptive Advisor")
    tab_advisor, tab_dice = st.tabs(["👨‍💼 Strategic Advisor (Gemini)", "🎯 The Recovery Path (DiCE)"])

    with tab_advisor:
        st.markdown("**Strategic Intervention Brief**")
        if st.button("Request Executive Briefing"):
            with st.spinner("Analyzing biometric signatures..."):
                prompt = f"""
                Strategic Brief for {sel_zone_name}. 
                Population: {birds_count} birds at Age {age}.
                Forecasted Crisis Probability: {sim_prob:.1%}.
                
                Current Intake Metrics:
                - Feed: {s_feed}g (Target: {target_f:.0f}g)
                - Water: {s_water}ml (Target: {target_w:.0f}ml)
                
                1. Identify the 'Appetite Gap' vs biological targets.
                2. Summarize the Financial Risk ($4/bird) of inaction for this zone.
                3. Provide 3 'Quick Wins' (Immediate barn floor fixes).
                4. Provide 2 'Long-Term Solutions' (Infrastructure or Management).
                """
                st.markdown(call_gemini_with_retry(prompt))

    with tab_dice:
        st.markdown("**Mathematical Recovery Path:** Smallest sensor changes needed today to reset tomorrow's forecast.")
        at_risk_list = [z for z, data in zone_risk_map.items() if data['prob'] > 0.4]
        if at_risk_list:
            target_focus = st.selectbox("Select critical incident to prescribe for:", at_risk_list)
            d_query = df_live[df_live['Zone_ID'] == target_focus][feature_cols].astype(float)
            
            controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']
            try:
                X_hist_clean = df_hist[feature_cols]
                y_hist_clean = label_encoder.transform(df_hist['Target_Status'])
                dice_data = dice_ml.Data(dataframe=pd.concat([X_hist_clean.astype(float), pd.Series(y_hist_clean, name='Target_Status', index=X_hist_clean.index)], axis=1), continuous_features=[c for c in X_hist_clean.columns if 'Zone_ID' not in c], categorical_features=[c for c in X_hist_clean.columns if 'Zone_ID' in c], outcome_name='Target_Status')
                exp_dice = dice_ml.Dice(dice_data, dice_ml.Model(model=model, backend='sklearn'), method='random')
                with st.spinner("Finding optimal path to health..."):
                    cf = exp_dice.generate_counterfactuals(d_query, total_CFs=2, desired_class=0, features_to_vary=controllable)
                    st.dataframe(cf.cf_examples_list[0].final_cfs_df.style.highlight_max(axis=0, color="#1e3a1e"))
            except:
                st.warning("Computational Limit: Could not find a mathematical flip for this case. Consult the Strategic Advisor.")
        else:
            st.success("All zones are stable. Prescriptive planner in idle mode.")
else:
    st.error("Missing Data: Please verify sensor connections.")

st.caption("Master's Thesis Project | Automated Canary System | Powered by XAI & Gemini 1.5 Flash")

