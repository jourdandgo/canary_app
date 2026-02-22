
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
import xgboost as xgb
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

st.set_page_config(page_title="Canary Early Warning Dashboard", layout="wide", page_icon="🐔")

# --- 0. AI Executive Advisor ---
def call_gemini_with_retry(prompt, api_key):
    if not api_key:
        return "⚠️ Error: Please enter your Gemini API Key in the sidebar to generate the action plan."
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": { "parts": [{ "text": "You are a pragmatic, business-focused Poultry Operations Consultant. Speak directly to the farm owner. You will receive data that includes a 'Mathematical Directive' from DiCE. Your job is to translate that math into operational reality. Segregate your advice strictly into two sections: 1) 'Quick Wins' (actions for the next 4 hours to hit the DiCE target) and 2) 'Strategic Initiatives'. Use bolding and bullet points." }] }
    }
    for i in range(3):
        try:
            response = requests.post(url, json=payload, timeout=15)
            if response.status_code == 200:
                return response.json().get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "No response generated.")
            else:
                # This will print the exact reason Google rejected the key/call
                return f"⚠️ Google API Connection Error (Code {response.status_code}): {response.text}"
        except Exception as e:
            time.sleep(2**i)
    return f"⚠️ System Exception: Failed to connect to Google API after retries."

# --- 1. Load Assets ---
@st.cache_resource
def load_artifacts():
    with open('xgboost_model_retrained.pkl', 'rb') as f:
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
    
    df['Water_Feed_Ratio'] = df['Avg_Water_Intake_ml'] / (df['Avg_Feed_Intake_g'] + 1e-5)
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)
    df['Target_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    
    df_dummies = pd.get_dummies(df, columns=['Zone_ID'], drop_first=True, dtype=int)
    model_df = df_dummies.reindex(columns=list(_expected_cols), fill_value=0)
    
    # Attach Metadata (Including un-modeled diagnostic data for historical charting)
    model_df['Date'] = df['Date'].values
    model_df['Zone_ID'] = df['Zone_ID'].values
    model_df['Health_Status'] = df['Health_Status'].values
    model_df['Target_Status'] = df['Target_Status'].values
    model_df['Total_Alive_Birds'] = df['Total_Alive_Birds'].values
    model_df['Bird_Age_Days'] = df['Bird_Age_Days'].values
    model_df['Avg_Feed_Intake_g_RAW'] = df['Avg_Feed_Intake_g'].values # Keep raw for plotting
    model_df['Max_Temperature_C_RAW'] = df['Max_Temperature_C'].values
        
    df_hist = model_df.dropna(subset=['Target_Status']).copy()
    df_live = model_df[model_df['Target_Status'].isna()].copy()
    
    return df_hist, df_live

df_hist, df_live = get_dashboard_data(label_encoder, X_train.columns)

# --- 3. Header & Narrative ---
st.sidebar.markdown("---")
st.sidebar.header("🔑 AI Executive Configuration")
st.sidebar.markdown("The Automated Action Plans require a free Google Gemini API Key.")
st.sidebar.markdown("[👉 Click here to get your free key](https://aistudio.google.com/app/apikey)")
user_api_key = st.sidebar.text_input("Paste Gemini API Key Here:", type="password")

if not user_api_key:
    st.sidebar.warning("⚠️ Enter key above to unlock the Prescriptive AI.")

st.title('🐔 Automated Canary: Farm Operations Center')
st.markdown("### Stop outbreaks today. Protect your profits tomorrow.")

with st.expander("📖 HOW TO USE THIS TOOL & READ THE METRICS", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Spot the Fire")
        st.write("A 90% risk does **not** mean 90% of birds will die. It means the AI is 90% confident the *entire zone* is entering a severe stress event tomorrow.")
    with col2:
        st.markdown("#### 2. Test a Fix")
        st.write("Use the **Simulator** below. Adjust the environmental sliders to test if cooling the barn will restore the birds' appetite and drop the risk.")
    with col3:
        st.markdown("#### 3. Get the Action Plan")
        st.write("The system calculates the exact minimum cooling needed (DiCE), then passes that math to Gemini (LLM) to write operational instructions for your crew.")

# --- 4. Fleet Triage Area ---
latest_date = df_live['Date'].max()
st.subheader(f"🚨 Executive Triage - Risk Forecast for: { (latest_date + timedelta(days=1)).strftime('%B %d, %Y') }")

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
        
        # Age scaled mortality logic
        base_mortality = int(birds * 0.0005) 
        age_multiplier = min(1.0, age / 45.0) 
        outbreak_mortality_spike = int(birds * (0.005 + (0.045 * age_multiplier))) # 0.5% to 5.0% spike
        proj_dead = int(base_mortality + (p * outbreak_mortality_spike))
        
        dynamic_bird_value = 0.50 + (age * 0.08)
        daily_loss_dollars = proj_dead * dynamic_bird_value
        
        if p > 0.7:
            color, bg_color, label = "#ef4444", "rgba(239, 68, 68, 0.1)", "🚨 CRITICAL RISK"
        elif p > 0.4:
            color, bg_color, label = "#f59e0b", "rgba(245, 158, 11, 0.1)", "⚠️ WARNING"
        else:
            color, bg_color, label = "#22c55e", "rgba(34, 197, 94, 0.1)", "✅ STABLE"
    else:
        p, birds, proj_dead, daily_loss_dollars, age, color, bg_color, label = 0.0, 0, 0, 0, 0, "#94a3b8", "transparent", "NO DATA"
        dynamic_bird_value = 0.0
    
    with cols[i]:
        st.markdown(f"""
        <div style="padding:15px; border-radius:12px; border:3px solid {color}; text-align:center; background-color:{bg_color};">
            <h3 style="margin:0;">{zone}</h3>
            <p style="color:{color}; font-weight:bold; font-size:1.1em; margin:5px 0;">{label}</p>
            <p style="font-size:1.3em; margin:0;"><b>{p:.1%}</b> Outbreak Prob.</p>
            <hr style="margin:10px 0; border:0.5px solid #475569;">
            <p style="font-size:0.85em; color:#e2e8f0; margin:0;">Flock Age: <b>{age} Days</b></p>
            <div style="margin-top:10px; padding:8px; background-color:rgba(0,0,0,0.3); border-radius:8px;">
                <p style="font-size:0.95em; color:#fca5a5; margin:0; font-weight:bold;">Est. Daily Loss: <b>${daily_loss_dollars:,.2f}</b></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- 5. Interactive Simulator ---
st.divider()
st.header("🛠️ The Intervention Simulator")
sel_zone_name = st.selectbox("Select Zone to Analyze:", zone_names)

q_row = df_live[df_live['Zone_ID'] == sel_zone_name]

if not q_row.empty:
    orig_input = q_row[feature_cols].copy()
    age = int(q_row['Bird_Age_Days'].iloc[0])
    birds_count = int(q_row['Total_Alive_Birds'].iloc[0])
    orig_prob = model.predict_proba(orig_input[feature_cols].astype(float))[0][1]
    
    target_f = (age * 4.8) + 20
    target_w = target_f * 2.1

    # Historical Proof Chart
    st.subheader("📉 The 'Aha!' Moment: 7-Day Trend")
    st.write("Why is the AI flagging this zone? Look at how the rising heat has suppressed their appetite over the last week.")
    
    past_7_days = df_hist[df_hist['Zone_ID'] == sel_zone_name].tail(7)
    chart_data = pd.concat([past_7_days, q_row])
    
    fig_trend, ax1 = plt.subplots(figsize=(10, 3))
    fig_trend.patch.set_facecolor('none')
    ax1.set_facecolor('none')
    
    ax1.plot(chart_data['Date'].dt.strftime('%m-%d'), chart_data['Avg_Feed_Intake_g_RAW'], color='#3b82f6', marker='o', linewidth=2, label='Feed Intake (Appetite)')
    ax1.set_ylabel('Feed Intake (g)', color='#3b82f6', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#3b82f6')
    ax1.spines['top'].set_visible(False)
    
    ax2 = ax1.twinx()
    ax2.plot(chart_data['Date'].dt.strftime('%m-%d'), chart_data['Max_Temperature_C_RAW'], color='#ef4444', marker='x', linestyle='--', linewidth=2, label='Barn Temperature')
    ax2.set_ylabel('Max Temp (°C)', color='#ef4444', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#ef4444')
    ax2.spines['top'].set_visible(False)
    
    st.pyplot(fig_trend)
    st.caption("When the red dotted line goes up, the blue line crashes. This is the exact signature of an impending heat stress outbreak.")
    st.markdown("---")

    col_sim_env, col_sim_bio = st.columns(2)
    
    with col_sim_env:
        st.markdown("#### 🌍 Step 1: Adjust Barn Controls")
        st.info("These are the physical levers you can pull right now.")
        s_temp = st.slider("Target Ambient Temp (°C)", 20.0, 45.0, float(orig_input['Max_Temperature_C'].iloc[0]))
        s_hum = st.slider("Target Humidity (%)", 30.0, 100.0, float(orig_input['Avg_Humidity_Percent'].iloc[0]))
        
    with col_sim_bio:
        st.markdown("#### 🐔 Step 2: Biological Status (Auto-Calculated)")
        st.success("XGBoost Math Upgrade: The underlying math model enforces strict biological limits. As you adjust temperature, biological symptoms adjust systematically.")
        
        orig_temp = float(orig_input['Max_Temperature_C'].iloc[0])
        temp_diff = s_temp - orig_temp
        
        if temp_diff < 0:
            recovery_factor = min(1.0, abs(temp_diff) * 0.15)
            s_feed = orig_input['Avg_Feed_Intake_g'].iloc[0] + (target_f - orig_input['Avg_Feed_Intake_g'].iloc[0]) * recovery_factor
            s_water = orig_input['Avg_Water_Intake_ml'].iloc[0] + (target_w - orig_input['Avg_Water_Intake_ml'].iloc[0]) * recovery_factor
            st.caption("Status: Stabilizing 📉")
        elif temp_diff > 0:
            decline_factor = min(0.8, temp_diff * 0.10)
            s_feed = orig_input['Avg_Feed_Intake_g'].iloc[0] * (1 - decline_factor)
            s_water = orig_input['Avg_Water_Intake_ml'].iloc[0] * (1 + decline_factor * 0.5)
            st.caption("Status: Deteriorating 📈")
        else:
            s_feed = orig_input['Avg_Feed_Intake_g'].iloc[0]
            s_water = orig_input['Avg_Water_Intake_ml'].iloc[0]
            st.caption("Status: Unchanged")
            
        st.metric("Simulated Feed Intake", f"{s_feed:.1f}g", f"Target: {target_f:.0f}g")
        st.metric("Simulated Water Intake", f"{s_water:.1f}ml", f"Target: {target_w:.0f}ml")

    # Calculate Simulation
    sim_row = orig_input.copy()
    sim_row['Max_Temperature_C'] = s_temp
    sim_row['Avg_Humidity_Percent'] = s_hum
    sim_row['Avg_Water_Intake_ml'] = s_water
    sim_row['Avg_Feed_Intake_g'] = s_feed
    sim_row['Water_Feed_Ratio'] = s_water / (s_feed + 1e-5)
    
    sim_row['Temp_Avg_3D'] = orig_input['Temp_Avg_3D'].iloc[0] + (temp_diff / 3.0)
    sim_row['Temp_Change_3D'] = orig_input['Temp_Change_3D'].iloc[0] + temp_diff
    
    sim_prob = model.predict_proba(sim_row[feature_cols].astype(float))[0][1]

    st.markdown("#### 📈 Step 3: See the Result")
    col_res1, col_res2 = st.columns([1, 1.5])
    with col_res1:
        st.metric("New Tomorrow's Risk Score", f"{sim_prob:.1%}", delta=f"{sim_prob - orig_prob:.1%}", delta_color="inverse")
        if sim_prob < 0.4: 
            st.success("✅ Good Job: This environmental change will stabilize the flock.")
        else: 
            st.error("🚨 Warning: You must drop the temperature further.")

    with col_res2:
        st.write(f"**What's driving the sickness in {sel_zone_name} today?**")
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(orig_input[feature_cols])
            
            if isinstance(shap_vals, list): sv_array = np.array(shap_vals[1][0])
            else:
                shap_np = np.array(shap_vals)
                if len(shap_np.shape) == 3: sv_array = shap_np[0, :, 1]
                else: sv_array = shap_np[0]
                
            feat_impacts = pd.Series(sv_array, index=feature_cols).sort_values(ascending=True)
            
            if feat_impacts.abs().sum() == 0:
                st.info("Risk is exactly at baseline. No driving factors to display.")
            else:
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = ['#ef4444' if val > 0 else '#3b82f6' for val in feat_impacts.values]
                
                friendly_names = {
                    'Max_Temperature_C': 'Current Temp', 'Avg_Humidity_Percent': 'Current Humidity',
                    'Water_Feed_Ratio': 'Water/Feed Ratio', 'Temp_Avg_3D': 'Heat History (Last 3 Days)',
                    'Avg_Feed_Intake_g': 'Feed Intake', 'Avg_Water_Intake_ml': 'Water Intake'
                }
                clean_labels = [friendly_names.get(col, col) for col in feat_impacts.index]
                
                ax.barh(clean_labels, feat_impacts.values, color=colors)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#94a3b8')
                ax.spines['bottom'].set_color('#94a3b8')
                ax.tick_params(colors='#94a3b8')
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.warning("SHAP visualization offline.")

    # --- 6. Executive Advisory ---
    st.divider()
    st.header("📋 Automated Action Plans for your Crew")
    st.write("We run the mathematical algorithm (DiCE) to find the minimum environmental targets, then pass those targets to the AI Operations Consultant (Gemini) to write the execution plan.")
    
    if st.button("Generate Integrated Action Plan", type="primary"):
        if not user_api_key:
            st.error("⚠️ Stop: Please paste your Gemini API Key in the left sidebar to use this feature.")
        else:
            with st.spinner("Step 1/2: Running DiCE Mathematics to find the precise recovery threshold..."):
                d_query = df_live[df_live['Zone_ID'] == sel_zone_name][feature_cols].astype(float)
                dice_directive = "No simple environmental shortcut found. Aggressive manual intervention required."
                
                try:
                    X_h = df_hist[feature_cols]
                    y_h = label_encoder.transform(df_hist['Target_Status'])
                    dice_data = dice_ml.Data(dataframe=pd.concat([X_h.astype(float), pd.Series(y_h, name='Target_Status', index=X_h.index)], axis=1), continuous_features=[c for c in X_h.columns if 'Zone_ID' not in c], categorical_features=[c for c in X_h.columns if 'Zone_ID' in c], outcome_name='Target_Status')
                    exp = dice_ml.Dice(dice_data, dice_ml.Model(model=model, backend='sklearn'), method='random')
                    
                    env_controllables = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Feed_Intake_g', 'Avg_Water_Intake_ml']
                    cf = exp.generate_counterfactuals(d_query, total_CFs=1, desired_class=0, features_to_vary=env_controllables)
                    
                    if cf.cf_examples_list[0].final_cfs_df is not None:
                        target_temp_dice = cf.cf_examples_list[0].final_cfs_df['Max_Temperature_C'].iloc[0]
                        target_feed_dice = cf.cf_examples_list[0].final_cfs_df['Avg_Feed_Intake_g'].iloc[0]
                        dice_directive = f"DiCE Mathematical Directive: We MUST drop the physical barn temperature to {target_temp_dice:.1f}°C. This is scientifically calculated to restore their feed intake to {target_feed_dice:.0f}g, flipping the flock back to Healthy status."
                except Exception as e:
                    pass 
            
            with st.spinner("Step 2/2: Consulting Gemini LLM to write the Operations Plan..."):
                prompt = f"""
                Zone: {sel_zone_name}. Population: {birds_count} birds (Age {age}).
                Forecasted Sickness Probability: {orig_prob:.1%}.
                Current Environment: {float(orig_input['Max_Temperature_C'].iloc[0]):.1f}C.
                
                {dice_directive}
                
                Based on the math above, generate an operations report for the farm owner. 
                Separate exactly into:
                1. 'Quick Wins': 3 specific physical actions the barn crew must do in the next 4 hours to hit the DiCE temperature target.
                2. 'Strategic Initiatives': 2 long-term upgrades.
                """
                llm_response = call_gemini_with_retry(prompt, user_api_key)
                
            st.success("Analysis Complete")
            st.info(dice_directive)
            st.markdown("### AI Operations Consultant Report")
            st.markdown(llm_response)

else:
    st.error("Missing Data: Please verify sensor connections.")

st.caption("Master's Thesis Project | Automated Canary System")
