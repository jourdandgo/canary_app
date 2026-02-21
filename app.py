import streamlit as st
import pandas as pd
import numpy as np
import shap
import dice_ml
import matplotlib.pyplot as plt
import pickle
import os
from datetime import datetime, timedelta

# Page Setup
st.set_page_config(page_title="Canary Early Warning Dashboard", layout="wide")

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

# --- 2. Data Preprocessing (Predictive Logic) ---
# We use the underscore _label_encoder to prevent Streamlit hashing errors
@st.cache_data
def get_processed_data(_label_encoder):
    df = pd.read_csv('broiler_health_noisy_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Zone_ID', 'Date']).reset_index(drop=True)

    # Feature Engineering (Must match the notebook exactly)
    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Feed_Avg_3D'] = df.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Water_Avg_3D'] = df.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Mortality_Rate_Avg_3D'] = df.groupby('Zone_ID')['Daily_Mortality_Rate_Pct'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)

    # Shift Target to predict TOMORROW'S health using TODAY'S data
    df['Target_Health_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)
    df = df.drop('Date', axis=1).dropna()

    # One-Hot Encoding for Zone IDs
    df = pd.get_dummies(df, columns=['Zone_ID'], drop_first=True, dtype=int)
    
    # Remove features that leak "current" health (behavioral metrics)
    leakage = ['Health_Status', 'Daily_Mortality_Count', 'Daily_Mortality_Rate_Pct', 
               'Panting_Percent', 'Huddling_Percent', 'Wing_Spreading_Percent', 'Droppings_Abnormality_Score']
    df = df.drop(columns=leakage, errors='ignore')

    X = df.drop('Target_Health_Status', axis=1)
    y_encoded = _label_encoder.transform(df['Target_Health_Status'])

    return X, y_encoded

X_full, y_full_encoded = get_processed_data(label_encoder)

# Create Test Set for display (last 25% of data)
train_size = int(len(X_full) * 0.75)
X_test = X_full.iloc[train_size:]

# --- 3. Dashboard UI ---
st.title('🐔 Automated Canary: Poultry Health Dashboard')
st.markdown("### Early Warning System & Prescriptive Intervention Planner")

# Run Global Predictions
preds = model.predict(X_test)
at_risk_df = X_test[preds == 1].copy()

col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("⚠️ Prediction Alerts")
    if not at_risk_df.empty:
        st.error(f"Found {len(at_risk_df)} zones predicted 'At Risk' for tomorrow.")
        zone_cols = [c for c in at_risk_df.columns if 'Zone_ID' in c]
        for idx, row in at_risk_df.head(5).iterrows():
            zone = "Zone_A" # Default if other dummies are 0
            for c in zone_cols:
                if row[c] == 1: zone = c.replace("Zone_ID_", "")
            st.warning(f"**Alert:** {zone} (Record ID: {idx})")
    else:
        st.success("✅ All zones predicted 'Healthy' for tomorrow.")

with col_b:
    # SHAP Feature Importance
    @st.cache_data
    def get_shap_vals(_model, X_data):
        explainer = shap.TreeExplainer(_model)
        return explainer.shap_values(X_data)[:, :, 1]

    st.subheader("📊 Primary Risk Drivers (Global)")
    s_vals = get_shap_vals(model, X_train)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(s_vals, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig)

# --- 4. Prescriptive Intervention Planner (DiCE) ---
st.divider()
st.header("🛠️ Prescriptive Intervention Planner")
st.info("The AI identifies controllable environmental factors to return 'At Risk' zones to health.")

if not at_risk_df.empty:
    selected_record = st.selectbox("Select an 'At Risk' Record to Analyze:", at_risk_df.index)
    query = X_test.loc[[selected_record]].astype(float)
    
    # Farmers can only control these 4 things
    controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']
    
    # Setup DiCE
    dice_data_df = pd.concat([X_full.astype(float), pd.Series(y_full_encoded, name='Target_Health_Status', index=X_full.index)], axis=1)
    d_data = dice_ml.Data(dataframe=dice_data_df, 
                          continuous_features=[c for c in X_full.columns if 'Zone_ID' not in c],
                          categorical_features=[c for c in X_full.columns if 'Zone_ID' in c],
                          outcome_name='Target_Health_Status')
    
    m_dice = dice_ml.Model(model=model, backend='sklearn')
    exp_dice = dice_ml.Dice(d_data, m_dice, method='random')

    if st.button("Generate Actionable Recommendations"):
        with st.spinner("Calculating optimal interventions..."):
            # We tell the AI to ONLY vary the controllable features
            cf = exp_dice.generate_counterfactuals(query, total_CFs=2, 
                                                   desired_class=0, 
                                                   features_to_vary=controllable)
            
            st.markdown("#### ✅ Recommended Environmental Adjustments")
            res_df = cf.cf_examples_list[0].final_cfs_df
            
            # Actionable Text logic
            for i, row in res_df.iterrows():
                st.write(f"**Intervention Strategy {i+1}:**")
                changes_found = False
                for feat in controllable:
                    orig = query[feat].values[0]
                    reco = row[feat]
                    
                    if abs(orig - reco) > 0.05:
                        changes_found = True
                        direction = "Increase" if reco > orig else "Decrease"
                        label = feat.replace('_', ' ')
                        st.markdown(f"- {direction} **{label}** from {orig:.1f} to **{reco:.1f}**")
                
                if not changes_found:
                    st.write("- No simple environmental changes identified. Check bird health manually.")
            
            st.subheader("Data Comparison Table")
            st.dataframe(res_df.style.highlight_max(axis=0, color='#1b5e20'))

else:
    st.write("No 'At Risk' incidents predicted. The flock is currently stable.")

st.markdown("---")
st.caption("Master's Thesis | Poultry Health Monitoring System | Explainable & Prescriptive AI")
