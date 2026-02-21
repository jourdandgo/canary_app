
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
@st.cache_data
def get_processed_data(_label_encoder):
    df = pd.read_csv('broiler_health_noisy_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Zone_ID', 'Date']).reset_index(drop=True)

    # Store the dates before dropping them so we can filter by 'Latest'
    df_with_dates = df.copy()

    # Feature Engineering
    df['DayOfMonth'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Temp_Avg_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Feed_Avg_3D'] = df.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Water_Avg_3D'] = df.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Mortality_Rate_Avg_3D'] = df.groupby('Zone_ID')['Daily_Mortality_Rate_Pct'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Temp_Change_3D'] = df.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3).fillna(0)

    # Shift Target
    df['Target_Health_Status'] = df.groupby('Zone_ID')['Health_Status'].shift(-1)

    # We keep the Date and Target for a moment to filter
    df_processed = df.dropna().copy()

    # One-Hot Encoding for Zone IDs
    df_encoded = pd.get_dummies(df_processed, columns=['Zone_ID'], drop_first=True, dtype=int)

    # Remove features that leak "current" health
    leakage = ['Health_Status', 'Daily_Mortality_Count', 'Daily_Mortality_Rate_Pct',
               'Panting_Percent', 'Huddling_Percent', 'Wing_Spreading_Percent', 'Droppings_Abnormality_Score']
    df_encoded = df_encoded.drop(columns=leakage, errors='ignore')

    return df_encoded

df_final = get_processed_data(label_encoder)

# --- NEW LOGIC: Filter for ONLY the latest data point per zone ---
latest_date = df_final['Date'].max()
X_full = df_final.drop(['Date', 'Target_Health_Status'], axis=1)
y_full_encoded = label_encoder.transform(df_final['Target_Health_Status'])

# This is what we show in the "Alerts" - only things happening on the most recent day
latest_day_data = df_final[df_final['Date'] == latest_date].copy()
X_latest = latest_day_data.drop(['Date', 'Target_Health_Status'], axis=1)

# Make predictions for the latest day
preds_latest = model.predict(X_latest)
latest_day_data['Predicted_Health_Status'] = label_encoder.inverse_transform(preds_latest)

# Filter for 'At_Risk' predictions on the latest day
at_risk_latest = latest_day_data[latest_day_data['Predicted_Health_Status'] == 'At_Risk']

# --- 3. Dashboard UI ---
st.title('🐔 Automated Canary: Poultry Health Dashboard')
st.markdown(
    """
    Welcome to the **Automated Canary: Poultry Health Dashboard**! This tool provides an early warning system
    for broiler chicken health, leveraging AI to predict potential 'At-Risk' statuses *before* they fully develop.

    The goal is to empower farm managers with actionable insights, enabling proactive interventions to maintain
    flock health, optimize production, and reduce losses.

    This dashboard integrates a machine learning model with eXplainable AI (XAI) techniques (SHAP and DiCE)
    to not only predict but also explain *why* a prediction was made and *how* to change an unfavorable outcome.
    """
)

st.markdown(f"### Monitoring Status for: **{latest_date.strftime('%Y-%m-%d')}**")

col_a, col_b = st.columns([1, 2])

with col_a:
    st.subheader("⚠️ Current Risk Alerts")
    st.markdown("Here, you'll see immediate alerts for any zones predicted to be 'At-Risk' for **tomorrow**. This is your first line of defense, providing a heads-up before issues escalate.")
    if not at_risk_latest.empty:
        st.error(f"Action Required: {len(at_risk_latest)} zones are at risk for TOMORROW.")
        zone_cols = [c for c in at_risk_latest.columns if 'Zone_ID' in c]
        for idx, row in at_risk_latest.iterrows():
            zone = "Zone_A" # Default in case no Zone_ID is set (e.g., if only Zone_A exists and is not one-hot encoded)
            # Determine the original Zone_ID from one-hot encoded columns
            # This logic needs to correctly reconstruct the Zone_ID
            # A better approach is to store the original Zone_ID in latest_day_data before one-hot encoding
            # For now, let's assume we can derive it from the one-hot columns
            zones_identified = []
            if 'Zone_ID_Zone_B' in row and row['Zone_ID_Zone_B'] == 1: zones_identified.append('Zone_B')
            if 'Zone_ID_Zone_C' in row and row['Zone_ID_Zone_C'] == 1: zones_identified.append('Zone_C')
            if 'Zone_ID_Zone_D' in row and row['Zone_ID_Zone_D'] == 1: zones_identified.append('Zone_D')
            
            if not zones_identified: # If no specific Zone_ID is 1, it must be the base (Zone_A)
                actual_zone = 'Zone_A'
            else:
                actual_zone = zones_identified[0] # Assuming only one zone is active per row

            st.warning(f"**ALERT:** {actual_zone} is showing early warning signs.")
    else:
        st.success("✅ All zones are currently stable for tomorrow's forecast.")

with col_b:
    # SHAP Feature Importance (Using X_train for global context)
    @st.cache_data
    def get_shap_vals(_model, X_data):
        explainer = shap.TreeExplainer(_model)
        # Corrected slicing for 3D shap_values output to get (n_samples, n_features)
        return explainer.shap_values(X_data)[:, :, 1]

    st.subheader("📊 Why does the AI flag risk? (Global Feature Importance)")
    st.markdown("This section helps you understand which factors generally contribute most to a chicken being predicted 'At-Risk'. The longer the bar, the more influence that feature has on the model's 'At-Risk' predictions.")
    s_vals = get_shap_vals(model, X_train)
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.summary_plot(s_vals, X_train, plot_type="bar", show=False)
    plt.title('Global Feature Importance (SHAP values for "At_Risk" class)')
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown(
        """
        **Key Insights from SHAP:**
        *   **Environmental Factors:** `Max_Temperature_C`, `Avg_Humidity_Percent`, and `Temp_Change_3D` are often critical, highlighting the impact of barn climate on bird health.
        *   **Behavioral Indicators:** Changes in `Avg_Feed_Intake_g` and `Avg_Water_Intake_ml` are strong signals, suggesting that monitoring consumption patterns is vital.
        *   **Demographics:** `Bird_Age_Days` and `Total_Alive_Birds` also play a role, reflecting age-related vulnerabilities and flock density.
        """
    )

# --- 4. Prescriptive Intervention Planner ---
st.divider()
st.header("🛠️ Prescriptive Intervention Planner: What can you do?")
st.markdown(
    """
    This powerful tool uses **DiCE (Diverse Counterfactual Explanations)** to suggest minimal, actionable changes
    that could shift an 'At-Risk' prediction to 'Healthy'. Think of it as a "What if?" scenario planner.

    **How to use it:**
    1.  **Select an 'At-Risk' record:** Choose a specific instance (e.g., a zone on a particular day) that the AI flagged.
    2.  **Generate Action Plan:** Click the button to see the suggested changes.
    3.  **Implement & Monitor:** These changes represent the *most efficient* ways to improve the predicted health status.
        Implementing them can help prevent actual health issues.

    *Note: Only controllable features (e.g., temperature, humidity, feed/water intake) are suggested for change.*"""
)

analysis_mode = st.radio("Analysis Mode:", ["Fix Current Risks", "Analyze Historical Risks"])

if analysis_mode == "Fix Current Risks":
    # Use at_risk_latest directly for current risks
    analysis_df_for_selection = at_risk_latest.drop(['Date', 'Predicted_Health_Status', 'Target_Health_Status'], axis=1, errors='ignore').copy()
    if analysis_df_for_selection.empty: st.info("No current 'At Risk' incidents to analyze for prescriptive interventions.")
else:
    # All historical at-risk instances in the test set
    # df_final contains 'Date', so we need to split it again for test block
    train_size = int(len(df_final) * 0.75)
    test_block_full_df = df_final.iloc[train_size:]
    X_test_historical = test_block_full_df.drop(['Date', 'Target_Health_Status'], axis=1)
    y_pred_historical = model.predict(X_test_historical)
    
    # Filter for historical 'At_Risk' predictions (class 1)
    at_risk_historical = test_block_full_df[y_pred_historical == 1].copy()
    analysis_df_for_selection = at_risk_historical.drop(['Date', 'Target_Health_Status'], axis=1, errors='ignore')
    if analysis_df_for_selection.empty: st.info("No historical 'At Risk' incidents found in the test set to analyze.")

if not analysis_df_for_selection.empty:
    # Create a display name for the selectbox that includes zone and other key info
    display_options = []
    for idx, row in analysis_df_for_selection.iterrows():
        zone = "Zone_A" # Default
        zones_identified = []
        if 'Zone_ID_Zone_B' in row and row['Zone_ID_Zone_B'] == 1: zones_identified.append('Zone_B')
        if 'Zone_ID_Zone_C' in row and row['Zone_ID_Zone_C'] == 1: zones_identified.append('Zone_C')
        if 'Zone_ID_Zone_D' in row and row['Zone_ID_Zone_D'] == 1: zones_identified.append('Zone_D')
        
        if not zones_identified: 
            zone = 'Zone_A'
        else:
            zone = zones_identified[0] 
            
        display_options.append(f"Zone: {zone}, Age: {int(row['Bird_Age_Days'])}, Temp: {row['Max_Temperature_C']}C")

    # Map display options back to original indices
    option_to_index = {opt: idx for opt, idx in zip(display_options, analysis_df_for_selection.index)}

    selected_display_option = st.selectbox("Select a Record to analyze for intervention:", display_options)
    selected_record_index = option_to_index[selected_display_option]
    
    query = analysis_df_for_selection.loc[[selected_record_index]].astype(float)

    controllable = ['Max_Temperature_C', 'Avg_Humidity_Percent', 'Avg_Water_Intake_ml', 'Avg_Feed_Intake_g']

    # Setup DiCE
    dice_data_df = pd.concat([X_full.astype(float), pd.Series(y_full_encoded, name='Target_Health_Status', index=df_final.index)], axis=1)
    d_data = dice_ml.Data(dataframe=dice_data_df,
                          continuous_features=[c for c in X_full.columns if 'Zone_ID' not in c],
                          categorical_features=[c for c in X_full.columns if 'Zone_ID' in c],
                          outcome_name='Target_Health_Status')

    m_dice = dice_ml.Model(model=model, backend='sklearn')
    exp_dice = dice_ml.Dice(d_data, m_dice, method='random')

    st.markdown(f"**Analyzing instance (Original Index: {selected_record_index}):**")
    st.dataframe(query)

    if st.button("Generate Action Plan"):
        with st.spinner("Calculating optimal interventions..."):
            cf = exp_dice.generate_counterfactuals(query, total_CFs=2,
                                                   desired_class=0,
                                                   features_to_vary=controllable)

            st.markdown("#### ✅ Recommended Recovery Strategy")
            res_df = cf.cf_examples_list[0].final_cfs_df

            for i, row in res_df.iterrows():
                st.write(f"**Option {i+1}:**")
                changes_found = False
                for feat in controllable:
                    orig = query[feat].values[0]
                    reco = row[feat]
                    if abs(orig - reco) > 0.05: # Threshold for showing a change
                        changes_found = True
                        direction = "Increase" if reco > orig else "Decrease"
                        st.markdown(f"- {direction} **{feat.replace('_', ' ')}** to **{reco:.1f}**")

                if not changes_found:
                    st.write("- No significant changes in controllable features found for a quick fix. Consider reviewing other factors like pathogen testing or equipment maintenance.")

            st.subheader("Comparison Table (Original vs. Suggested Actions)")
            st.dataframe(pd.concat([query, res_df]))

else:
    st.info("Select an 'At Risk' incident above to see intervention recommendations.")

st.markdown("--- ---")
st.header("📚 Model Performance & Technical Details")
st.markdown(
    f"""
    The underlying Random Forest Classifier model serves as an Early Warning System, predicting **tomorrow's** health status.
    It was trained on a comprehensive dataset incorporating various environmental and behavioral metrics, including
    time-series features like 3-day rolling averages and temperature changes, to enhance predictive power and avoid data leakage.

    **Performance on Test Set:**
    *   **Accuracy:** {accuracy_retrained:.4f} - Overall correctness of predictions.
    *   **Precision:** {precision_retrained:.4f} - Of all predicted 'At-Risk' cases, how many were actually 'At-Risk'.
    *   **Recall:** {recall_retrained:.4f} - Of all actual 'At-Risk' cases, how many did we correctly identify.
    *   **F1-Score:** {f1_retrained:.4f} - A balanced measure of precision and recall.

    These metrics indicate a robust model, particularly strong in identifying actual 'At-Risk' cases (high Recall), crucial for an early warning system.
    """
)

st.markdown("--- ---")
st.subheader("💡 Overall Recommendations & Next Steps")
st.markdown(
    """
    *   **Daily Monitoring:** Regularly check the "Current Risk Alerts" to identify immediate areas of concern.
    *   **Targeted Interventions:** Utilize the "Prescriptive Intervention Planner" to apply precise changes based on DiCE recommendations.
    *   **Long-term Strategy:** Analyze global feature importance (SHAP) to understand systemic factors and implement preventative measures across the farm.
    *   **Data Integration:** Consider integrating this system with real-time sensor data for automated alerts and faster response times.
    *   **Veterinary Consultation:** Always consult with a veterinarian for definitive diagnoses and treatment plans, especially if interventions don't yield desired results.
    """
)

st.markdown("--- --- ---")
st.caption("Master's Project | Canary Early Warning System | Prescriptive Analytics")
