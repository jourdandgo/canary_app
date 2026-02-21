
import streamlit as st
import pandas as pd
import numpy as np
import shap
import dice_ml
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split # Still used by dice_ml for internal sampling if needed, but not for primary split.
from sklearn.preprocessing import LabelEncoder
import os
from datetime import datetime, timedelta # Needed for preprocessing in Streamlit app

# --- 1. Load the saved model, X_train, and label_encoder ---
@st.cache_resource
def load_artifacts():
    # Load the retrained model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Load the retrained X_train (with predictive features)
    with open('X_train_data_retrained.pkl', 'rb') as f:
        X_train_data = pickle.load(f)
    # The label_encoder is still the same
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, X_train_data, label_encoder

model, X_train, label_encoder = load_artifacts()

# Regenerate full preprocessed data (X_full, y_full) for dice_ml.Data definition
@st.cache_data
def get_full_preprocessed_data(_loaded_label_encoder):
    df_full = pd.read_csv('broiler_health_noisy_dataset.csv')

    df_full['Date'] = pd.to_datetime(df_full['Date'])
    df_full = df_full.sort_values(by=['Zone_ID', 'Date']).reset_index(drop=True)

    df_full['DayOfMonth'] = df_full['Date'].dt.day
    df_full['DayOfWeek'] = df_full['Date'].dt.dayofweek

    df_full['Temp_Avg_3D'] = df_full.groupby('Zone_ID')['Max_Temperature_C'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df_full['Feed_Avg_3D'] = df_full.groupby('Zone_ID')['Avg_Feed_Intake_g'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df_full['Water_Avg_3D'] = df_full.groupby('Zone_ID')['Avg_Water_Intake_ml'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    df_full['Mortality_Rate_Avg_3D'] = df_full.groupby('Zone_ID')['Daily_Mortality_Rate_Pct'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    df_full['Temp_Change_3D'] = df_full.groupby('Zone_ID')['Max_Temperature_C'].diff(periods=3)

    df_full['Target_Health_Status'] = df_full.groupby('Zone_ID')['Health_Status'].shift(-1)

    df_full = df_full.drop('Date', axis=1)
    df_full.dropna(inplace=True)

    df_full = pd.get_dummies(df_full, columns=['Zone_ID'], drop_first=True, dtype=int)

    data_leaking_features = [
        'Health_Status',
        'Daily_Mortality_Count',
        'Daily_Mortality_Rate_Pct',
        'Panting_Percent',
        'Huddling_Percent',
        'Wing_Spreading_Percent',
        'Droppings_Abnormality_Score'
    ]
    df_full = df_full.drop(columns=data_leaking_features, errors='ignore')

    X_full = df_full.drop('Target_Health_Status', axis=1)
    y_full = df_full['Target_Health_Status']
    y_full_encoded = _loaded_label_encoder.transform(y_full)

    return X_full, y_full_encoded

X_full, y_full_encoded = get_full_preprocessed_data(label_encoder)

# We also need X_test for selecting 'At_Risk' instances
# Regenerate X_test here using the same revised preprocessing steps
@st.cache_data
def get_test_set(X_full_data, y_full_data):
    train_size = int(len(X_full_data) * 0.75)
    X_test_df = X_full_data.iloc[train_size:]
    y_test_arr = y_full_data[train_size:]
    return X_test_df, y_test_arr

X_test, y_test = get_test_set(X_full, y_full_encoded)


# --- 2. Streamlit App Layout ---
st.title('Broiler Chicken Health Prediction & Explanations')
st.markdown(
    """
    This application predicts the health status of broiler chickens and provides
    explanations for these predictions using SHAP (global feature importance) and
    DICE (actionable counterfactuals). The model now acts as an Early Warning System,
    predicting **tomorrow's** health status using **today's** data and incorporates
    time-series features to improve predictive power and prevent data leakage.
    """
)

# --- 3. Global Feature Importance (SHAP) ---
st.header('Global Feature Importance')
st.write(
    """
    The plot below shows the most important features influencing the prediction
    of **tomorrow's** 'At_Risk' status in broiler chickens, as determined by SHAP values.
    A higher SHAP value indicates a stronger positive influence on the 'At_Risk' prediction.
    Features like `Temp_Avg_3D` and `Temp_Change_3D` highlight the importance of
    recent environmental conditions and sudden shifts as early indicators.
    """
)

@st.cache_resource
def get_shap_plot(_model_obj, X_data):
    explainer = shap.TreeExplainer(_model_obj)
    # SHAP values for tree models typically return 2 arrays for binary classification
    # [shap_values_for_class_0, shap_values_for_class_1]
    shap_values = explainer.shap_values(X_data)

    # We need to extract the SHAP values for the 'At_Risk' class (class 1).
    # This means taking the second element from the list of SHAP value arrays.
    # Corrected slicing for 3D shap_values output to get (n_samples, n_features)
    shap_values_for_at_risk_class = shap_values[:, :, 1]

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_for_at_risk_class, X_data, plot_type="bar", show=False)
    plt.title('Global Feature Importance (SHAP values for "At_Risk" class)')
    plt.tight_layout()
    return plt

shap_plot = get_shap_plot(model, X_train) # Using X_train for global plot as it represents the overall data distribution
st.pyplot(shap_plot)

# --- 4. Counterfactual Explanations (DICE) ---
st.header('Counterfactual Explanations')
st.write(
    """
    Select an 'At_Risk' chicken instance below to generate counterfactuals.
    Counterfactuals show the minimal changes needed for an 'At_Risk' chicken
    to be classified as 'Healthy' **tomorrow**. These changes suggest proactive
    interventions.
    """
)

# Filter for instances predicted as 'At_Risk' (class 1) using the retrained model
# Ensure X_test indices are aligned with y_test for prediction
at_risk_indices = X_test.index[model.predict(X_test) == 1].tolist()

if at_risk_indices:
    # Select an instance to explain
    selected_index = st.selectbox(
        "Select an 'At_Risk' instance by index from the test set (predicting 'At_Risk' tomorrow):",
        at_risk_indices
    )

    query_instance = X_test.loc[[selected_index]].copy()

    # Explicitly cast numerical columns in query_instance to float for DICE
    for col in query_instance.select_dtypes(include=['number']).columns:
        query_instance[col] = query_instance[col].astype(float)

    st.subheader(f"Selected Instance (Index: {selected_index}, Predicted 'At_Risk' for tomorrow):")
    st.dataframe(query_instance)

    # Recreate DICE objects (need to ensure feature types are correctly passed)
    # Identify continuous features and categorical features based on the full preprocessed data
    # Ensure all numerical columns in X_full_for_dice_streamlit are float type
    X_full_for_dice_streamlit = X_full.copy()
    for col in X_full_for_dice_streamlit.select_dtypes(include=['number']).columns:
        X_full_for_dice_streamlit[col] = X_full_for_dice_streamlit[col].astype(float)

    outcome_series_for_dice_streamlit = pd.Series(y_full_encoded, name='Target_Health_Status', index=X_full_for_dice_streamlit.index).astype(float)

    # Update continuous and categorical features based on new feature engineering
    continuous_features = [col for col in X_full_for_dice_streamlit.columns if 'Zone_ID' not in col]
    categorical_features = [col for col in X_full_for_dice_streamlit.columns if 'Zone_ID' in col]

    dice_data = dice_ml.Data(
        dataframe=pd.concat([X_full_for_dice_streamlit, outcome_series_for_dice_streamlit], axis=1),
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        outcome_name='Target_Health_Status'
    )
    dice_model = dice_ml.Model(model=model, backend='sklearn')
    dice = dice_ml.Dice(dice_data, dice_model, method='random')

    if st.button('Generate Counterfactuals'):
        with st.spinner('Generating counterfactuals... This may take a moment.'):
            # Generate counterfactuals targeting 'Healthy' (class 0)
            cf = dice.generate_counterfactuals(query_instance, total_CFs=3, desired_class=0, verbose=False)
            st.subheader("Counterfactual Explanations (Changes needed for **tomorrow's** prediction to become 'Healthy'):")
            st.write(cf.visualize_as_dataframe(show_only_changes=True))
else:
    st.warning("No instances predicted as 'At_Risk' in the test set. Cannot generate counterfactuals for future prediction.")

st.markdown(
    """
    ---
    *Note: The model and data used in this application are for demonstration purposes.* 
    """
)
