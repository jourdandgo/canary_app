🐔 Automated Canary Dashboard: Broiler Farm Early Warning System

Overview

This repository contains a Master's-level Machine Learning project designed to act as a Predictive Early Warning System and Root Cause Analyzer for commercial broiler farms.

Instead of relying on manual, diagnostic observations, this project uses environmental data (temperature, humidity) and biological behavioral metrics (panting, huddling, feed/water intake) to predict imminent flock health risks using lag features and rolling averages.

Key Academic Features

Realistic Synthetic Data Generation: Includes a robust data generator (generate_noisy_data.py) that simulates continuous flock lifecycles. It removes perfect class separation by introducing overlapping distributions, biological noise, and a hidden "cumulative stress" variable to rigorously test ML algorithms.

Predictive vs. Diagnostic: Utilizes Time-Series splits and 3-day rolling lag features to predict tomorrow's health status using today's data, preventing temporal data leakage.

Explainable AI (SHAP): Integrates SHAP (SHapley Additive exPlanations) to provide farm managers with waterfall plots explaining exactly why an AI agent flagged a zone as "At Risk" (e.g., identifying temperature vs. humidity as the primary driver).

Prescriptive Analytics (DiCE): Utilizes Diverse Counterfactual Explanations (DiCE) to generate actionable "what-if" scenarios, telling the farm manager exactly which environmental levers to adjust to return the flock to a healthy state.

Tech Stack

Data Processing & ML: Python, Pandas, NumPy, Scikit-Learn, XGBoost / Random Forest

Explainable AI (XAI): SHAP, DiCE (Diverse Counterfactual Explanations)

Deployment: Streamlit (for the interactive front-end dashboard)

Project Structure

generate_noisy_data.py: Script to generate the synthetic, noisy time-series dataset.

Automated_Canary_Streamlit.ipynb: The core Jupyter Notebook containing the data preprocessing, model training, and XAI pipeline.

app.py: The deployment-ready Streamlit dashboard script.

requirements.txt: Dependencies required to run the pipeline and web app.
