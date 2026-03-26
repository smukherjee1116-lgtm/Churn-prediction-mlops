import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import json
import os

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load model artifacts ──────────────────────────────────
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, '..', 'models')
    with open(os.path.join(model_path, 'churn_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_path, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_path, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ── Title ─────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("*End-to-end ML system for predicting customer churn*")
st.divider()

# ── Sidebar navigation ────────────────────────────────────
page = st.sidebar.selectbox(
    "Navigate",
    ["🔮 Predict Churn", "📈 Drift Monitor", "ℹ️ Model Info"]
)

# ═══════════════════════════════════════════════════════════
# PAGE 1 — PREDICT CHURN
# ═══════════════════════════════════════════════════════════
if page == "🔮 Predict Churn":
    st.header("🔮 Predict Customer Churn")
    st.markdown("Enter customer details below to get a churn prediction.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Account Info")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract Type",
                                 ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method",
                                ["Electronic check", "Mailed check",
                                 "Bank transfer (automatic)",
                                 "Credit card (automatic)"])

    with col2:
        st.subheader("Demographics")
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])

    with col3:
        st.subheader("Services & Charges")
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
        num_services = st.slider("Number of Services", 0, 7, 3)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        internet = st.selectbox("Internet Service",
                                 ["Fiber optic", "DSL", "No"])

    # ── Predict button ──
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        
        # Build input
        is_month_to_month = 1 if contract == "Month-to-month" else 0
        is_high_value = 1 if monthly_charges > 64.76 else 0
        total_charges = monthly_charges * tenure
        avg_monthly_spend = total_charges / (tenure + 1)
        charge_per_service = monthly_charges / (num_services + 1)
        tenure_group = 0 if tenure <= 12 else (1 if tenure <= 36 else 2)

        input_dict = {
            'SeniorCitizen': 1 if senior == "Yes" else 0,
            'gender': 1 if gender == "Female" else 0,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'tenure': tenure,
            'PhoneService': 1 if phone_service == "Yes" else 0,
            'PaperlessBilling': 1 if paperless == "Yes" else 0,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'NumServices': num_services,
            'AvgMonthlySpend': avg_monthly_spend,
            'IsMonthToMonth': is_month_to_month,
            'ChargePerService': charge_per_service,
            'IsHighValue': is_high_value,
            'TenureGroup': tenure_group
        }

        input_df = pd.DataFrame([input_dict])
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_names]

        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                    'AvgMonthlySpend', 'ChargePerService']
        input_df[num_cols] = scaler.transform(input_df[num_cols])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # ── Display results ──
        st.divider()
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            if prediction == 1:
                st.error("⚠️ Customer Likely to CHURN!")
            else:
                st.success("✅ Customer Likely to STAY!")

        with col_b:
            st.metric("Churn Probability", f"{probability:.1%}")

        with col_c:
            risk = "🔴 High" if probability > 0.7 else \
                   "🟡 Medium" if probability > 0.4 else "🟢 Low"
            st.metric("Risk Level", risk)

        # ── Probability gauge ──
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh(["Churn Risk"], [probability],
                color='red' if probability > 0.7 else
                      'orange' if probability > 0.4 else 'green',
                height=0.4)
        ax.barh(["Churn Risk"], [1 - probability],
                left=[probability], color='lightgray', height=0.4)
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
        ax.set_title(f'Churn Probability: {probability:.1%}')
        ax.set_xlabel('Probability')
        st.pyplot(fig)

# ═══════════════════════════════════════════════════════════
# PAGE 2 — DRIFT MONITOR
# ═══════════════════════════════════════════════════════════
elif page == "📈 Drift Monitor":
    st.header("📈 Data Drift Monitoring")

    # Load drift report
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'reports', 'drift_report.json')

    if os.path.exists(report_path):
        with open(report_path) as f:
            drift_results = json.load(f)

        st.subheader("Feature Drift Summary")

        cols = st.columns(len(drift_results))
        for i, (feature, result) in enumerate(drift_results.items()):
            with cols[i]:
                if result['drift_detected']:
                    st.error(f"🔴 {feature}")
                else:
                    st.success(f"🟢 {feature}")
                st.metric("PSI", result['psi'])
                st.metric("P-Value", result['p_value'])

        # Summary metrics
        st.divider()
        drifted = sum(1 for r in drift_results.values()
                      if r['drift_detected'])
        total = len(drift_results)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Features", total)
        col2.metric("Drifted Features", drifted,
                    delta=f"{drifted} need attention",
                    delta_color="inverse")
        col3.metric("Drift Rate", f"{drifted/total:.0%}")

        if drifted > total * 0.5:
            st.error("🚨 More than 50% features drifted! "
                     "Consider retraining the model.")
        else:
            st.success("✅ Drift within acceptable limits.")
    else:
        st.warning("No drift report found. "
                   "Run the drift monitoring notebook first!")

# ═══════════════════════════════════════════════════════════
# PAGE 3 — MODEL INFO
# ═══════════════════════════════════════════════════════════
elif page == "ℹ️ Model Info":
    st.header("ℹ️ Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Details")
        st.info("""
        **Model:** XGBoost Classifier
        **Tuning:** Optuna (50 trials)
        **Features:** 37 (including 6 engineered)
        **Training Data:** Telco Customer Churn Dataset
        **Records:** 7,043 customers
        """)

        st.subheader("Performance Metrics")
        metrics = {
            'Metric': ['AUC-ROC', 'F1 Score',
                        'Precision', 'Recall'],
            'Score': ['0.84+', '0.62+', '0.52+', '0.77+']
        }
        st.table(pd.DataFrame(metrics))

    with col2:
        st.subheader("Top Predictors of Churn")
        features = ['ChargePerService', 'TotalCharges',
                    'AvgMonthlySpend', 'tenure',
                    'MonthlyCharges', 'IsMonthToMonth']
        importance = [0.155, 0.109, 0.099,
                      0.098, 0.094, 0.078]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(features[::-1], importance[::-1],
                color='steelblue')
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Tech Stack")
        st.success("""
        🐍 Python | 🤖 XGBoost | 📊 MLflow
        🚀 FastAPI | 🐳 Docker | 📈 Streamlit
        """)