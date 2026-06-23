import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

model = joblib.load('fraud_detection_pipeline.pkl')

# --- Sidebar Navigation ---
page = st.sidebar.selectbox("Navigate", ["🔍 Fraud Detector", "📊 Model Performance"])

# =====================
# PAGE 1 - FRAUD DETECTOR
# =====================
if page == "🔍 Fraud Detector":

    st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>🔍 Fraud Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>ML-powered transaction fraud analyzer — Random Forest Model</p>", unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📋 Transaction Info")
        transaction_type = st.selectbox("Transaction Type", ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT'])
        amount = st.number_input("Amount (₹)", min_value=0.0, value=1000.0)

    with col2:
        st.markdown("#### 👤 Sender Details")
        old_balance_orig = st.number_input("Sender Old Balance", min_value=0.0, value=10000.0)
        new_balance_orig = st.number_input("Sender New Balance", min_value=0.0, value=9000.0)

    with col3:
        st.markdown("#### 👥 Receiver Details")
        old_balance_dest = st.number_input("Receiver Old Balance", min_value=0.0, value=0.0)
        new_balance_dest = st.number_input("Receiver New Balance", min_value=0.0, value=0.0)

    st.divider()

    # Risk Score
    risk_score = 0
    if transaction_type in ['TRANSFER', 'CASH_OUT']:
        risk_score += 40
    if new_balance_orig == 0 and old_balance_orig > 0:
        risk_score += 40
    if amount > 50000:
        risk_score += 20
    risk_score = min(risk_score, 100)

    st.markdown("#### 🌡️ Pre-check Risk Indicator")
    if risk_score < 40:
        st.progress(risk_score / 100, text=f"Low Risk — {risk_score}%")
    elif risk_score < 70:
        st.progress(risk_score / 100, text=f"Medium Risk — {risk_score}%")
    else:
        st.progress(risk_score / 100, text=f"High Risk — {risk_score}%")

    st.divider()

    if st.button("🔍 Analyze Transaction", use_container_width=True):
        input_data = pd.DataFrame([{
            'type': transaction_type,
            'amount': amount,
            'oldbalanceOrg': old_balance_orig,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest
        }])

        prediction = model.predict(input_data)[0]
        balance_dropped = old_balance_orig - new_balance_orig

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Amount", f"₹{amount:,.0f}")
        with col_b:
            st.metric("Balance Change (Sender)", f"₹{balance_dropped:,.0f}")
        with col_c:
            st.metric("Risk Score", f"{risk_score}%")

        st.divider()

        if prediction == 1:
            st.error("## 🚨 FRAUD DETECTED!\nThis transaction has been flagged as potentially fraudulent.")
        else:
            st.success("## ✅ TRANSACTION IS SAFE\nThis transaction appears to be legitimate.")

        with st.expander("📄 View Full Transaction Summary"):
            st.json({
                "Transaction Type": transaction_type,
                "Amount": f"₹{amount:,.0f}",
                "Sender Old Balance": f"₹{old_balance_orig:,.0f}",
                "Sender New Balance": f"₹{new_balance_orig:,.0f}",
                "Receiver Old Balance": f"₹{old_balance_dest:,.0f}",
                "Receiver New Balance": f"₹{new_balance_dest:,.0f}",
                "Prediction": "FRAUD 🚨" if prediction == 1 else "SAFE ✅",
                "Risk Score": f"{risk_score}%"
            })

# =====================
# PAGE 2 - MODEL PERFORMANCE
# =====================
elif page == "📊 Model Performance":

    st.markdown("<h1 style='text-align:center;'>📊 Model Performance Comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Trained and evaluated 3 ML models — best one selected for deployment</p>", unsafe_allow_html=True)
    st.divider()

    perf_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
        'Accuracy (%)': [94.70, 99.46, 99.97],
        'Fraud Recall (%)': [92.53, 98.54, 76.79],
        'Fraud Precision (%)': [2.21, 19.18, 97.08],
        'F1 Score (%)': [4.31, 32.11, 85.75]
    })

    st.dataframe(perf_data, use_container_width=True, hide_index=True)
    st.divider()

    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    models_list = perf_data['Model']

    axes[0].bar(models_list, perf_data['Accuracy (%)'], color=colors)
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_ylabel('Accuracy %')
    axes[0].set_ylim([90, 101])
    for i, v in enumerate(perf_data['Accuracy (%)']):
        axes[0].text(i, v + 0.1, f"{v}%", ha='center', fontweight='bold')

    axes[1].bar(models_list, perf_data['Fraud Precision (%)'], color=colors)
    axes[1].set_title('Fraud Precision Comparison')
    axes[1].set_ylabel('Precision %')
    for i, v in enumerate(perf_data['Fraud Precision (%)']):
        axes[1].text(i, v + 0.5, f"{v}%", ha='center', fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

    st.divider()
    st.info("✅ **Random Forest selected** as the final model — highest accuracy (99.97%) and highest fraud precision (97.08%), meaning least false alarms on real customers.")