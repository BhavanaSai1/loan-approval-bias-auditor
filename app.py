import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import shap

# Page config
st.set_page_config(
    page_title="Loan Approval Bias Auditor",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Loan Approval Bias Auditor")
st.markdown("**Auditing racial and gender bias in mortgage approvals using real HMDA data**")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("hmda_sample.csv")
    df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
    df['loan_to_value_ratio'] = pd.to_numeric(df['loan_to_value_ratio'], errors='coerce')
    df['property_value'] = pd.to_numeric(df['property_value'], errors='coerce')
    df['applicant_age'] = pd.to_numeric(df['applicant_age'], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))
    return df

df = load_data()

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🔍 Bias by Race",
    "👥 Bias by Gender",
    "🤖 Predict My Loan",
    "📈 SHAP Analysis"
])

# -------------------- PAGE 1: Overview --------------------
if page == "📊 Overview":
    st.header("📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Approved", f"{df['approved'].sum():,}")
    col3.metric("Denied", f"{(df['approved']==0).sum():,}")
    col4.metric("Approval Rate", f"{round(df['approved'].mean()*100,1)}%")

    st.markdown("---")
    st.subheader("Approval vs Denial Distribution")
    fig, ax = plt.subplots()
    ax.pie(
        [df['approved'].sum(), (df['approved']==0).sum()],
        labels=['Approved', 'Denied'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c']
    )
    st.pyplot(fig)

# -------------------- PAGE 2: Race Bias --------------------
elif page == "🔍 Bias by Race":
    st.header("🔍 Loan Approval Bias by Race")
    st.markdown("**Real findings from 824,920 Texas mortgage applications (2022)**")

    race_map = {
        1: 'American Indian', 2: 'Asian', 3: 'Black',
        4: 'Pacific Islander', 5: 'White',
        6: 'Not Provided', 7: 'Not Applicable'
    }
    df['race_label'] = df['applicant_race-1'].map(race_map)
    race_approval = df.groupby('race_label')['approved'].mean() * 100
    race_approval = race_approval.sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if r in ['Black', 'American Indian'] else '#3498db'
              for r in race_approval.index]
    ax.barh(race_approval.index, race_approval.values, color=colors)
    ax.axvline(x=race_approval['White'], color='black',
               linestyle='--', label='White approval rate')
    ax.set_xlabel("Approval Rate (%)")
    ax.set_title("Loan Approval Rate by Race")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 🚨 Key Finding")
    black_rate = round(race_approval.get('Black', 0), 1)
    white_rate = round(race_approval.get('White', 0), 1)
    gap = round(white_rate - black_rate, 1)
    st.error(f"Black applicants are approved at {black_rate}% vs White applicants at {white_rate}% — a gap of {gap}%")

# -------------------- PAGE 3: Gender Bias --------------------
elif page == "👥 Bias by Gender":
    st.header("👥 Loan Approval Bias by Gender")

    sex_map = {1: 'Male', 2: 'Female', 3: 'Not Provided', 4: 'Not Applicable'}
    df['sex_label'] = df['applicant_sex'].map(sex_map)
    sex_approval = df.groupby('sex_label')['approved'].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#e74c3c' if s == 'Female' else '#3498db' for s in sex_approval.index]
    ax.bar(sex_approval.index, sex_approval.values, color=colors)
    ax.set_ylabel("Approval Rate (%)")
    ax.set_title("Loan Approval Rate by Gender")
    st.pyplot(fig)

    st.markdown("### 🚨 Key Finding")
    male_rate = round(sex_approval.get('Male', 0), 1)
    female_rate = round(sex_approval.get('Female', 0), 1)
    gap = round(male_rate - female_rate, 1)
    st.error(f"Female applicants are approved at {female_rate}% vs Male applicants at {male_rate}% — a gap of {gap}%")

# -------------------- PAGE 4: Predict --------------------
elif page == "🤖 Predict My Loan":
    st.header("🤖 Will Your Loan Get Approved?")
    st.markdown("Enter your details below to see your approval probability")

    col1, col2 = st.columns(2)

    with col1:
        loan_amount = st.number_input("Loan Amount ($)", min_value=10000, max_value=2000000, value=250000)
        income = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=80000)
        property_value = st.number_input("Property Value ($)", min_value=10000, max_value=5000000, value=300000)

    with col2:
        debt_to_income = st.slider("Debt to Income Ratio (%)", 0, 100, 35)
        loan_to_value = st.slider("Loan to Value Ratio (%)", 0, 100, 80)
        loan_purpose = st.selectbox("Loan Purpose", [1, 2, 3, 4, 5],
            format_func=lambda x: {1:'Home Purchase', 2:'Refinancing',
                                    3:'Cash-out Refinancing', 4:'Home Improvement',
                                    5:'Other'}[x])

    race = st.selectbox("Race", [1, 2, 3, 4, 5],
        format_func=lambda x: {1:'American Indian', 2:'Asian', 3:'Black',
                                4:'Pacific Islander', 5:'White'}[x])
    sex = st.selectbox("Gender", [1, 2], format_func=lambda x: {1:'Male', 2:'Female'}[x])
    age = st.slider("Age", 18, 80, 35)
    occupancy_type = st.selectbox("Occupancy Type", [1, 2, 3],
        format_func=lambda x: {1:'Principal Residence', 2:'Second Residence',
                                3:'Investment Property'}[x])

    if st.button("🔍 Check My Approval Chances"):
        input_data = pd.DataFrame([[
            loan_amount, income, debt_to_income, loan_to_value,
            loan_purpose, race, sex, age, property_value, occupancy_type
        ]], columns=[
            'loan_amount', 'income', 'debt_to_income_ratio', 'loan_to_value_ratio',
            'loan_purpose', 'applicant_race-1', 'applicant_sex', 'applicant_age',
            'property_value', 'occupancy_type'
        ])

        prob = model.predict_proba(input_data)[0][1]

        if prob >= 0.7:
            st.success(f"✅ High chance of approval! Probability: {round(prob*100, 1)}%")
        elif prob >= 0.5:
            st.warning(f"⚠️ Moderate chance of approval. Probability: {round(prob*100, 1)}%")
        else:
            st.error(f"❌ Low chance of approval. Probability: {round(prob*100, 1)}%")

# -------------------- PAGE 5: SHAP --------------------
elif page == "📈 SHAP Analysis":
    st.header("📈 What Factors Drive Loan Approval?")
    st.markdown("SHAP values show which features impact the model's decisions the most")

    try:
        st.image("shap_summary.png", caption="SHAP Feature Importance", use_column_width=True)
        st.markdown("### How to read this chart:")
        st.info("Features at the top have the most impact on loan approval decisions. Red means higher value, blue means lower value.")
    except:
        st.warning("SHAP chart not found. Please run bias_audit.py first.")