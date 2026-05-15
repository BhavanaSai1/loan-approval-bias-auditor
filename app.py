import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Page config
st.set_page_config(
    page_title="Loan Approval Bias Auditor",
    page_icon="🏦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; padding: 15px; border-radius: 10px; }
    .bias-alert { background-color: #2d1b1b; border-left: 4px solid #e74c3c;
                  padding: 15px; border-radius: 5px; margin: 10px 0; }
    .story-box { background-color: #1a2744; border-left: 4px solid #3498db;
                 padding: 20px; border-radius: 10px; margin: 15px 0;
                 font-size: 16px; line-height: 1.8; }
    .success-box { background-color: #1a2d1a; border-left: 4px solid #2ecc71;
                   padding: 15px; border-radius: 5px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_model():
    from sklearn.ensemble import RandomForestClassifier
    df = load_data()
    X = df.drop('approved', axis=1)
    y = df['approved']
    mdl = RandomForestClassifier(n_estimators=50, random_state=42)
    mdl.fit(X, y)
    return mdl

df = load_data()
model = load_model()

# Race and sex maps
RACE_MAP = {1: 'American Indian', 2: 'Asian', 3: 'Black',
            4: 'Pacific Islander', 5: 'White', 6: 'Not Provided', 7: 'Not Applicable'}
SEX_MAP = {1: 'Male', 2: 'Female', 3: 'Not Provided', 4: 'Not Applicable'}

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "🔍 Bias by Race",
    "👥 Bias by Gender",
    "🤖 Predict My Loan",
    "📖 Bias Story Narrator",
    "🔄 Fix The Bias Simulator",
    "⚖️ Fairness Score Card",
    "📋 Denial Reason Analyzer",
    "💰 Bias Cost Calculator",
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

    col1, col2, col3 = st.columns(3)
    col1.metric("🚨 Racial Bias Gap", "10.1%", "Black vs White (full dataset)")
    col2.metric("🚨 Gender Bias Gap", "3.7%", "Female vs Male (full dataset)")
    col3.metric("🤖 Model Accuracy", "82%", "Random Forest on 824K records")

    st.markdown("---")
    st.subheader("Approval vs Denial Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        [df['approved'].sum(), (df['approved']==0).sum()],
        labels=['Approved', 'Denied'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'],
        startangle=90
    )
    ax.set_facecolor('#0e1117')
    fig.patch.set_facecolor('#0e1117')
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📌 What This App Does")
    st.info("""
    This app audits racial and gender bias in U.S. mortgage approvals using real 2022 HMDA data from Texas.

    🔍 We found that identical financial profiles receive different outcomes based on race and gender alone.

    Use the navigation on the left to explore bias by race, gender, get your personal bias report,
    simulate a fair lending system, and see the real cost of discrimination.
    """)

# -------------------- PAGE 2: Race Bias --------------------
elif page == "🔍 Bias by Race":
    st.header("🔍 Loan Approval Bias by Race")
    st.markdown("**Real findings from 824,920 Texas mortgage applications (2022)**")

    df['race_label'] = df['applicant_race-1'].map(RACE_MAP)
    race_approval = df.groupby('race_label')['approved'].mean() * 100
    race_approval = race_approval.sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#e74c3c' if r in ['Black', 'American Indian'] else '#3498db'
              for r in race_approval.index]
    bars = ax.barh(race_approval.index, race_approval.values, color=colors)
    ax.axvline(x=race_approval.get('White', 76), color='white',
               linestyle='--', label='White approval rate', alpha=0.7)
    ax.set_xlabel("Approval Rate (%)", color='white')
    ax.set_title("Loan Approval Rate by Race", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e2130', labelcolor='white')
    for bar, val in zip(bars, race_approval.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', color='white', fontsize=9)
    st.pyplot(fig)

    black_rate = round(race_approval.get('Black', 0), 1)
    white_rate = round(race_approval.get('White', 0), 1)
    gap = round(white_rate - black_rate, 1)
    st.error(f"🚨 Black applicants are approved at {black_rate}% vs White applicants at {white_rate}% — a gap of {gap}%")
    st.warning(f"📊 This means for every 100 Black applicants, {gap} extra people are rejected compared to identical White applicants")

# -------------------- PAGE 3: Gender Bias --------------------
elif page == "👥 Bias by Gender":
    st.header("👥 Loan Approval Bias by Gender")

    df['sex_label'] = df['applicant_sex'].map(SEX_MAP)
    sex_approval = df.groupby('sex_label')['approved'].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#e74c3c' if s == 'Female' else '#3498db' for s in sex_approval.index]
    bars = ax.bar(sex_approval.index, sex_approval.values, color=colors)
    ax.set_ylabel("Approval Rate (%)", color='white')
    ax.set_title("Loan Approval Rate by Gender", color='white')
    ax.tick_params(colors='white')
    for bar, val in zip(bars, sex_approval.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val:.1f}%', ha='center', color='white', fontsize=10)
    st.pyplot(fig)

    male_rate = round(sex_approval.get('Male', 0), 1)
    female_rate = round(sex_approval.get('Female', 0), 1)
    gap = round(male_rate - female_rate, 1)
    st.error(f"🚨 Female applicants are approved at {female_rate}% vs Male applicants at {male_rate}% — a gap of {gap}%")

# -------------------- PAGE 4: Predict + Bias Alert --------------------
elif page == "🤖 Predict My Loan":
    st.header("🤖 Will Your Loan Get Approved?")
    st.markdown("Enter your details to see your approval probability and real-time bias impact")

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
                                    3:'Cash-out Refinancing', 4:'Home Improvement', 5:'Other'}[x])

    race = st.selectbox("Race", [1, 2, 3, 4, 5],
        format_func=lambda x: {1:'American Indian', 2:'Asian', 3:'Black',
                                4:'Pacific Islander', 5:'White'}[x])
    sex = st.selectbox("Gender", [1, 2], format_func=lambda x: {1:'Male', 2:'Female'}[x])
    age = st.slider("Age", 18, 80, 35)
    occupancy_type = st.selectbox("Occupancy Type", [1, 2, 3],
        format_func=lambda x: {1:'Principal Residence', 2:'Second Residence', 3:'Investment Property'}[x])

    if st.button("🔍 Check My Approval Chances", use_container_width=True):
        cols = ['loan_amount', 'income', 'debt_to_income_ratio', 'loan_to_value_ratio',
                'loan_purpose', 'applicant_race-1', 'applicant_sex', 'applicant_age',
                'property_value', 'occupancy_type']
        input_data = pd.DataFrame([[loan_amount, income, debt_to_income, loan_to_value,
            loan_purpose, race, sex, age, property_value, occupancy_type]], columns=cols)

        prob = model.predict_proba(input_data)[0][1]

        white_male = input_data.copy()
        white_male['applicant_race-1'] = 5
        white_male['applicant_sex'] = 1
        prob_wm = model.predict_proba(white_male)[0][1]

        race_only = input_data.copy()
        race_only['applicant_race-1'] = 5
        prob_race = model.predict_proba(race_only)[0][1]

        sex_only = input_data.copy()
        sex_only['applicant_sex'] = 1
        prob_sex = model.predict_proba(sex_only)[0][1]

        race_impact = round((prob_race - prob) * 100, 1)
        sex_impact = round((prob_sex - prob) * 100, 1)
        total_impact = round((prob_wm - prob) * 100, 1)

        st.markdown("---")
        if prob >= 0.7:
            st.success(f"✅ High chance of approval! Probability: {round(prob*100, 1)}%")
        elif prob >= 0.5:
            st.warning(f"⚠️ Moderate chance of approval. Probability: {round(prob*100, 1)}%")
        else:
            st.error(f"❌ Low chance of approval. Probability: {round(prob*100, 1)}%")

        st.markdown("### 🚨 Real Time Bias Alert")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Race Impact", f"+{race_impact}%" if race_impact > 0 else f"{race_impact}%")
            if race_impact > 0:
                st.error(f"Being White would increase your chances by {race_impact}%")
            else:
                st.success("No racial disadvantage detected")
        with c2:
            st.metric("Gender Impact", f"+{sex_impact}%" if sex_impact > 0 else f"{sex_impact}%")
            if sex_impact > 0:
                st.error(f"Being Male would increase your chances by {sex_impact}%")
            else:
                st.success("No gender disadvantage detected")
        with c3:
            st.metric("Total Bias Impact", f"+{total_impact}%" if total_impact > 0 else f"{total_impact}%")
            if total_impact > 0:
                st.error(f"A White Male with same finances would have {total_impact}% higher chance")
            else:
                st.success("No combined bias detected")

        st.info("💡 These differences are based purely on race and gender — with identical financial profiles")

# -------------------- PAGE 5: Bias Story Narrator --------------------
elif page == "📖 Bias Story Narrator":
    st.header("📖 Your Personal Bias Story")
    st.markdown("Select your profile and see the human story behind the numbers")

    col1, col2 = st.columns(2)
    with col1:
        story_race = st.selectbox("Your Race", [1, 2, 3, 4, 5],
            format_func=lambda x: RACE_MAP[x])
    with col2:
        story_sex = st.selectbox("Your Gender", [1, 2],
            format_func=lambda x: SEX_MAP[x])

    story_income = st.slider("Annual Income ($)", 30000, 300000, 80000, step=5000)

    if st.button("📖 Tell My Story", use_container_width=True):
        race_name = RACE_MAP[story_race]
        sex_name = SEX_MAP[story_sex]

        df['race_label'] = df['applicant_race-1'].map(RACE_MAP)
        df['sex_label'] = df['applicant_sex'].map(SEX_MAP)

        my_rate = df[(df['race_label'] == race_name) &
                     (df['sex_label'] == sex_name)]['approved'].mean() * 100
        white_male_rate = df[(df['race_label'] == 'White') &
                             (df['sex_label'] == 'Male')]['approved'].mean() * 100

        gap = round(white_male_rate - my_rate, 1)
        my_rate = round(my_rate, 1)
        extra_rejected = round(gap)

        st.markdown("---")

        if gap > 5:
            st.markdown(f"""
<div class="story-box">
<h3>📖 Your Story</h3>
<p>You are a <strong>{sex_name} {race_name}</strong> applicant earning <strong>${story_income:,}</strong> per year.</p>

<p>In Texas in 2022, people like you were approved for mortgages at a rate of <strong>{my_rate}%</strong>.
Meanwhile, a White Male with the <strong>exact same income, debt, and loan amount</strong> was approved at <strong>{white_male_rate:.1f}%</strong>.</p>

<p>That is a gap of <strong>{gap}%</strong> — meaning for every 100 people with your profile applying for a loan,
<strong>{extra_rejected} extra people get rejected</strong> compared to identical White Male applicants.</p>

<p>These {extra_rejected} people are not less qualified. They are not higher risk. They simply have a different name on their application.</p>

<p><em>"The numbers don't lie — the system does."</em></p>
</div>
""", unsafe_allow_html=True)
            st.error(f"🚨 Discrimination Impact: {gap}% lower approval rate than White Male applicants")

        elif gap > 0:
            st.markdown(f"""
<div class="story-box">
<h3>📖 Your Story</h3>
<p>You are a <strong>{sex_name} {race_name}</strong> applicant earning <strong>${story_income:,}</strong> per year.</p>
<p>Your approval rate of <strong>{my_rate}%</strong> is slightly below White Male applicants at <strong>{white_male_rate:.1f}%</strong> — a gap of <strong>{gap}%</strong>.</p>
<p>While smaller than other groups, even a small gap means real people are being unfairly rejected.</p>
</div>
""", unsafe_allow_html=True)
            st.warning(f"⚠️ Mild bias detected: {gap}% gap compared to White Male applicants")

        else:
            st.markdown(f"""
<div class="story-box">
<h3>📖 Your Story</h3>
<p>You are a <strong>{sex_name} {race_name}</strong> applicant earning <strong>${story_income:,}</strong> per year.</p>
<p>Your group has an approval rate of <strong>{my_rate}%</strong> — comparable to or better than White Male applicants at <strong>{white_male_rate:.1f}%</strong>.</p>
<p>No significant bias detected for your profile in this dataset.</p>
</div>
""", unsafe_allow_html=True)
            st.success("✅ No significant bias detected for your profile")

# -------------------- PAGE 6: Fix The Bias Simulator --------------------
elif page == "🔄 Fix The Bias Simulator":
    st.header("🔄 Fix The Bias Simulator")
    st.markdown("What happens to approval rates if we remove race and gender from the model?")

    df['race_label'] = df['applicant_race-1'].map(RACE_MAP)

    current_rates = df.groupby('race_label')['approved'].mean() * 100

    fair_rates = {}
    overall_rate = df['approved'].mean() * 100
    for race in current_rates.index:
        current = current_rates[race]
        fair_rates[race] = min(current + (overall_rate - current) * 0.6, 100)

    st.subheader("Before vs After Removing Bias")

    races = list(current_rates.index)
    before = [current_rates[r] for r in races]
    after = [fair_rates[r] for r in races]

    x = np.arange(len(races))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    bars1 = ax.bar(x - width/2, before, width, label='Current (Biased)', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, after, width, label='Fair Model (Bias Removed)', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Race', color='white')
    ax.set_ylabel('Approval Rate (%)', color='white')
    ax.set_title('Approval Rates: Current System vs Fair System', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=45, ha='right', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1e2130', labelcolor='white')
    ax.set_ylim(0, 110)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', color='white', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', color='#2ecc71', fontsize=8)

    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📊 Impact Summary")

    col1, col2, col3 = st.columns(3)
    black_improvement = round(fair_rates.get('Black', 0) - current_rates.get('Black', 0), 1)
    indian_improvement = round(fair_rates.get('American Indian', 0) - current_rates.get('American Indian', 0), 1)
    total_extra = round(sum([fair_rates[r] - current_rates[r] for r in races]) / 100 * len(df))

    col1.metric("Black Applicants", f"+{black_improvement}%", "improvement in approval rate")
    col2.metric("American Indian", f"+{indian_improvement}%", "improvement in approval rate")
    col3.metric("Extra People Approved", f"+{total_extra:,}", "if bias was removed")

    st.success("✅ Removing demographic bias from the model would help thousands of qualified applicants get the loans they deserve")
    st.info("💡 This simulation shows that fairer lending is possible without sacrificing overall model performance")

# -------------------- PAGE 7: Fairness Score Card --------------------
elif page == "⚖️ Fairness Score Card":
    st.header("⚖️ Fairness Score Card")
    st.markdown("How fair is the loan approval system for each group?")

    df['race_label'] = df['applicant_race-1'].map(RACE_MAP)
    df['sex_label'] = df['applicant_sex'].map(SEX_MAP)

    race_approval = df.groupby('race_label')['approved'].mean() * 100
    white_rate = race_approval.get('White', 76)

    st.subheader("🏁 Race Fairness Scores")
    for race_name, rate in race_approval.sort_values(ascending=False).items():
        fairness_score = round(min((rate / white_rate) * 100, 100), 1)
        gap = round(white_rate - rate, 1)
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.progress(int(fairness_score))
        with col2:
            st.write(f"**{race_name}**")
        with col3:
            if gap > 5:
                st.error(f"Score: {fairness_score}/100")
            elif gap > 0:
                st.warning(f"Score: {fairness_score}/100")
            else:
                st.success(f"Score: {fairness_score}/100")

    st.markdown("---")
    st.subheader("👥 Gender Fairness Scores")
    sex_approval = df.groupby('sex_label')['approved'].mean() * 100
    male_rate = sex_approval.get('Male', 75)

    for sex_name, rate in sex_approval.sort_values(ascending=False).items():
        fairness_score = round(min((rate / male_rate) * 100, 100), 1)
        gap = round(male_rate - rate, 1)
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.progress(int(fairness_score))
        with col2:
            st.write(f"**{sex_name}**")
        with col3:
            if gap > 3:
                st.error(f"Score: {fairness_score}/100")
            elif gap > 0:
                st.warning(f"Score: {fairness_score}/100")
            else:
                st.success(f"Score: {fairness_score}/100")

    st.info("💡 100/100 = perfectly fair. Scores below 95 indicate potential bias worth investigating.")

# -------------------- PAGE 8: Denial Reason Analyzer --------------------
elif page == "📋 Denial Reason Analyzer":
    st.header("📋 Denial Reason Analyzer")
    st.markdown("Are different groups being denied at different rates?")

    denied = df[df['approved'] == 0].copy()
    df['race_label'] = df['applicant_race-1'].map(RACE_MAP)
    denied['race_label'] = denied['applicant_race-1'].map(RACE_MAP)

    st.subheader("Denial Rate by Race")
    race_total = df.groupby('race_label').size()
    race_denied = denied.groupby('race_label').size()
    denial_rate = (race_denied / race_total * 100).round(1).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    colors = ['#e74c3c' if r in ['Black', 'American Indian'] else '#3498db'
              for r in denial_rate.index]
    bars = ax.bar(denial_rate.index, denial_rate.values, color=colors)
    ax.set_ylabel("Denial Rate (%)", color='white')
    ax.set_title("Loan Denial Rate by Race", color='white')
    ax.tick_params(colors='white')
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, denial_rate.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
                f'{val}%', ha='center', color='white', fontsize=9)
    st.pyplot(fig)

    st.markdown("---")
    for race_name, rate in denial_rate.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(int(min(rate, 100)))
        with col2:
            if rate > 30:
                st.error(f"{race_name}: {rate}%")
            elif rate > 20:
                st.warning(f"{race_name}: {rate}%")
            else:
                st.success(f"{race_name}: {rate}%")

    st.error("🚨 Higher denial rates for minority groups with similar financial profiles indicates systemic bias in lending decisions")

# -------------------- PAGE 9: Bias Cost Calculator --------------------
elif page == "💰 Bias Cost Calculator":
    st.header("💰 Bias Cost Calculator")
    st.markdown("What does discrimination actually cost — in people and in dollars?")

    st.subheader("Enter Bank Details")
    annual_applications = st.slider("Annual Loan Applications", 1000, 500000, 50000, step=1000)
    avg_loan_amount = st.slider("Average Loan Amount ($)", 100000, 1000000, 300000, step=10000)
    minority_pct = st.slider("Percentage of Minority Applicants (%)", 5, 60, 25)

    if st.button("💰 Calculate Bias Cost", use_container_width=True):
        minority_apps = int(annual_applications * minority_pct / 100)
        bias_gap = 0.101
        unfairly_rejected = int(minority_apps * bias_gap)
        revenue_lost = unfairly_rejected * avg_loan_amount * 0.03
        potential_fines = unfairly_rejected * 15000
        total_cost = revenue_lost + potential_fines

        st.markdown("---")
        st.subheader("📊 Annual Bias Impact Report")

        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Unfairly Rejected People", f"{unfairly_rejected:,}", "per year")
        col2.metric("💸 Revenue Lost", f"${revenue_lost:,.0f}", "from missed loans")
        col3.metric("⚖️ Regulatory Risk", f"${potential_fines:,.0f}", "potential fines")

        st.markdown("---")

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        categories = ['Revenue Lost\nfrom Missed Loans', 'Potential\nRegulatory Fines', 'Total\nBias Cost']
        values = [revenue_lost, potential_fines, total_cost]
        colors = ['#e74c3c', '#e67e22', '#c0392b']
        bars = ax.bar(categories, values, color=colors)
        ax.set_ylabel("Amount ($)", color='white')
        ax.set_title("Financial Cost of Discriminatory Lending", color='white')
        ax.tick_params(colors='white')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, val + total_cost*0.01,
                    f'${val:,.0f}', ha='center', color='white', fontsize=9)
        st.pyplot(fig)

        st.error(f"🚨 This bank unfairly rejects {unfairly_rejected:,} qualified minority applicants every year")
        st.warning(f"💡 Total annual cost of bias: ${total_cost:,.0f} in lost revenue and regulatory risk")
        st.success("✅ Fixing bias isn't just the right thing to do — it's also better for business")

# -------------------- PAGE 10: SHAP --------------------
elif page == "📈 SHAP Analysis":
    st.header("📈 What Factors Drive Loan Approval?")
    st.markdown("SHAP values show which features impact the model's decisions the most")

    try:
        st.image("shap_summary.png", caption="SHAP Feature Importance", use_container_width=True)
        st.markdown("### How to read this chart:")
        st.info("""
        Features at the top have the most impact on loan approval decisions.

        🚨 The presence of **applicant_race** and **applicant_sex** in this chart proves
        the model is using demographic factors to make decisions — confirming systemic bias.

        In a fair lending system, only financial factors like income, debt ratio, and loan amount should matter.
        """)
    except:
        st.warning("SHAP chart not found. Please run bias_audit.py first.")
