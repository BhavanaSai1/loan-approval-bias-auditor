import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt

print("Loading data and model...")
df = pd.read_csv("hmda_cleaned.csv")

# Same preprocessing as before
df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
df['loan_to_value_ratio'] = pd.to_numeric(df['loan_to_value_ratio'], errors='coerce')
df['property_value'] = pd.to_numeric(df['property_value'], errors='coerce')
df['applicant_age'] = pd.to_numeric(df['applicant_age'], errors='coerce')
df = df.fillna(df.median(numeric_only=True))

# Load saved model
model = pickle.load(open("model.pkl", "rb"))

X = df.drop('approved', axis=1)
y = df['approved']

# Use a sample for SHAP (full data takes too long)
X_sample = X.sample(1000, random_state=42)

print("Running SHAP analysis... this takes 2-3 minutes")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Plot 1 - Feature Importance
print("Saving feature importance chart...")
shap.summary_plot(
    shap_values[:,:,1],
    X_sample,
    show=False
)
plt.title("What Factors Drive Loan Approval?")
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.close()

# Bias Analysis by Race
print("\nApproval Rate by Race:")
race_map = {
    1: 'American Indian',
    2: 'Asian',
    3: 'Black',
    4: 'Pacific Islander',
    5: 'White',
    6: 'Not Provided',
    7: 'Not Applicable'
}
df['race_label'] = df['applicant_race-1'].map(race_map)
race_approval = df.groupby('race_label')['approved'].mean().round(3) * 100
print(race_approval.sort_values())

# Bias Analysis by Gender
print("\nApproval Rate by Gender:")
sex_map = {1: 'Male', 2: 'Female', 3: 'Not Provided', 4: 'Not Applicable'}
df['sex_label'] = df['applicant_sex'].map(sex_map)
sex_approval = df.groupby('sex_label')['approved'].mean().round(3) * 100
print(sex_approval)

# Save bias results
race_approval.to_csv("race_bias.csv")
sex_approval.to_csv("gender_bias.csv")

print("\nBias audit complete!")
print("Charts saved as shap_summary.png")