import pandas as pd

print("Loading data... this may take a minute")

# Load the data
df = pd.read_csv("hmda_data.csv", low_memory=False)

print("Data loaded!")

# Keep only approved (1) and denied (3)
df = df[df['action_taken'].isin([1, 3])]

# Create target column: 1 = Approved, 0 = Denied
df['approved'] = (df['action_taken'] == 1).astype(int)

# Keep only the columns we need
columns_to_keep = [
    'approved',
    'loan_amount',
    'income',
    'debt_to_income_ratio',
    'loan_to_value_ratio',
    'loan_purpose',
    'applicant_race-1',
    'applicant_sex',
    'applicant_age',
    'property_value',
    'occupancy_type'
]

df = df[columns_to_keep]

# Drop rows where important columns are missing
df = df.dropna(subset=['income', 'loan_amount', 'approved'])

# Save cleaned data
df.to_csv("hmda_cleaned.csv", index=False)

print("Cleaned data saved!")
print("Shape after cleaning:", df.shape)
print("Approval rate:", round(df['approved'].mean() * 100, 2), "%")
print("\nApproved vs Denied:")
print(df['approved'].value_counts())