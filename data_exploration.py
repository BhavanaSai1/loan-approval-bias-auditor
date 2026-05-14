import pandas as pd

# Load the data
df = pd.read_csv("hmda_data.csv", low_memory=False)

# See how many rows and columns we have
print("Shape:", df.shape)

# See the first 5 rows
print(df.head())

# See all column names
print(df.columns.tolist())

# See how many approvals vs denials
print(df['action_taken'].value_counts())