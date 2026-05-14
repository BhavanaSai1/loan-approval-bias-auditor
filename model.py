import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("Loading cleaned data...")
df = pd.read_csv("hmda_cleaned.csv")

# Convert non-numeric columns to numbers
df['debt_to_income_ratio'] = pd.to_numeric(df['debt_to_income_ratio'], errors='coerce')
df['loan_to_value_ratio'] = pd.to_numeric(df['loan_to_value_ratio'], errors='coerce')
df['property_value'] = pd.to_numeric(df['property_value'], errors='coerce')
df['applicant_age'] = pd.to_numeric(df['applicant_age'], errors='coerce')

# Fill missing values with median
df = df.fillna(df.median(numeric_only=True))

# Separate features and target
X = df.drop('approved', axis=1)
y = df['approved']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Logistic Regression model...")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("Logistic Regression Accuracy:", round(lr_accuracy * 100, 2), "%")

print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("Random Forest Accuracy:", round(rf_accuracy * 100, 2), "%")

print("\nDetailed Report for Random Forest:")
print(classification_report(y_test, rf_pred))

# Save the best model
pickle.dump(rf_model, open("model.pkl", "wb"))
pickle.dump(X_train.columns.tolist(), open("columns.pkl", "wb"))
print("\nModel saved!")