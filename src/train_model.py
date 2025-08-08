import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib

# Load data
df = pd.read_csv(r"E:\Projects\churn\data\processed\churn_data_cleaned.csv")

# Drop unnecessary columns
df = df.drop(columns=['customer_id', 'days_since_churn'])

# Drop missing
df = df.dropna()

# Features & target
X = df.drop("churned", axis=1)
y = df["churned"]

# Train-val-test split (60/20/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

print("Train shape:", X_train.shape, "Validation shape:", X_val.shape, "Test shape:", X_test.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Class weights for imbalance
counter = Counter(y_train)
class_weight = {0:1, 1: counter[0]/counter[1]}
print("Class weights:", class_weight)

# Logistic Regression with class weights and GridSearch for C
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
logreg = LogisticRegression(class_weight=class_weight, solver='liblinear', random_state=42)
grid = GridSearchCV(logreg, param_grid, cv=3, scoring='roc_auc')
grid.fit(X_train_scaled, y_train)

best_log = grid.best_estimator_
print("Best params:", grid.best_params_)

# Validation performance
val_preds = best_log.predict(X_val_scaled)
val_proba = best_log.predict_proba(X_val_scaled)[:, 1]
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")
print(f"Validation AUC: {roc_auc_score(y_val, val_proba):.4f}")

# Test performance
test_preds = best_log.predict(X_test_scaled)
test_proba = best_log.predict_proba(X_test_scaled)[:, 1]
print(f"\nTest Accuracy: {accuracy_score(y_test, test_preds):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, test_proba):.4f}")

print("\nClassification Report (Test):")
print(classification_report(y_test, test_preds))

# Save model & scaler
joblib.dump(best_log, 'logistic_best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\nModel and scaler saved as 'logistic_best_model.pkl' and 'scaler.pkl'")
