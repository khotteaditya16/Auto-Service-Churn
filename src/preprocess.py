import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse

def preprocess_churn_data(input_path, clean_output_path, scaled_output_path):
    print(f"📥 Loading data from: {input_path}")
    df = pd.read_csv(input_path, parse_dates=['registration_date', 'last_service_date', 'churn_date'])
    print(f"Initial data shape: {df.shape}")

    # Feature Engineering
    df['tenure_months'] = (df['customer_tenure_days'] / 30).astype(int)
    df['days_since_last_service'] = (pd.to_datetime('2021-07-01') - df['last_service_date']).dt.days

    # Days since churn (if churned)
    df['days_since_churn'] = (pd.to_datetime('2021-07-01') - df['churn_date']).dt.days
    df['days_since_churn'] = df['days_since_churn'].where(df['churned'] == 1, np.nan)

    # Encode categorical features
    df['gender_encoded'] = LabelEncoder().fit_transform(df['gender'])
    df['vehicle_type_encoded'] = LabelEncoder().fit_transform(df['vehicle_type'])

    # Drop irrelevant or text-based columns (except customer_id for reference)
    columns_to_keep = [
        'customer_id', 'age', 'avg_service_interval_days', 'total_services',
        'monthly_spend_inr', 'customer_tenure_days', 'tenure_months',
        'days_since_last_service', 'days_since_churn',
        'gender_encoded', 'vehicle_type_encoded', 'churned'
    ]
    df_cleaned = df[columns_to_keep]

    print(f"Data shape after cleaning: {df_cleaned.shape}")

    # Save cleaned data
    df_cleaned.to_csv(clean_output_path, index=False)
    print(f"💾 Saved cleaned data to: {clean_output_path}")

    # Scale numeric columns (excluding customer_id and churned)
    numeric_cols = [
        'age', 'avg_service_interval_days', 'total_services',
        'monthly_spend_inr', 'customer_tenure_days', 'tenure_months',
        'days_since_last_service', 'days_since_churn'
    ]
    
    scaler = StandardScaler()
    df_scaled = df_cleaned.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    # Save scaled data
    df_scaled.to_csv(scaled_output_path, index=False)
    print(f"💾 Saved scaled data to: {scaled_output_path}")

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'churn_data.csv')
    clean_data_path = os.path.join(base_dir, 'data', 'processed', 'churn_data_cleaned.csv')
    scaled_data_path = os.path.join(base_dir, 'data', 'processed', 'churn_data_scaled.csv')

    preprocess_churn_data(raw_data_path, clean_data_path, scaled_data_path)
