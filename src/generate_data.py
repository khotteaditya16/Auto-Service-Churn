import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import os

def generate_auto_churn_data(num_customers=1000, base_date='2021-07-01', output_path='data/raw/churn_data.csv'):
    fake = Faker('en_IN')
    base_date = datetime.strptime(base_date, '%Y-%m-%d')
    np.random.seed(42)

    customer_ids = [f'CUST{i:05d}' for i in range(1, num_customers + 1)]
    genders = np.random.choice(['Male', 'Female'], size=num_customers, p=[0.7, 0.3])
    gender_encoded = [1 if g == 'Male' else 0 for g in genders]
    ages = np.random.randint(20, 65, size=num_customers)
    vehicle_types = np.random.choice(['Two-wheeler', 'Four-wheeler'], size=num_customers, p=[0.6, 0.4])
    vehicle_type_encoded = [0 if v == 'Two-wheeler' else 1 for v in vehicle_types]

    reg_start = datetime(2016, 1, 1)
    reg_end = base_date - timedelta(days=30)
    registration_dates = [datetime.combine(fake.date_between_dates(date_start=reg_start, date_end=reg_end), datetime.min.time()) for _ in range(num_customers)]

    total_services = []
    last_service_dates = []
    avg_service_intervals = []

    for reg_date in registration_dates:
        tenure_days = (base_date - reg_date).days
        avg_interval = np.random.randint(30, 180)
        n_services = max(1, int(tenure_days / avg_interval + np.random.normal(0, 1)))
        total_services.append(n_services)
        avg_service_intervals.append(avg_interval)
        last_service_date = reg_date + timedelta(days=avg_interval * n_services)
        if last_service_date > base_date:
            last_service_date = base_date - timedelta(days=np.random.randint(1, 30))
        last_service_dates.append(last_service_date)

    monthly_spend = []
    for vt in vehicle_types:
        spend = np.random.normal(1500 if vt == 'Two-wheeler' else 3500, 400)
        monthly_spend.append(round(max(300, spend), 2))

    city = ['Pune'] * num_customers
    customer_tenure_days = [(last - reg).days for last, reg in zip(last_service_dates, registration_dates)]

    churn_prob = []
    churned = []
    churn_dates = []

    for i in range(num_customers):
        p = 0.1
        if total_services[i] < 5:
            p += 0.2
        if avg_service_intervals[i] > 90:
            p += 0.15
        if monthly_spend[i] < 1000:
            p += 0.2
        days_since_last = (base_date - last_service_dates[i]).days
        if days_since_last > 180:
            p += 0.2
        if ages[i] < 25:
            p += 0.05
        if vehicle_type_encoded[i] == 0:  # two-wheeler
            p += 0.05

        p = min(p, 0.95)
        churn_prob.append(p)

        is_churn = np.random.rand() < p
        churned.append(int(is_churn))

        if is_churn:
            churn_start = last_service_dates[i] + timedelta(days=1)
            if churn_start >= base_date:
                churn_date = base_date - timedelta(days=np.random.randint(1, 10))
            else:
                churn_date = fake.date_between_dates(date_start=churn_start, date_end=base_date)
                churn_date = datetime.combine(churn_date, datetime.min.time())
            churn_dates.append(churn_date)
        else:
            churn_dates.append(None)

    df = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': genders,
        'age': ages,
        'gender_encoded': gender_encoded,
        'vehicle_type': vehicle_types,
        'vehicle_type_encoded': vehicle_type_encoded,
        'registration_date': registration_dates,
        'last_service_date': last_service_dates,
        'avg_service_interval_days': avg_service_intervals,
        'total_services': total_services,
        'monthly_spend_inr': monthly_spend,
        'city': city,
        'customer_tenure_days': customer_tenure_days,
        'churned': churned,
        'churn_date': churn_dates
    })

    # Add additional features for modeling
    df['tenure_months'] = df['customer_tenure_days'] // 30
    df['days_since_last_service'] = (base_date - df['last_service_date']).dt.days
    df['service_frequency'] = df['total_services'] / (df['customer_tenure_days'] / 30 + 1e-5)
    df['spend_per_service'] = df['monthly_spend_inr'] / (df['total_services'] + 1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Churn data generated and saved to: {output_path}")
    return df


if __name__ == '__main__':
    df = generate_auto_churn_data(num_customers=1000)
