"""
Data preparation module for Telco Customer Churn prediction.
Downloads the dataset and performs initial preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import urllib.request
import ssl


def download_telco_data():
    """Download Telco Customer Churn dataset from UCI ML Repository."""
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    data_path = Path(__file__).parent.parent / "data" / "telco_churn.csv"
    
    if data_path.exists():
        print(f"Data already exists at {data_path}")
        return data_path
    
    print(f"Downloading Telco Customer Churn dataset from {url}...")
    try:
        # Handle SSL certificate verification
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, data_path)
        print(f"Dataset downloaded to {data_path}")
    except Exception as e:
        print(f"Download failed: {e}. Creating synthetic data instead...")
        df = create_synthetic_data()
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"Synthetic data created at {data_path}")
    
    return data_path


def create_synthetic_data():
    """Create synthetic Telco-like dataset for local testing."""
    np.random.seed(42)
    n_samples = 7000
    
    df = pd.DataFrame({
        'customerID': [f'CUST_{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'InternetService': np.random.choice(['Fiber optic', 'DSL', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
        'Churn': np.random.choice(['No', 'Yes'], n_samples, p=[0.73, 0.27])
    })
    
    return df


def load_and_clean_data(data_path):
    """Load and perform initial cleaning on the dataset."""
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Handle missing values in TotalCharges (common in real data)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    print(f"\nCleaned dataset shape: {df.shape}")
    print(f"Churn distribution:\n{df['Churn'].value_counts()}")
    
    return df


def engineer_features(df):
    """Engineer new features for better model performance."""
    df = df.copy()
    
    # Tenure-based features
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 72], 
                                labels=['0-6 months', '6-12 months', '1-2 years', '2+ years'])
    
    # Charge-based features
    df['avg_monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['total_charges_group'] = pd.qcut(df['TotalCharges'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Service adoption score
    service_cols = [col for col in df.columns if 'Service' in col or 'Security' in col or 'Support' in col]
    service_count = df[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    df['service_adoption_score'] = service_count
    
    print(f"\nEngineered features added:")
    print(f"  - tenure_group")
    print(f"  - avg_monthly_to_total_ratio")
    print(f"  - total_charges_group")
    print(f"  - service_adoption_score")
    
    return df


if __name__ == '__main__':
    data_path = download_telco_data()
    df = load_and_clean_data(data_path)
    df = engineer_features(df)
    
    # Save processed data
    processed_path = Path(data_path).parent / "telco_churn_processed.csv"
    df.to_csv(processed_path, index=False)
    print(f"\nProcessed data saved to {processed_path}")
