import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
def load_and_clean_data(file_path):
    """
    Load and clean the credit risk dataset
    """
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = data.dropna()
    
    # Remove duplicates
    data = data.drop_duplicates()
    
    return data
def preprocess_data(data):
    """
    Preprocess the data for model training
    """
    # Create a copy of the data
    df = data.copy()
    
    # Identify numeric and categorical columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Encode categorical variables
    label_encoders = {}
    for column in categorical_columns:
        if column != 'loan_status':  # Don't encode the target variable yet
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
    
    # Encode target variable
    label_encoders['loan_status'] = LabelEncoder()
    df['loan_status'] = label_encoders['loan_status'].fit_transform(df['loan_status'])
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'loan_status']
    X = df[feature_columns]
    y = df['loan_status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns, scaler