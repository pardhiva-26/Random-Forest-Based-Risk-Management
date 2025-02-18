# assess_risk.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def assess_credit_risk(new_application, model, scaler, feature_columns):
    """
    Assess credit risk for a new application
    """
    # Create a copy of the new application
    application = new_application.copy()
    
    # Identify categorical columns
    categorical_columns = application.select_dtypes(include=['object']).columns
    
    # Encode categorical variables
    for column in categorical_columns:
        le = LabelEncoder()
        # Fit the encoder on the column values we have
        le.fit(application[column])
        # Transform the column
        application[column] = le.transform(application[column])
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in application.columns:
            application[col] = 0
            
    # Reorder columns to match feature_columns
    application = application[feature_columns]
    
    # Scale the features
    application_scaled = scaler.transform(application)
    
    # Get prediction probability
    risk_probability = model.predict_proba(application_scaled)[0][1]
    risk_label = "High Risk" if risk_probability > 0.5 else "Low Risk"
    
    return risk_label, risk_probability
