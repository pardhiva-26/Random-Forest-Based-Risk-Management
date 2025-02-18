---

# Random Forest-Based Risk Management

This project provides a framework for assessing financial risk (e.g., credit, liquidity, or market risk) using predictive analytics. It leverages machine learning to identify high-risk cases, helping decision-makers manage risk effectively.

## Features
- **Data Preprocessing**: Cleans data, handles missing values, and encodes categorical variables.
- **Model Training**: Trains a Random Forest Classifier for predictive modeling.
- **Risk Assessment**: Provides functions to assess risk for new applications.
- **Feature Importance**: Visualizes the most significant features affecting the model.

## Usage

1. **Preprocess and Load Data**: Use `data_preprocessing.py` to clean and split data.
2. **Train the Model**: Run `model_training.py` to train the risk model.
3. **Evaluate the Model**: Run `evaluate_model.py` to get classification reports and accuracy metrics.
4. **Plot Feature Importance**: Use `feature_importance_plot.py` to visualize key features.
5. **Assess Risk**: Run `assess_risk.py` to evaluate risk levels for new applications.

### Example

```python
from data_preprocessing import load_and_clean_data, preprocess_data
from model_training import train_model
from assess_risk import assess_credit_risk

# Load and preprocess data
data = load_and_clean_data('data/credit_risk_dataset.csv')
X_train, X_test, y_train, y_test, feature_columns, scaler = preprocess_data(data)

# Train the model
model = train_model(X_train, y_train)

# Assess risk for a new application
new_application = pd.DataFrame({...})
risk_label, risk_probability = assess_credit_risk(new_application, model, scaler, feature_columns)
print(f"Risk Level: {risk_label}, Probability: {risk_probability}")
```

## Project Structure
- `data_preprocessing.ipynb`: Data cleaning and preprocessing.
- `model_training.ipynb`: Model training setup.
- `evaluate_model.ipynb`: Evaluation metrics and confusion matrix.
- `feature_importance_plot.ipynb`: Visualizes feature importance.
- `assess_risk.ipynb`: Function to assess credit risk for new applications.
