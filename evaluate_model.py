# evaluate_model.py
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using various metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    # Print results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)
    
    return {
        'accuracy': accuracy,
        'auc_score': auc_score,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }
