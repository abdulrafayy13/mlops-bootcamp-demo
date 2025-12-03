import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
from datetime import datetime

def train_model():
    """Train a simple classifier and save it"""
    
    # Load data
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save model
    joblib.dump(model, '../models/iris_model.pkl')
    
    # Save metrics
    metrics = {
        'accuracy': float(accuracy),
        'timestamp': datetime.now().isoformat(),
        'model_type': 'RandomForestClassifier'
    }
    
    with open('../models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model trained! Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    
    return model, metrics

if __name__ == '__main__':
    train_model()
