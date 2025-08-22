import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load dataset from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)

def load_model(model_path):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except pickle.UnpicklingError:
        raise ValueError("Model file is corrupted or incompatible.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return various performance metrics."""
    try:
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)

        print("Classification Report:", class_report)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')   
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        with open('reports/confusion_matrix.png', 'wb') as f:
            plt.savefig(f)
        plt.close()

        return metrics
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate model: {e}")
    
def save_metrics(metrics, save_path: str) -> None:
    """Save evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {save_path}")
    except Exception as e:
        raise IOError(f"Error saving metrics: {e}")
    
def main():
    try:
        model_path = 'models/best_model.pkl'
        test_data_path = './data/featured/test_featured.csv'
        metrics_save_path = 'reports/metrics.json'

        # Load the trained model
        model = load_model(model_path)

        # Load test data
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test data file not found: {test_data_path}")
        test_data = load_data(test_data_path)
        with open("models/feature_order.pkl", "rb") as f:
            feature_order = pickle.load(f)
        test_data = test_data[feature_order + ['satisfaction']]
        
        test_data = test_data.apply(pd.to_numeric, errors='raise')

        X_test = test_data.drop(columns=['satisfaction'])
        y_test = test_data['satisfaction']

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Save the evaluation metrics
        save_metrics(metrics, metrics_save_path)

    except Exception as e:
        print(f"Error in evaluation process: {e}")

if __name__ == "__main__":
    main()