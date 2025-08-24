import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
import dagshub

dagshub.init(repo_owner='Sudip-8345', repo_name='SatisFlight-Customer-Satisfaction-Intelligence', mlflow=True)

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
def model_interpretation(model, feature_names):
    """Interpret the model using feature importance."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            with open('reports/feature_importance.png', 'wb') as f:
                plt.savefig(f)
            plt.close()
        else:
            print("Model does not have feature_importances_ attribute.")
    except Exception as e:
        raise RuntimeError(f"Failed to interpret model: {e}")
    
def local_interpretation(model, X_sample):
    """Interpret a single prediction using SHAP values."""
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_sample)

        shap.plots.waterfall(shap_values[0])
        plt.title("SHAP Waterfall Plot")
    except ImportError:
        print("SHAP library is not installed. Skipping local interpretation.")
    except Exception as e:
        raise RuntimeError(f"Failed to perform local interpretation: {e}")
def main():
    try:
        mlflow.set_tracking_uri("https://dagshub.com/Sudip-8345/SatisFlight-Customer-Satisfaction-Intelligence.mlflow")
        mlflow.set_experiment("SatisFlight_Evaluation")

        data_path = './data/featured/test_featured.csv'
        model_path = 'models/best_model.pkl'    
        metrics_path = 'reports/metrics.json'

        with mlflow.start_run():
            # Load data
            data = load_data(data_path)

            data = data.apply(pd.to_numeric, errors='raise')

            X_test = data.drop(columns=['satisfaction'])
            y_test = data['satisfaction']

            with open("models/feature_order.pkl", "rb") as f:
                feature_order = pickle.load(f)
            X_test = X_test[feature_order]

            # Load model
            model = load_model(model_path)

            # Evaluate model
            metrics = evaluate_model(model, X_test, y_test)

            # Log metrics to MLflow
            mlflow.log_metrics({
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })
            # Save metrics
            save_metrics(metrics, metrics_path)
            
            mlflow.log_artifact('reports/confusion_matrix.png')
            mlflow.log_artifact('reports/feature_importance.png')
            mlflow.log_artifact(metrics_path)

            # Model interpretation
            model_interpretation(model, X_test.columns)

            # Local interpretation on a random sample
            sample_index = np.random.choice(X_test.index, size=1)
            X_sample = X_test.loc[sample_index]
            local_interpretation(model, X_sample)

    except Exception as e:
        print(f"Error in evaluation process: {e}")

if __name__ == "__main__":
    main()