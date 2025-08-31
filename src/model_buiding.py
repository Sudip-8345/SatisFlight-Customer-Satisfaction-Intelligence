import os
import pickle
import yaml
import pandas as pd
# from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import mlflow
# import dagshub
# dagshub.init(repo_owner='Sudip-8345', repo_name='SatisFlight-Customer-Satisfaction-Intelligence', mlflow=True)

dagshub_token = os.getenv("MLFLOW_TRACKING_PASSWORD") 
if dagshub_token:
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "Sudip-8345")
    os.environ["MLFLOW_TRACKING_URI"] = os.getenv(
        "MLFLOW_TRACKING_URI",
        "https://dagshub.com/Sudip-8345/SatisFlight-Customer-Satisfaction-Intelligence.mlflow"
    )


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return pd.read_csv(file_path)

def load_params(params_path: str) -> dict:
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Params file not found: {params_path}")
    with open(params_path, 'r') as file:
        return yaml.safe_load(file)

def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict) -> XGBClassifier:
    try:
        mlflow.set_experiment("SatisFlight_using_XGBoost")
        with mlflow.start_run():
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            mlflow.log_params(params)
            return model
    except Exception as e:
        raise RuntimeError(f"Error training model: {e}")
    
def save_model(model: XGBClassifier, model_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {model_path}")
    except Exception as e:
        raise IOError(f"Error saving model: {e}")

def main():
    try:
        params_path = 'params.yaml'
        model_path = 'models/best_model.pkl'
        data_path = './data/featured/train_featured.csv'

        # Load parameters
        params_dict = load_params(params_path)
        params = params_dict.get('model_building', {})

        # Load data
        data = load_data(data_path)
        data = data.apply(pd.to_numeric, errors='raise')

        X = data.drop(columns=['satisfaction'])
        y = data['satisfaction']

        # Train model
        model = train_model(X, y, params)

        # Save model
        save_model(model, model_path)

        with open("models/feature_order.pkl", "wb") as f:
            pickle.dump(list(X.columns), f)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
