import os
import pandas as pd

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def load_file(data_path):
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return None
    
def save_file(data:pd.DataFrame, data_path:str)-> None:
    try:
        data.to_csv(data_path, index=False)
        print(f"File saved: {data_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    try:
        train_file = "train.csv"
        test_file = "test.csv"

        raw_data_path = os.path.join('data','raw')
        create_directory(raw_data_path)
        print(f"Directory created: {raw_data_path}")
        
        train_data = load_file(train_file)
        test_data = load_file(test_file)

        save_file(train_data, os.path.join(raw_data_path, "train.csv"))
        save_file(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
