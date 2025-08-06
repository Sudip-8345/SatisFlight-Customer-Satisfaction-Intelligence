import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from  sklearn.feature_selection import mutual_info_classif
import pickle

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"{file_path} is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
def feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['total_delay'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']
        df['delay_ratio'] = df['total_delay'] / (df['Flight Distance']+1)
        df['age_group'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 100], labels=['teenage', 'youngster', 'adult', 'senior'])

        return df
    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        cols_to_encode = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction', 'age_group']

        label_mappings = {}

        for col in cols_to_encode:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        pickle.dump(label_mappings, open('models/label_mappings.pkl', 'wb'))
        # print(label_mappings)
        print("Label mappings saved to label_mappings.pkl")
        return df

    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")

def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    try:
        mutual_info = mutual_info_classif(
            df.drop(columns=['satisfaction']), 
            df['satisfaction'], discrete_features=True
        )
        mutual_info_df = pd.DataFrame({
            'Feature': df.drop(columns=['satisfaction']).columns,
            'Mutual Info': mutual_info
        }).sort_values(by='Mutual Info', ascending=False)

        top_features = mutual_info_df.head(15)['Feature'].tolist()
        final_df = df[top_features + ['satisfaction']]
        return final_df
    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")
    
def create_directory(path):
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)

def save_featured_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        df.to_csv(file_path, index=False)
        print(f"Featured data saved to {file_path}")
    except Exception as e:
        raise IOError(f"Error saving featured data: {e}")
    
def main():
    try:
        processed_data_path = os.path.join('data', 'processed')
        featured_data_path = os.path.join('data', 'featured')
        train_processed = os.path.join(processed_data_path, 'train_cleaned.csv')
        test_processed = os.path.join(processed_data_path, 'test_cleaned.csv')

        # Load data
        train_data = load_csv(train_processed)
        test_data = load_csv(test_processed)

        # Feature creation
        train_data = feature_creation(train_data)
        test_data = feature_creation(test_data)

        # Encode categorical features
        train_data = encode_categorical_features(train_data)
        test_data = encode_categorical_features(test_data)

        # Feature selection
        train_data = feature_selection(train_data)
        test_data = feature_selection(test_data)

        # Create directory for featured data
        create_directory(featured_data_path)

        # Save featured data
        save_file_path_train = os.path.join(featured_data_path, 'train_featured.csv')
        save_file_path_test = os.path.join(featured_data_path, 'test_featured.csv')

        save_featured_data(train_data, save_file_path_train)
        save_featured_data(test_data, save_file_path_test)
        
        with open('models/label_mappings.pkl', 'rb') as f:
            label_mappings = pickle.load(f)
            print(f"Label mappings loaded: {label_mappings}")    

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
