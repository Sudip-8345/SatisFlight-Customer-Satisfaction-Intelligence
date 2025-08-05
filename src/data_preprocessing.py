import pandas as pd
import os

def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"{file_path} is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")

def cleaning_data(df: pd.DataFrame) -> pd.DataFrame:
    # Example cleaning: drop rows with any NaN values
    df = df.drop(['Unnamed: 0','id'],axis=1)
    df = df.drop_duplicates()

    ### Handle Null values of Arrival Delay in Minutes
    df['Arrival Delay in Minutes'] = \
        df['Arrival Delay in Minutes'].fillna(df['Arrival Delay in Minutes'].median())
    return df

def handle_outliers(df: pd.DataFrame, columns) -> pd.DataFrame:
    try:
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

        return df
    except KeyError as e:
        raise KeyError(f"Column not found in DataFrame: {e}")
    
def create_directory(path):
    """Create a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)

def save_cleaned_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        df.to_csv(file_path, index=False)
        print(f"Cleaned data saved to {file_path}")
    except Exception as e:
        raise IOError(f"Error saving cleaned data: {e}")
    
def main():
    try:
        raw_data_path = os.path.join('data', 'raw')
        preprocessed_data_path = os.path.join('data', 'processed')
        train_file = os.path.join(raw_data_path, 'train.csv')
        test_file = os.path.join(raw_data_path, 'test.csv')

        # Load data
        train_data = load_csv(train_file)
        test_data = load_csv(test_file)

        # Clean data
        train_data = cleaning_data(train_data)
        test_data = cleaning_data(test_data)
        # Handle outliers
        columns_to_handle = ['Flight Distance','Departure Delay in Minutes',\
                             'Arrival Delay in Minutes', 'Checkin service']
        train_data = handle_outliers(train_data, columns_to_handle)
        test_data = handle_outliers(test_data, columns_to_handle)

        # Create directory for preprocessed data
        create_directory(preprocessed_data_path)

        # Save cleaned data
        save_file_path_train = os.path.join(preprocessed_data_path, 'train_cleaned.csv')
        save_file_path_test = os.path.join(preprocessed_data_path, 'test_cleaned.csv')

        save_cleaned_data(train_data, save_file_path_train)
        save_cleaned_data(test_data, save_file_path_test)

    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == "__main__":
    main()