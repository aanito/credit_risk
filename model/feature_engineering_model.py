import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler
from xverse.transformer import WOE
import argparse

# Define functions for feature engineering

def load_data(file_path):
    """Load the dataset from the given file path."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def create_aggregate_features(data):
    """Create aggregate features for each customer."""
    print("\n--- Creating Aggregate Features ---\n")
    aggregate_features = data.groupby('CustomerId').agg(
        TotalTransactionAmount=('Amount', 'sum'),
        AverageTransactionAmount=('Amount', 'mean'),
        TransactionCount=('TransactionId', 'count'),
        StdTransactionAmount=('Amount', 'std')
    ).reset_index()
    print("Aggregate features created successfully.")
    return aggregate_features


def extract_time_features(data):
    """Extract time-based features from the transaction start time."""
    print("\n--- Extracting Time Features ---\n")
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['TransactionHour'] = data['TransactionStartTime'].dt.hour
    data['TransactionDay'] = data['TransactionStartTime'].dt.day
    data['TransactionMonth'] = data['TransactionStartTime'].dt.month
    data['TransactionYear'] = data['TransactionStartTime'].dt.year
    print("Time features extracted successfully.")
    return data


def encode_categorical_features(data):
    """Encode categorical features using One-Hot and Label Encoding."""
    print("\n--- Encoding Categorical Features ---\n")
    categorical_cols = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_cols]),
                                columns=encoder.get_feature_names_out(categorical_cols))
    data = pd.concat([data, encoded_data], axis=1).drop(columns=categorical_cols)
    print("Categorical features encoded successfully.")
    return data


def handle_missing_values(data):
    """Handle missing values using mean imputation."""
    print("\n--- Handling Missing Values ---\n")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])
    data.fillna(data.mean(), inplace=True)
    print("Missing values handled successfully.")
    return data


def normalize_features(data):
    """Normalize numerical features to [0, 1] range."""
    print("\n--- Normalizing Numerical Features ---\n")
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    print("Normalization completed successfully.")
    return data


def calculate_woe_iv(data, target_col):
    """Calculate Weight of Evidence (WOE) and Information Value (IV) for features."""
    print("\n--- Calculating WOE and IV ---\n")
    woe_transformer = WOE()
    woe_transformer.fit(data, data[target_col])
    transformed_data = woe_transformer.transform(data)
    print("WOE and IV calculation completed successfully.")
    return transformed_data


def save_data(data, output_path):
    """Save the processed data to the given output path."""
    try:
        data.to_csv(output_path, index=False)
        print(f"Data saved successfully at {output_path}.")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise


def main(file_path, output_path):
    """Main function to execute the feature engineering pipeline."""
    data = load_data(file_path)
    data = handle_missing_values(data)
    data = extract_time_features(data)
    data = encode_categorical_features(data)
    data = normalize_features(data)
    data = calculate_woe_iv(data, target_col='FraudResult')

    aggregate_features = create_aggregate_features(data)
    save_data(aggregate_features, output_path)
    print("Feature engineering pipeline executed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Feature Engineering Pipeline")
    parser.add_argument("file_path", type=str, help="Path to the input dataset")
    parser.add_argument("output_path", type=str, help="Path to save the processed dataset")
    args = parser.parse_args()

    main(args.file_path, args.output_path)
