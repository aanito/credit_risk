import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse
import os

def preprocess_data(input_path, output_path):
    """
    Preprocesses the dataset for modeling.
    - Encodes categorical variables.
    - Drops irrelevant columns.
    - Extracts time-based features.
    - Saves the processed data.

    Args:
        input_path (str): Path to the raw dataset.
        output_path (str): Directory to save the processed dataset.
    """
    # Load the dataset
    data = pd.read_csv(input_path)

    # Drop irrelevant columns
    irrelevant_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId']
    data.drop(columns=irrelevant_cols, inplace=True)

    # Handle categorical variables
    categorical_cols = ['CurrencyCode', 'CountryCode', 'ProductCategory']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Save encoders for future use if needed

    # Parse TransactionStartTime to extract features
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])
    data['Year'] = data['TransactionStartTime'].dt.year
    data['Month'] = data['TransactionStartTime'].dt.month
    data['Day'] = data['TransactionStartTime'].dt.day
    data['Hour'] = data['TransactionStartTime'].dt.hour
    data.drop(columns=['TransactionStartTime'], inplace=True)

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    processed_file_path = os.path.join(output_path, "processed_data.csv")
    data.to_csv(processed_file_path, index=False)
    print(f"Processed data saved at: {processed_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess data for modeling.")
    parser.add_argument('--input_path', type=str, required=True, help="Path to the input dataset.")
    parser.add_argument('--output_path', type=str, required=True, help="Directory to save the processed dataset.")
    args = parser.parse_args()

    preprocess_data(args.input_path, args.output_path)
