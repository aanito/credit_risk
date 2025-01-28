import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from xverse.transformer import WOE

# Define functions for Default Estimation and WoE Binning

def calculate_rfms(data):
    """Calculate RFMS (Recency, Frequency, Monetary, Stability) metrics for each customer."""
    print("\n--- Calculating RFMS metrics ---\n")
    rfms = data.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (pd.Timestamp.now() - pd.to_datetime(x)).min().days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum'),
        Stability=('Amount', 'std')
    ).reset_index()
    rfms['Stability'] = rfms['Stability'].fillna(0)  # Handle cases with a single transaction
    print("RFMS metrics calculated successfully.")
    return rfms

def classify_users_rfms(rfms):
    """Classify users into good (high RFMS score) and bad (low RFMS score) categories."""
    print("\n--- Classifying Users into Good and Bad ---\n")
    rfms['RFMS_Score'] = (rfms['Recency'] * -1) + (rfms['Frequency'] * 2) + rfms['Monetary'] - rfms['Stability']
    median_score = rfms['RFMS_Score'].median()
    rfms['DefaultLabel'] = np.where(rfms['RFMS_Score'] >= median_score, 'Good', 'Bad')
    print("Users classified successfully.")
    return rfms

def plot_rfms_distribution(rfms):
    """Visualize the RFMS score distribution and classifications."""
    print("\n--- Visualizing RFMS Score Distribution ---\n")
    plt.figure(figsize=(10, 6))
    sns.histplot(rfms['RFMS_Score'], kde=True, bins=30, color='blue', label='RFMS Score')
    plt.axvline(rfms['RFMS_Score'].median(), color='red', linestyle='--', label='Median')
    plt.title('RFMS Score Distribution')
    plt.xlabel('RFMS Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def perform_woe_binning(data, target_col):
    """Perform Weight of Evidence (WoE) binning on the dataset."""
    print("\n--- Performing WoE Binning ---\n")
    woe_transformer = WOE()
    woe_transformer.fit(data, data[target_col])
    woe_data = woe_transformer.transform(data)
    print("WoE binning completed successfully.")
    return woe_data

def main(file_path, output_path):
    """Main function to execute Default Estimator and WoE Binning pipeline."""
    data = pd.read_csv(file_path)

    # Ensure datetime format for TransactionStartTime
    data['TransactionStartTime'] = pd.to_datetime(data['TransactionStartTime'])

    # Calculate RFMS metrics
    rfms = calculate_rfms(data)

    # Classify users into Good and Bad
    rfms = classify_users_rfms(rfms)

    # Visualize RFMS Distribution
    plot_rfms_distribution(rfms)

    # Perform WoE Binning
    woe_data = perform_woe_binning(rfms, target_col='DefaultLabel')

    # Save the results
    try:
        woe_data.to_csv(output_path, index=False)
        print(f"Processed data saved successfully at {output_path}.")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Default Estimator and WoE Binning Pipeline")
    parser.add_argument("file_path", type=str, help="Path to the input dataset")
    parser.add_argument("output_path", type=str, help="Path to save the processed dataset")
    args = parser.parse_args()

    main(args.file_path, args.output_path)
