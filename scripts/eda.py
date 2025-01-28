import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define functions for EDA

def load_data(file_path):
    """Load the dataset from the given file path."""
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def overview_of_data(data):
    """Print the structure and basic information of the dataset."""
    print("\n--- Overview of the Data ---\n")
    print(data.info())
    print("\nFirst 5 rows of the dataset:\n", data.head())
    print("\nShape of the dataset: ", data.shape)

def summary_statistics(data):
    """Display summary statistics for numerical columns."""
    print("\n--- Summary Statistics ---\n")
    print(data.describe())

def visualize_numerical_distributions(data, numerical_cols):
    """Visualize the distribution of numerical features."""
    print("\n--- Visualizing Numerical Features ---\n")
    for col in numerical_cols:
        sns.histplot(data[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

def visualize_categorical_distributions(data, categorical_cols):
    """Visualize the distribution of categorical features."""
    print("\n--- Visualizing Categorical Features ---\n")
    for col in categorical_cols:
        sns.countplot(x=col, data=data, order=data[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.show()

def correlation_analysis(data, numerical_cols):
    """Display the correlation matrix of numerical features."""
    print("\n--- Correlation Analysis ---\n")
    correlation_matrix = data[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def identify_missing_values(data):
    """Identify and display missing values."""
    print("\n--- Identifying Missing Values ---\n")
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])
    return missing_values

def detect_outliers(data, numerical_cols):
    """Detect outliers using box plots."""
    print("\n--- Outlier Detection ---\n")
    for col in numerical_cols:
        sns.boxplot(x=data[col])
        plt.title(f"Outliers in {col}")
        plt.xlabel(col)
        plt.show()

if __name__ == "__main__":
    # File path to the dataset
    file_path = "data/raw/data.csv"  # Update this path to your dataset

    # Load the dataset
    data = load_data(file_path)

    # EDA Tasks
    overview_of_data(data)
    summary_statistics(data)

    # Identify numerical and categorical columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    # Visualize distributions
    visualize_numerical_distributions(data, numerical_cols)
    visualize_categorical_distributions(data, categorical_cols)

    # Correlation analysis
    correlation_analysis(data, numerical_cols)

    # Missing values analysis
    missing_values = identify_missing_values(data)

    # Outlier detection
    detect_outliers(data, numerical_cols)
