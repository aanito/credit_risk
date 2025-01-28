Task 1: Understanding Credit Risk
Research Credit Risk Concepts

Review the provided references to understand:
Key definitions: default, creditworthiness, Basel II compliance.
Risk probability scoring and optimal loan determination.
Techniques for building alternative credit scoring models.
Define the Proxy Variable for Risk

Use FraudResult as a starting point to differentiate "bad" (high-risk) vs. "good" (low-risk) customers.
Augment this with additional derived metrics like overdue transactions or excessively high amounts spent relative to income.
Prepare for Feature Selection

Identify observable features in the dataset that are likely correlated with default behavior, such as transaction frequency, average transaction amount, and product category.
Task 2: Exploratory Data Analysis (EDA)
Data Overview
Inspect structure: Use Python/Pandas to review the dataset (e.g., .info(), .describe()).
Summarize key stats: Mean, median, standard deviation, and null counts for all fields.
Analyze Numerical Features
Distribution plots: Histograms, boxplots, and KDE plots for Amount, Value, and TransactionStartTime.
Correlation analysis: Pearson/Spearman correlation to find relationships between numerical variables.
Analyze Categorical Features
Value counts: Check the frequency of categories in CountryCode, ProductCategory, PricingStrategy, etc.
Cross-tabulations: Analyze relationships between FraudResult and categorical features.
Handle Missing Values
Identify missing values via .isnull().sum() and evaluate

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

if **name** == "**main**": # File path to the dataset
file_path = "data/raw/data.csv" # Update this path to your dataset

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

Task 3: Model Development

1. Risk Probability Model
   Algorithm selection: Start with logistic regression, decision trees, or gradient boosting (e.g., XGBoost, LightGBM).
   Target variable: Use the proxy variable for default (FraudResult).
   Input features: Include significant predictors identified during EDA (e.g., transaction frequency, product category, amounts).
2. Credit Scoring
   Convert risk probabilities to credit scores using a scorecard approach:
   Normalize probabilities to a 300–850 range.
   Use thresholds (e.g., low-risk: 700+, moderate: 500–699, high-risk: below 500).
3. Optimal Loan Prediction
   Build a regression model to predict the optimal loan amount and duration.
   Input features: Customer historical spending patterns, income proxies, product categories.
   Target variables: Derived loan size and duration from risk scores.

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

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Run Feature Engineering Pipeline")
parser.add_argument("--file_path", type=str, required=True, help="Path to the input dataset")
parser.add_argument("--output_path", type=str, required=True, help="Path to save the processed dataset")
args = parser.parse_args()

    main(args.file_path, args.output_path)





Task 4: Modelling

Preprocessing data:

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import argparse
import os

def preprocess_data(input_path, output_path):
"""
Preprocesses the dataset for modeling. - Encodes categorical variables. - Drops irrelevant columns. - Extracts time-based features. - Saves the processed data.

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

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Preprocess data for modeling.")
parser.add_argument('--input_path', type=str, required=True, help="Path to the input dataset.")
parser.add_argument('--output_path', type=str, required=True, help="Directory to save the processed dataset.")
args = parser.parse_args()

    preprocess_data(args.input_path, args.output_path

Concepts Applied
Model Selection and Training: Selecting models suitable for classification problems like logistic regression, decision trees, random forest, and gradient boosting machines.
Data Splitting: Dividing data into training and testing sets to ensure models are tested on unseen data.
Hyperparameter Tuning: Improving model performance by searching for the best hyperparameters using grid search and random search.
Evaluation Metrics:
Accuracy
Precision
Recall (Sensitivity)
F1 Score
ROC-AUC Curve
Python Scripts

1. Splitting the Data
   python
   Copy
   Edit
   from sklearn.model_selection import train_test_split

def split_data(data, target_col):
"""Split the data into training and testing sets."""
X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Data split completed: {X_train.shape[0]} train rows, {X_test.shape[0]} test rows.")
return X_train, X_test, y_train, y_test 2. Model Training
python
Copy
Edit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def train_models(X_train, y_train):
"""Train selected models on training data."""
models = {
"Logistic Regression": LogisticRegression(random_state=42),
"Decision Tree": DecisionTreeClassifier(random_state=42),
"Random Forest": RandomForestClassifier(random_state=42),
"Gradient Boosting": GradientBoostingClassifier(random_state=42)
}
trained_models = {}
for name, model in models.items():
model.fit(X_train, y_train)
trained_models[name] = model
print(f"Trained {name}.")
return trained_models 3. Hyperparameter Tuning
python
Copy
Edit
from sklearn.model_selection import GridSearchCV

def hyperparameter*tuning(model, param_grid, X_train, y_train):
"""Perform hyperparameter tuning using GridSearchCV."""
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params*}")
return grid*search.best_estimator*
Example for Random Forest:

python
Copy
Edit
rf_param_grid = {
'n_estimators': [100, 200, 300],
'max_depth': [5, 10, 15],
'min_samples_split': [2, 5, 10]
} 4. Model Evaluation
python
Copy
Edit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
"""Evaluate model performance using various metrics."""
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
    print(f"Evaluation Metrics: {metrics}")
    return metrics

5.  Main Modelling Pipeline
    python
    Copy
    Edit
    def main_modelling_pipeline(data_path, target_col):
    """Execute the modelling pipeline.""" # Load preprocessed data
    data = pd.read_csv(data_path) # Split the data
    X_train, X_test, y_train, y_test = split_data(data, target_col)

        # Train models
        trained_models = train_models(X_train, y_train)

        # Hyperparameter tuning (example for Random Forest)
        print("\n--- Hyperparameter Tuning for Random Forest ---\n")
        rf_model = hyperparameter_tuning(trained_models["Random Forest"], rf_param_grid, X_train, y_train)

        # Evaluate models
        print("\n--- Model Evaluation ---\n")
        for name, model in trained_models.items():
            print(f"\n{name} Evaluation:")
            evaluate_model(model, X_test, y_test)

        print("\nTuned Random Forest Evaluation:")
        evaluate_model(rf_model, X_test, y_test)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import argparse
import os

def train_models(X_train, y_train):
    """
    Train Logistic Regression and Random Forest models.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        dict: Trained models.
    """
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models and print metrics.

    Args:
        models (dict): Trained models.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
    """
    for name, model in models.items():
        predictions = model.predict(X_test)
        print(f"Evaluation for {name}:")
        print(classification_report(y_test, predictions))
        if hasattr(model, "predict_proba"):
            roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            print(f"ROC-AUC Score: {roc_auc}")
        print("-" * 40)

def main(data_path, output_dir, target_col):
    # Load the processed dataset
    data = pd.read_csv(data_path)

    # Split data into features and target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    trained_models = train_models(X_train, y_train)

    # Evaluate models
    evaluate_models(trained_models, X_test, y_test)

    # Save models (optional, if needed later)
    os.makedirs(output_dir, exist_ok=True)
    for name, model in trained_models.items():
        model_path = os.path.join(output_dir, f"{name}_model.pkl")
        pd.to_pickle(model, model_path)
        print(f"Saved {name} model to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training and evaluation pipeline.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the processed dataset.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save outputs.")
    parser.add_argument('--target_col', type=str, required=True, help="Target column for modeling.")
    args = parser.parse_args()

    main(args.data_path, args.output_dir, args.target_col)
