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
