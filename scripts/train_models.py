import logging
import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Set up logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_models():
    """
    Initializes and returns a dictionary of machine learning models.
    
    Returns:
        dict: A dictionary containing models.
    """
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=200, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=200),
        'AdaBoost': AdaBoostClassifier(n_estimators=200)
    }
    logging.info(f"Models initialized: {list(models.keys())}")
    return models


def load_data(X_path, y_path):
    """
    Loads training features and labels from separate CSV files.

    Args:
        X_path (str): Path to the CSV file containing training features.
        y_path (str): Path to the CSV file containing training labels.

    Returns:
        tuple: Tuple containing feature matrix and target vector (X_train, y_train).
    """
    logging.info(f"Loading features from {X_path}")
    X_train = pd.read_csv(X_path)

    logging.info(f"Loading labels from {y_path}")
    y_train = pd.read_csv(y_path)
    y_train = y_train['piezo_groundwater_level_category']

    logging.info(f"Data loaded successfully. Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")
    return X_train, y_train


def roc_auc_ovr(y_true, y_pred):
    """
    Custom function to calculate AUC-ROC for multiclass classification.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.

    Returns:
        float: AUC-ROC score.
    """
    return roc_auc_score(y_true, y_pred, multi_class='ovr', 
                         average='macro', labels=np.unique(y_true))


def evaluate_model(model, X_train, y_train, scoring):
    """
    Evaluates a given model using cross-validation and various scoring metrics.

    Args:
        model (sklearn model): The model to be evaluated.
        X_train (ndarray): The training features.
        y_train (ndarray): The training labels.
        scoring (list): A list of scoring metrics.

    Returns:
        dict: Mean and standard deviation of each metric.
    """
    logging.info(f"Evaluating model: {model.__class__.__name__}")
    
    cv_results = cross_validate(model, X_train, y_train, 
                                cv=3, scoring=scoring, 
                                n_jobs=-1, verbose=1)
    
    # Extract mean and standard deviation for each metric
    metrics = {
        'accuracy': (
            round(cv_results['test_accuracy'].mean(), 4), 
            round(cv_results['test_accuracy'].std(), 4)
            ),
        'f1_score': (
            round(cv_results['test_f1_macro'].mean(), 4), 
            round(cv_results['test_f1_macro'].std(), 4)
            ),
        'precision': (
            round(cv_results['test_precision_macro'].mean(), 4), 
            round(cv_results['test_precision_macro'].std(), 4)
            ),
        'recall': (
            round(cv_results['test_recall_macro'].mean(), 4), 
            round(cv_results['test_recall_macro'].std(), 4)
            ),
        'roc_auc': (
            round(cv_results['test_roc_auc_ovr'].mean(), 4), 
            round(cv_results['test_roc_auc_ovr'].std(), 4)
            )
    }
    
    return metrics


def evaluate_models(models, X_train, y_train, scoring):
    """
    Evaluates multiple models using cross-validation.

    Args:
        models (dict): A dictionary containing models.
        X_train (ndarray): The training features.
        y_train (ndarray): The training labels.
        scoring (list): A list of scoring metrics.
    """
    for model_name, model in models.items():
        metrics = evaluate_model(model, X_train, y_train, scoring)
        
        # Print results for each model
        logging.info(f"{model_name}:")
        for metric, (mean, std) in metrics.items():
            logging.info(f"  {metric.capitalize()}: {mean:.4f} ± {std:.4f}")
        logging.info("-" * 50)


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate multiple models using cross-validation.")
    parser.add_argument('--X_path', type=str, help="Path to the CSV file containing the training features.")
    parser.add_argument('--y_path', type=str, help="Path to the CSV file containing the training labels.")
    return parser.parse_args()


def main(X_path, y_path):
    """
    Main function to load data, evaluate models, and print results.

    Args:
        X_path (str): Path to the CSV file containing training features.
        y_path (str): Path to the CSV file containing training labels.
    """
    # Load data
    X_train, y_train = load_data(X_path, y_path)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, 
                                                        y_train, 
                                                        test_size=0.3, 
                                                        random_state=42)

    # Initialize models
    models = initialize_models()

    # Define scoring metrics
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro', 'roc_auc_ovr']

    # Evaluate models
    evaluate_models(models, X_train, y_train, scoring)


if __name__ == "__main__":

    # Parse command-line arguments
    args = parse_arguments()

    # Run the main function with the provided paths
    main(args.X_path, args.y_path)

    