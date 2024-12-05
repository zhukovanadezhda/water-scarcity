import logging
import numpy as np
import pandas as pd
import argparse
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# Set up logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def build_model(input_shape):
    """
    Builds and compiles the CNN model.

    Args:
        input_shape (tuple): The shape of the input data (e.g., (num_features, 1)).

    Returns:
        model (keras.Model): Compiled CNN model.
    """
    logging.info("Building the CNN model.")

    model = Sequential()

    # First Conv1D layer
    model.add(Conv1D(64, kernel_size=7, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Second Conv1D layer
    model.add(Conv1D(256, kernel_size=7, activation='relu'))  
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Third Conv1D layer
    model.add(Conv1D(512, kernel_size=7, activation='relu'))  
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))

    # Flatten the output for dense layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    logging.info("CNN model built and compiled successfully.")

    return model


def train_and_evaluate_fold(model, X_train_fold, y_train_fold, X_val_fold, y_val_fold):
    """
    Trains and evaluates the model for a single fold of cross-validation.

    Args:
        model (keras.Model): The model to be trained.
        X_train_fold (ndarray): Training features for the fold.
        y_train_fold (ndarray): Training labels for the fold.
        X_val_fold (ndarray): Validation features for the fold.
        y_val_fold (ndarray): Validation labels for the fold.

    Returns:
        tuple: Accuracy, F1, Precision, Recall, ROC AUC for the fold.
    """
    logging.info("Training and evaluating the model for the fold.")

    # Callbacks for early stopping and learning rate reduction
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train the model
    model.fit(X_train_fold, y_train_fold, epochs=150, batch_size=128, 
              validation_data=(X_val_fold, y_val_fold),
              callbacks=[lr_scheduler, early_stop], verbose=0)

    # Evaluate the model on the validation fold
    y_pred = model.predict(X_val_fold)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val_fold, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_val_classes, y_pred_classes)
    f1 = f1_score(y_val_classes, y_pred_classes, average='weighted')
    precision = precision_score(y_val_classes, y_pred_classes, average='weighted')
    recall = recall_score(y_val_classes, y_pred_classes, average='weighted')
    roc_auc = roc_auc_score(y_val_fold, y_pred, multi_class='ovr')

    logging.info("Fold evaluation complete.")

    return accuracy, f1, precision, recall, roc_auc


def cross_validate(X_train, y_train, n_splits=3):
    """
    Performs K-fold cross-validation and logs the results.

    Args:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
        n_splits (int): Number of folds for cross-validation. Default is 3.
    """
    logging.info(f"Starting {n_splits}-fold cross-validation.")

    # Reshape X_train for CNN (as if each sample is a 1D "image")
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1) 

    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes=5)

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results for each fold
    accuracies, f1_scores, precisions, recalls, roc_aucs = [], [], [], [], []

    # Loop through each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logging.info(f"Processing fold {fold}.")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        # Build and compile the model
        model = build_model(input_shape=(X_train.shape[1], 1))

        # Train and evaluate the model
        accuracy, f1, precision, recall, roc_auc = train_and_evaluate_fold(
            model, X_train_fold, y_train_fold, X_val_fold, y_val_fold
        )

        # Append metrics to their respective lists
        accuracies.append(accuracy)
        f1_scores.append(f1)
        precisions.append(precision)
        recalls.append(recall)
        roc_aucs.append(roc_auc)

    # Calculate and log mean ± std for each metric
    logging.info(f"Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    logging.info(f"F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    logging.info(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    logging.info(f"Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    logging.info(f"ROC AUC: {np.mean(roc_aucs):.4f} ± {np.std(roc_aucs):.4f}")


def load_data(X_path, y_path):
    """
    Loads training features and labels from separate CSV files.

    Args:
        X_path (str): Path to the CSV file containing training features.
        y_path (str): Path to the CSV file containing training labels.

    Returns:
        X_train (ndarray): Training features.
        y_train (ndarray): Training labels.
    """
    logging.info(f"Loading features from {X_path}")
    X_train = pd.read_csv(X_path).values

    logging.info(f"Loading labels from {y_path}")
    y_train = pd.read_csv(y_path).values.ravel()  # Flatten labels to a 1D array

    logging.info(f"Data loaded successfully. Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")

    return X_train, y_train


def main(X_path, y_path):
    """
    Main function to run the entire training and evaluation pipeline.

    Args:
        X_path (str): Path to the CSV file containing training features.
        y_path (str): Path to the CSV file containing training labels.
    """
    # Load data
    X_train, y_train = load_data(X_path, y_path)

    # Perform cross-validation
    cross_validate(X_train, y_train, n_splits=3)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN model using 3-fold cross-validation.")
    parser.add_argument('--X_path', type=str, help="Path to the CSV file containing the training features.")
    parser.add_argument('--y_path', type=str, help="Path to the CSV file containing the training labels.")

    # Parse arguments
    args = parser.parse_args()

    # Run the main function with the provided paths
    main(args.X_path, args.y_path)
