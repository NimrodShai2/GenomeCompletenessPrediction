import argparse

import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Networks.fully_connecnted_feed_forward import create_fully_connected_feed_forward, \
    create_fully_connected_feed_forward_with_dropout
import matplotlib.pyplot as plt

from cross_validate import run_cross_validation


def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance using Mean Squared Error (MSE) and R-squared metrics.

    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values

    Returns:
    - mse: float, Mean Squared Error
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def run_train_test(X, y, dropout_rate, epochs):
    """
    Run the train-test split and fit multiple models to predict genome completeness.
    :param X: The feature set containing genome metadata.
    :param y: The target variable representing genome completeness.
    :return: None
    """
    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data preprocessing completed. Starting model training...", flush=True)
    # Fit a linear regression model
    lr = LinearRegression().fit(X_train, y_train)
    # Predict completeness on the test set
    y_pred_lr = lr.predict(X_test)
    print("Linear regression model trained.", flush=True)
    # Fit a random forest model
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1).fit(X_train, y_train)
    # Predict completeness on the test set
    y_pred_rf = rf.predict(X_test)
    print("Random forest model trained.", flush=True)
    # Fit a fully connected feed-forward neural network model
    if dropout_rate < 0:
        ff_nn = create_fully_connected_feed_forward(X_train.shape[1:], 1, loss='mse')
    else:
        ff_nn = create_fully_connected_feed_forward_with_dropout(X_train.shape[1:], 1, loss='mse',
                                                                 dropout_rate=dropout_rate)

    ff_nn.fit(X_train, y_train, epochs=epochs, batch_size=50, verbose=0,
              validation_split=0.2)
    # Predict completeness on the test set
    y_pred_ff_nn = ff_nn.predict(X_test).flatten()
    print("Fully connected feed-forward neural network model trained.", flush=True)
    # Scale the prediction to be 0 to 100
    y_pred_ff_nn = np.clip(y_pred_ff_nn, 0, 100)
    y_test = np.clip(y_test, 0, 100)
    y_pred_rf = np.clip(y_pred_rf, 0, 100)
    # Evaluate the models
    mse_lr = evaluate_model(y_test, y_pred_lr)
    mse_rf = evaluate_model(y_test, y_pred_rf)
    mse_ff_nn = evaluate_model(y_test, y_pred_ff_nn)
    print(f"Linear Regression MSE: {mse_lr:.4f}")
    print(f"Random Forest MSE: {mse_rf:.4f}")
    print(f"Fully Connected Feed-Forward NN MSE: {mse_ff_nn:.4f}")
    print("Model evaluation completed.")
    # Plot the results
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred_ff_nn, alpha=0.5, label='Neural Net')
    plt.scatter(y_test, y_pred_rf, alpha=0.3, label='Random Forest')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--')
    plt.xlabel("True Completeness")
    plt.ylabel("Predicted Completeness")
    plt.legend()
    plt.title("Genome Completeness Prediction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to UHGG metadata file")
    parser.add_argument("--cv", action="store_true", help="Enable cross-validation")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate for the neural network (default: 0)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs for training the neural network (default: 100)")
    args = parser.parse_args()

    input_file = args.input_file
    df = pd.read_csv(input_file, sep="\t")

    # Select relevant columns and preprocess the data
    features = ['Length', 'N_contigs', 'N50', 'GC_content',
                'Contamination', 'rRNA_5S', 'rRNA_16S', 'rRNA_23S', 'tRNAs']
    target = 'Completeness'

    # Ensure all features and target are numeric
    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
        df[target] = pd.to_numeric(df[target], errors='coerce')

    # Drop rows with NaN values in features or target
    df = df.dropna(subset=features + [target])

    # clip completeness, contamination, and Gc content to be between 0 and 100
    df.loc[:, 'Completeness'] = df['Completeness'].clip(0, 100)
    df.loc[:, 'Contamination'] = df['Contamination'].clip(0, 100)
    df.loc[:, 'GC_content'] = df['GC_content'].clip(0, 100)

    X = df[features]
    y = df[target]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if args.cv:
        print(f"Running cross-validation with {args.folds} folds...", flush=True)
        run_cross_validation(X_scaled, y, args.folds, args.dropout, args.epochs)
        print("Cross-validation completed.")
    else:
        print("Running train-test split...", flush=True)
        run_train_test(X_scaled, y, args.dropout, args.epochs)
        print("Train-test split completed.")
