import pandas as pd
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Networks.fully_connecnted_feed_forward import create_fully_connected_feed_forward
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_completeness.py <input_file>")
        print("Please provide the UHGG database metadata file as input.")
        sys.exit(1)
    input_file = sys.argv[1]
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

    # Split and scale the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Data preprocessing completed. Starting model training...", flush=True)

    # Fit a linear regression model
    lr = LinearRegression().fit(X_train_scaled, y_train)
    # Predict completeness on the test set
    y_pred_lr = lr.predict(X_test_scaled)

    print("Linear regression model trained.", flush=True)

    # Fit a random forest model
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)
    # Predict completeness on the test set
    y_pred_rf = rf.predict(X_test_scaled)

    print("Random forest model trained.", flush=True)

    # Fit a fully connected feed-forward neural network model
    ff_nn = create_fully_connected_feed_forward(X_train_scaled.shape[1:], 1, loss='mse')
    ff_nn.fit(X_train_scaled, y_train, epochs=100, batch_size=50, verbose=0,
              validation_split=0.2)

    # Predict completeness on the test set
    y_pred_ff_nn = ff_nn.predict(X_test_scaled).flatten()

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
