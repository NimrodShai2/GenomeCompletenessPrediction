import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
from tensorflow.python.keras.backend import clear_session

from Networks.fully_connecnted_feed_forward import create_fully_connected_feed_forward, \
    create_fully_connected_feed_forward_with_dropout


def run_cross_validation(X, y, n_folds, dropout_rate, epochs=100):
    """
    Run cross-validation for multiple models to predict genome completeness.
    This function evaluates Linear Regression, Random Forest, and a Neural Network model
    :param X: The feature set containing genome metadata.
    :param y: The target variable representing genome completeness.
    :param n_folds:  Number of folds for cross-validation.
    :return: None
    """
    # Linear Regression
    lr = LinearRegression()
    lr_mse = -cross_val_score(lr, X, y, cv=n_folds, scoring='neg_mean_squared_error')
    lr_mae = -cross_val_score(lr, X, y, cv=n_folds, scoring='neg_mean_absolute_error')
    lr_r2 = cross_val_score(lr, X, y, cv=n_folds, scoring='r2')

    print("Cross-validation for Linear Regression completed.", flush=True)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_mse = -cross_val_score(rf, X, y, cv=n_folds, scoring='neg_mean_squared_error')
    rf_mae = -cross_val_score(rf, X, y, cv=n_folds, scoring='neg_mean_absolute_error')
    rf_r2 = cross_val_score(rf, X, y, cv=n_folds, scoring='r2')

    print("Cross-validation for Random Forest completed.", flush=True)

    # Neural Network (manual CV)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    nn_mse, nn_mae, nn_r2 = [], [], []

    for train_idx, test_idx in kf.split(X):
        clear_session()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if dropout_rate < 0:
            model = create_fully_connected_feed_forward(X_train.shape[1:], 1, loss='mse')
        else:
            model = create_fully_connected_feed_forward_with_dropout(X_train.shape[1:], 1, loss='mse',
                                                                     dropout_rate=dropout_rate)

        model.fit(X_train, y_train, epochs=epochs, batch_size=50, verbose=0)

        y_pred = model.predict(X_test).flatten()
        y_pred = np.clip(y_pred, 0, 100)
        y_test = np.clip(y_test, 0, 100)

        nn_mse.append(mean_squared_error(y_test, y_pred))
        nn_mae.append(mean_absolute_error(y_test, y_pred))
        nn_r2.append(r2_score(y_test, y_pred))
        print("Neural Network fold completed.", flush=True)

    print("Cross-validation for Neural Network completed.", flush=True)

    print("\nCross-Validation Results ({} folds):".format(n_folds))
    for name, mse, mae, r2 in [
        ("Linear Regression", lr_mse, lr_mae, lr_r2),
        ("Random Forest", rf_mse, rf_mae, rf_r2),
        ("Neural Network", nn_mse, nn_mae, nn_r2),
    ]:
        print(f"{name}:")
        print(f"  Avg MSE: {np.mean(mse):.4f} ± {np.std(mse):.4f}")
        print(f"  Avg MAE: {np.mean(mae):.4f} ± {np.std(mae):.4f}")
        print(f"  Avg R² : {np.mean(r2):.4f} ± {np.std(r2):.4f}\n")

    # Print the best model based on mean MSE
    best_model = min(
        [("Linear Regression", np.mean(lr_mse)),
         ("Random Forest", np.mean(rf_mse)),
         ("Neural Network", np.mean(nn_mse))],
        key=lambda x: x[1]
    )
    print(f"Best model based on MSE: {best_model[0]} with MSE {best_model[1]:.4f}")
