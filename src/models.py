"""Train the baseline, Random Forest, and LSTM stock-price models."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.callbacks import History


def _validate_training_data(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple[int, int]:
    """Check that the inputs are valid windows and return their dimensions."""
    if X_train.ndim != 3:
        raise ValueError(
            "X_train must have shape (samples, window_size, features)."
        )
    if y_train.ndim != 1:
        raise ValueError("y_train must be a one-dimensional target array.")
    if len(X_train) == 0 or len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must contain the same non-zero samples.")

    return X_train.shape[1], X_train.shape[2]


def train_linear_regression(
    X_train: np.ndarray, y_train: np.ndarray
) -> LinearRegression:
    """Fit the linear regression baseline on flattened time windows."""
    _validate_training_data(X_train, y_train)
    model = LinearRegression()
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Fit a Random Forest regressor on flattened time windows."""
    _validate_training_data(X_train, y_train)
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train.reshape(len(X_train), -1), y_train)
    return model


def build_lstm_model(input_shape: tuple[int, int]) -> Sequential:
    """Create an LSTM network for a window of scaled stock features."""
    if len(input_shape) != 2 or any(dimension < 1 for dimension in input_shape):
        raise ValueError("input_shape must contain positive window and feature sizes.")

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 32,
) -> tuple[Sequential, History]:
    """Train the LSTM and return both the fitted model and its loss history."""
    window_size, feature_count = _validate_training_data(X_train, y_train)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be at least 1.")

    model = build_lstm_model((window_size, feature_count))
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        shuffle=False,
        verbose=1,
    )
    return model, history
