"""Prepare stock-price features and time-windowed model inputs."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


STOCK_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
DEFAULT_FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def clean_stock_data(data: pd.DataFrame) -> pd.DataFrame:
    """Sort stock records and handle missing or invalid values.

    Missing numeric values are forward-filled from earlier trading days.
    Rows with no earlier value to fill from are dropped to avoid using future
    prices to fill past data.
    """
    missing_columns = [column for column in STOCK_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(
            "Stock data is missing required columns: "
            + ", ".join(missing_columns)
        )

    cleaned = data[STOCK_COLUMNS].copy()
    cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["Date"])

    for column in DEFAULT_FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned = cleaned.sort_values("Date").drop_duplicates(
        subset="Date", keep="last"
    )
    cleaned[DEFAULT_FEATURE_COLUMNS] = cleaned[DEFAULT_FEATURE_COLUMNS].ffill()
    cleaned = cleaned.dropna(subset=DEFAULT_FEATURE_COLUMNS).reset_index(drop=True)

    if cleaned.empty:
        raise ValueError("No complete stock records remain after cleaning.")

    return cleaned


def prepare_sequences(
    data: pd.DataFrame,
    window_size: int = 60,
    test_size: float = 0.2,
    feature_columns: Sequence[str] | None = None,
    target_column: str = "Close",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Clean, scale, and split data into chronological prediction windows.

    The scaler is fitted only on the training period. Each input contains the
    previous ``window_size`` trading days, and its target is the following
    day's value for ``target_column``.

    Returns:
        X_train, X_test, y_train, y_test, and the fitted MinMaxScaler.
        The feature arrays have shape (samples, window_size, features).
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    cleaned = clean_stock_data(data)
    selected_features = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    if target_column not in selected_features:
        raise ValueError("target_column must also be included in feature_columns.")

    missing_features = [
        column for column in selected_features if column not in cleaned.columns
    ]
    if missing_features:
        raise ValueError(
            "Stock data is missing selected features: " + ", ".join(missing_features)
        )

    split_index = int(len(cleaned) * (1 - test_size))
    if split_index <= window_size or split_index >= len(cleaned):
        raise ValueError(
            "Not enough data for the requested window and time-based split. "
            "Add more rows, reduce window_size, or increase test_size."
        )

    scaler = MinMaxScaler()
    training_values = cleaned[selected_features].iloc[:split_index]
    scaler.fit(training_values)
    scaled_values = scaler.transform(cleaned[selected_features])
    target_index = selected_features.index(target_column)

    X_train, X_test = [], []
    y_train, y_test = [], []
    for target_position in range(window_size, len(scaled_values)):
        sequence = scaled_values[target_position - window_size : target_position]
        target = scaled_values[target_position, target_index]

        # Assign by the target date so the test set stays strictly after training.
        if target_position < split_index:
            X_train.append(sequence)
            y_train.append(target)
        else:
            X_test.append(sequence)
            y_test.append(target)

    return (
        np.asarray(X_train, dtype=np.float32),
        np.asarray(X_test, dtype=np.float32),
        np.asarray(y_train, dtype=np.float32),
        np.asarray(y_test, dtype=np.float32),
        scaler,
    )
