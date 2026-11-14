"""Generate the next trading day's predicted closing price."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential

from src.evaluation import inverse_scale_target
from src.features import DEFAULT_FEATURE_COLUMNS, clean_stock_data


ModelType = Literal["linear_regression", "random_forest", "lstm"]
StockModel = LinearRegression | RandomForestRegressor | Sequential


def predict_next_day(
    model: StockModel,
    recent_data: pd.DataFrame,
    scaler: MinMaxScaler,
    model_type: ModelType = "lstm",
    window_size: int = 60,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = "Close",
) -> float:
    """Forecast the next trading day's closing price.

    ``recent_data`` should contain the latest raw OHLCV rows, including at
    least ``window_size`` records. The model must have been trained with the
    same features, window size, and fitted scaler.
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    if model_type not in {"linear_regression", "random_forest", "lstm"}:
        raise ValueError(f"Unsupported model_type: {model_type!r}")

    columns = list(feature_columns)
    if target_column not in columns:
        raise ValueError("target_column must also be included in feature_columns.")
    if len(columns) != scaler.n_features_in_:
        raise ValueError(
            "feature_columns must match the features used to fit the scaler."
        )

    cleaned = clean_stock_data(recent_data)
    if len(cleaned) < window_size:
        raise ValueError(
            f"At least {window_size} valid stock rows are needed; "
            f"received {len(cleaned)}."
        )

    latest_features = cleaned[columns].tail(window_size)
    scaled_window = scaler.transform(latest_features)
    model_input = scaled_window.reshape(1, window_size, len(columns))

    if model_type == "lstm":
        scaled_prediction = model.predict(model_input, verbose=0)
    else:
        scaled_prediction = model.predict(model_input.reshape(1, -1))

    predicted_price = inverse_scale_target(
        np.asarray(scaled_prediction),
        scaler,
        columns,
        target_column,
    )
    return float(predicted_price[0])
