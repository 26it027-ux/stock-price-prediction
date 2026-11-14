"""Evaluate price forecasts and compare model performance."""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from src.features import DEFAULT_FEATURE_COLUMNS


def inverse_scale_target(
    values: np.ndarray,
    scaler: MinMaxScaler,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = "Close",
) -> np.ndarray:
    """Convert scaled target values back to their original price scale."""
    columns = list(feature_columns)
    if target_column not in columns:
        raise ValueError("target_column must also be included in feature_columns.")
    if len(columns) != scaler.n_features_in_:
        raise ValueError(
            "feature_columns must match the features used to fit the scaler."
        )

    values = np.asarray(values).reshape(-1)
    target_index = columns.index(target_column)
    scaled_rows = np.zeros((len(values), len(columns)))
    scaled_rows[:, target_index] = values
    return scaler.inverse_transform(scaled_rows)[:, target_index]


def evaluate_predictions(
    y_test: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    scaler: MinMaxScaler,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    target_column: str = "Close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a MAE/RMSE/R² comparison table and original-scale predictions.

    Args:
        y_test: Scaled true target values from the chronological test set.
        predictions: Model names mapped to their scaled prediction arrays.
        scaler: The fitted MinMaxScaler returned by ``prepare_sequences``.
        feature_columns: Features used when fitting the scaler.
        target_column: Column whose predictions should be evaluated.

    Returns:
        A comparison table and a DataFrame containing actual and predicted
        closing prices, both in the original price scale.
    """
    actual_scaled = np.asarray(y_test).reshape(-1)
    if len(actual_scaled) < 2:
        raise ValueError("At least two test samples are needed to calculate R².")
    if not predictions:
        raise ValueError("Provide at least one model's predictions.")

    actual_prices = inverse_scale_target(
        actual_scaled, scaler, feature_columns, target_column
    )
    prediction_data = {"Actual": actual_prices}
    metric_rows = []

    for model_name, model_predictions in predictions.items():
        predicted_scaled = np.asarray(model_predictions).reshape(-1)
        if len(predicted_scaled) != len(actual_scaled):
            raise ValueError(
                f"{model_name} produced {len(predicted_scaled)} predictions for "
                f"{len(actual_scaled)} test samples."
            )

        predicted_prices = inverse_scale_target(
            predicted_scaled, scaler, feature_columns, target_column
        )
        prediction_data[model_name] = predicted_prices
        metric_rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(actual_prices, predicted_prices),
                "RMSE": np.sqrt(mean_squared_error(actual_prices, predicted_prices)),
                "R2": r2_score(actual_prices, predicted_prices),
            }
        )

    comparison = pd.DataFrame(metric_rows).sort_values("RMSE").reset_index(drop=True)
    return comparison, pd.DataFrame(prediction_data)
