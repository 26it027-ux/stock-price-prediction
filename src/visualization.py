"""Create exploratory data analysis and model-result plots."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_eda(
    data: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> list[Path]:
    """Save closing-price, moving-average, volume, and correlation plots."""
    required_columns = {"Date", "Close", "Volume"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "EDA data is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_data = data.copy()
    plot_data["Date"] = pd.to_datetime(plot_data["Date"], errors="coerce")
    plot_data = plot_data.dropna(subset=["Date"]).sort_values("Date")
    if plot_data.empty:
        raise ValueError("EDA data contains no rows with valid dates.")

    saved_files = []

    # Plot the closing price together with simple moving averages.
    plot_data["MA20"] = plot_data["Close"].rolling(window=20).mean()
    plot_data["MA50"] = plot_data["Close"].rolling(window=50).mean()
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.plot(plot_data["Date"], plot_data["Close"], label="Close", linewidth=1)
    axis.plot(plot_data["Date"], plot_data["MA20"], label="20-day average")
    axis.plot(plot_data["Date"], plot_data["MA50"], label="50-day average")
    axis.set(title="Stock Closing Price and Moving Averages", xlabel="Date", ylabel="Price")
    axis.legend()
    fig.tight_layout()
    price_file = output_path / "closing_price_moving_averages.png"
    fig.savefig(price_file, dpi=150)
    plt.close(fig)
    saved_files.append(price_file)

    # Show traded volume separately because its scale differs from price.
    fig, axis = plt.subplots(figsize=(12, 4))
    axis.bar(plot_data["Date"], plot_data["Volume"], width=1.0, color="slateblue")
    axis.set(title="Daily Trading Volume", xlabel="Date", ylabel="Volume")
    fig.tight_layout()
    volume_file = output_path / "trading_volume.png"
    fig.savefig(volume_file, dpi=150)
    plt.close(fig)
    saved_files.append(volume_file)

    numeric_data = plot_data.select_dtypes(include="number")
    if numeric_data.shape[1] < 2:
        raise ValueError("At least two numeric columns are needed for a correlation plot.")
    fig, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axis)
    axis.set_title("Stock Feature Correlation")
    fig.tight_layout()
    correlation_file = output_path / "correlation_heatmap.png"
    fig.savefig(correlation_file, dpi=150)
    plt.close(fig)
    saved_files.append(correlation_file)

    return saved_files


def plot_actual_vs_predicted(
    predictions: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> list[Path]:
    """Save one actual-versus-predicted closing-price plot per model."""
    if "Actual" not in predictions.columns:
        raise ValueError("The predictions table must include an 'Actual' column.")
    model_names = [column for column in predictions.columns if column != "Actual"]
    if not model_names:
        raise ValueError("The predictions table does not contain model results.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for model_name in model_names:
        fig, axis = plt.subplots(figsize=(12, 5))
        axis.plot(predictions["Actual"].to_numpy(), label="Actual")
        axis.plot(predictions[model_name].to_numpy(), label="Predicted")
        axis.set(
            title=f"Actual vs Predicted Closing Price: {model_name}",
            xlabel="Test trading day",
            ylabel="Price",
        )
        axis.legend()
        fig.tight_layout()
        safe_name = re.sub(r"[^a-z0-9]+", "_", str(model_name).lower()).strip("_")
        plot_file = output_path / f"actual_vs_predicted_{safe_name}.png"
        fig.savefig(plot_file, dpi=150)
        plt.close(fig)
        saved_files.append(plot_file)

    return saved_files


def plot_lstm_loss(
    loss_history: Mapping[str, Sequence[float]],
    output_dir: str | Path = "outputs",
) -> Path:
    """Save the LSTM training and validation loss curves."""
    if "loss" not in loss_history:
        raise ValueError("The LSTM history must contain a 'loss' series.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(loss_history["loss"], label="Training loss")
    if "val_loss" in loss_history:
        axis.plot(loss_history["val_loss"], label="Validation loss")
    axis.set(title="LSTM Training Loss", xlabel="Epoch", ylabel="Mean squared error")
    axis.legend()
    fig.tight_layout()
    loss_file = output_path / "lstm_loss.png"
    fig.savefig(loss_file, dpi=150)
    plt.close(fig)
    return loss_file
