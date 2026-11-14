"""Train, test, and run simple next-day stock-price models."""

import argparse
from math import sqrt
from pathlib import Path

import matplotlib  # pyright: ignore[reportMissingModuleSource]
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestRegressor  # pyright: ignore[reportMissingModuleSource]
from sklearn.linear_model import LinearRegression  # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import (  # pyright: ignore[reportMissingModuleSource]
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

if __package__:
    from .data_loader import load_stock_data
else:
    from data_loader import load_stock_data


FEATURES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA_5",
    "MA_20",
    "Daily_Return",
    "Previous_Close",
]


def prepare_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create past/current-day features and tomorrow's close target."""
    data = raw_data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"]).sort_values("Date")

    price_columns = ["Open", "High", "Low", "Close", "Volume"]
    for column in price_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[price_columns] = data[price_columns].ffill()
    data = data.dropna(subset=price_columns).drop_duplicates("Date").reset_index(drop=True)

    # Features for day t only use information available by the end of day t.
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["Daily_Return"] = data["Close"].pct_change()
    data["Previous_Close"] = data["Close"].shift(1)
    data["Target"] = data["Close"].shift(-1)
    data["Target_Date"] = data["Date"].shift(-1)

    usable_features = data.dropna(subset=FEATURES).copy()
    training_rows = usable_features.dropna(subset=["Target"]).copy()
    return training_rows, usable_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Tesla's next trading-day closing price."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional Kaggle CSV with Date, Open, High, Low, Close, and Volume.",
    )
    parser.add_argument("--ticker", default="TSLA", help="Yahoo Finance ticker.")
    parser.add_argument("--start", default="2018-01-01", help="Download start date.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = args.csv
    if csv_path is not None and not csv_path.is_absolute():
        csv_path = project_root / csv_path
    default_csv = project_root / "data" / "Tesla.csv"
    if csv_path is None and default_csv.is_file():
        csv_path = default_csv

    if csv_path is None:
        print(
            f"Downloading {args.ticker} daily prices from {args.start}...",
            flush=True,
        )
    else:
        print(f"Loading CSV: {csv_path}", flush=True)
    raw_data = load_stock_data(
        csv_path=csv_path,
        ticker=args.ticker,
        start=args.start,
    )
    print(f"Loaded {len(raw_data)} daily rows.", flush=True)
    training_data, feature_data = prepare_data(raw_data)

    if len(training_data) < 30:
        raise ValueError(
            "Not enough usable stock rows. Use a longer date range or a larger CSV."
        )

    # Keep the latest 20% of dates for a realistic, chronological test.
    split_at = int(len(training_data) * 0.8)
    if len(training_data) - split_at < 2:
        raise ValueError("At least two test rows are needed to calculate R².")
    train = training_data.iloc[:split_at]
    test = training_data.iloc[split_at:]
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["Target"], test["Target"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=1
        ),
    }
    results = []
    predictions = {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[name] = predicted
        results.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, predicted),
                "RMSE": sqrt(mean_squared_error(y_test, predicted)),
                "R2": r2_score(y_test, predicted),
            }
        )

    comparison = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\nTest results (lower MAE/RMSE is better):")
    print(comparison.round(3).to_string(index=False))

    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    # Plot test-set forecasts against actual next-day closing prices.
    plt.figure(figsize=(12, 5))
    plt.plot(test["Target_Date"], y_test, label="Actual", linewidth=2)
    for name, predicted in predictions.items():
        plt.plot(test["Target_Date"], predicted, label=name)
    plt.title("Actual vs Predicted Next-Day Closing Price")
    plt.xlabel("Trading date")
    plt.ylabel("Closing price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    # Train-set feature importance is available for the Random Forest model.
    forest = models["Random Forest"]
    importance = pd.Series(forest.feature_importances_, index=FEATURES).sort_values()
    importance.plot(kind="barh", figsize=(8, 5), title="Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png", dpi=150)
    plt.close()

    # Use the latest completed day's features for a next-trading-day forecast.
    latest = feature_data[FEATURES].iloc[[-1]]
    forecast_date = feature_data["Date"].iloc[-1]
    forecasts = pd.DataFrame(
        [
            {
                "Model": name,
                "Last observed date": forecast_date.date(),
                "Predicted next close": model.predict(latest)[0],
            }
            for name, model in models.items()
        ]
    )
    forecasts.to_csv(output_dir / "next_day_forecast.csv", index=False)
    print("\nNext trading-day forecasts:")
    print(forecasts.round(2).to_string(index=False))
    print(f"\nCharts and CSV results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
