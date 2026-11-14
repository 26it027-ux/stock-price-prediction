"""Streamlit dashboard for educational next-day stock estimates."""

from datetime import date
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import (
    load_stock_data,
    normalize_multi_stock_data,
    normalize_stock_data,
)
from src.main import FEATURES, prepare_data


TOP_FIVE_TICKERS = ["AAPL", "MSFT", "AMZN", "META", "TSLA"]
SUGGESTED_STOCKS = {
    "Tesla (TSLA)": "TSLA",
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "NVIDIA (NVDA)": "NVDA",
    "Amazon (AMZN)": "AMZN",
    "Alphabet (GOOGL)": "GOOGL",
    "Meta Platforms (META)": "META",
    "Netflix (NFLX)": "NFLX",
    "JPMorgan Chase (JPM)": "JPM",
    "Coca-Cola (KO)": "KO",
}


st.set_page_config(page_title="Stock Direction & Price Lab", layout="wide")
st.title("Stock Direction & Price Lab")
st.write(
    "Choose one stock to estimate its next trading-day close and whether "
    "the close may be up or not up."
)
st.warning(
    "These are educational estimates only—not final or guaranteed predictions, "
    "financial advice, or trading signals."
)


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_history(ticker: str, start_date: str) -> pd.DataFrame:
    """Fetch and cache one ticker's historical daily prices for one hour."""
    return load_stock_data(ticker=ticker, start=start_date)


@st.cache_data(ttl=3600, show_spinner=False)
def read_uploaded_history(file_bytes: bytes, ticker: str) -> pd.DataFrame:
    """Parse and cache an uploaded CSV using the shared stock-data validator."""
    raw_data = pd.read_csv(BytesIO(file_bytes))
    return normalize_stock_data(raw_data, ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def read_uploaded_multi_stock_history(file_bytes: bytes) -> pd.DataFrame:
    """Parse and cache an uploaded multi-company Kaggle CSV."""
    raw_data = pd.read_csv(BytesIO(file_bytes))
    return normalize_multi_stock_data(raw_data)


def compare_top_stocks(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Fit the dashboard models separately for each selected company."""
    results = []
    for ticker_symbol in TOP_FIVE_TICKERS:
        history = stock_data.loc[stock_data["Ticker"] == ticker_symbol].drop(
            columns="Ticker"
        )
        training_data, feature_data = prepare_data(history)
        if len(training_data) < 30:
            raise ValueError(
                f"Not enough usable daily rows for {ticker_symbol}; "
                "use a longer history."
            )

        split_at = int(len(training_data) * 0.8)
        if len(training_data) - split_at < 2:
            raise ValueError(f"At least two test rows are needed for {ticker_symbol}.")
        train = training_data.iloc[:split_at]
        test = training_data.iloc[split_at:]
        X_train, X_test = train[FEATURES], test[FEATURES]
        y_train, y_test = train["Target"], test["Target"]

        linear_model = LinearRegression().fit(X_train, y_train)
        forest_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        ).fit(X_train, y_train)
        linear_predictions = linear_model.predict(X_test)
        forest_predictions = forest_model.predict(X_test)

        direction_train = (train["Target"] > train["Close"]).astype(int)
        direction_test = (test["Target"] > test["Close"]).astype(int)
        direction_model = DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
        ).fit(X_train, direction_train)
        direction_predictions = direction_model.predict(X_test)

        latest_features = feature_data[FEATURES].iloc[[-1]]
        last_close = float(latest_features["Close"].iloc[0])
        latest_date = feature_data["Date"].iloc[-1].date()
        linear_rmse = mean_squared_error(y_test, linear_predictions) ** 0.5
        forest_rmse = mean_squared_error(y_test, forest_predictions) ** 0.5
        baseline_rmse = mean_squared_error(
            y_test,
            test["Close"].to_numpy(),
        ) ** 0.5
        candidate_rmses = {
            "Linear Regression": linear_rmse,
            "Random Forest": forest_rmse,
            "Last-close baseline": baseline_rmse,
        }
        selected_method = min(candidate_rmses, key=candidate_rmses.get)
        next_close_estimates = {
            "Linear Regression": float(linear_model.predict(latest_features)[0]),
            "Random Forest": float(forest_model.predict(latest_features)[0]),
            "Last-close baseline": last_close,
        }
        results.append(
            {
                "Ticker": ticker_symbol,
                "Data through": latest_date,
                "Last close": last_close,
                "Linear next close": next_close_estimates["Linear Regression"],
                "Forest next close": next_close_estimates["Random Forest"],
                "Selected method": selected_method,
                "Selected next close": next_close_estimates[selected_method],
                "Direction": (
                    "UP"
                    if int(direction_model.predict(latest_features)[0]) == 1
                    else "NOT UP"
                ),
                "Direction test accuracy": accuracy_score(
                    direction_test,
                    direction_predictions,
                ),
                "Linear test RMSE": linear_rmse,
                "Forest test RMSE": forest_rmse,
                "Last-close test RMSE": baseline_rmse,
            }
        )

    return pd.DataFrame(results)


with st.sidebar:
    st.header("Stock selection")
    analysis_mode = st.radio(
        "Analysis mode",
        ["Single stock", "Top 5 from Kaggle"],
    )
    if analysis_mode == "Single stock":
        stock_choice = st.selectbox(
            "Choose or search a stock",
            [*SUGGESTED_STOCKS, "Other (enter ticker)"],
        )
        if stock_choice == "Other (enter ticker)":
            ticker = st.text_input("Enter ticker symbol").strip().upper()
        else:
            ticker = SUGGESTED_STOCKS[stock_choice]
        start_date = st.date_input(
            "History starts",
            value=date(2018, 1, 1),
            max_value=date.today(),
        )
    uploaded_file = st.file_uploader(
        "Optional Kaggle CSV",
        type=["csv"],
        help=(
            "Single-stock mode needs Date, Open, High, Low, Close, and Volume. "
            "Top-5 mode needs a Company column plus daily price columns."
        ),
    )
    button_label = (
        "Analyze stock"
        if analysis_mode == "Single stock"
        else "Train and compare top 5"
    )
    run_prediction = st.button(button_label, type="primary")
    if analysis_mode == "Single stock":
        st.caption("Search suggestions by company name or ticker, or enter a ticker manually.")

if analysis_mode == "Top 5 from Kaggle":
    if not run_prediction:
        st.info("Upload a multi-company Kaggle CSV, then select **Train and compare top 5**.")
        st.stop()

    try:
        if uploaded_file is not None:
            multi_stock_data = read_uploaded_multi_stock_history(
                uploaded_file.getvalue()
            )
        else:
            default_dataset = (
                Path(__file__).resolve().parent
                / "data"
                / "top_10_companies.csv"
            )
            if not default_dataset.is_file():
                st.error("Upload the Kaggle multi-company CSV to analyze five stocks.")
                st.stop()
            multi_stock_data = normalize_multi_stock_data(
                pd.read_csv(default_dataset)
            )

        missing_tickers = [
            symbol
            for symbol in TOP_FIVE_TICKERS
            if symbol not in set(multi_stock_data["Ticker"])
        ]
        if missing_tickers:
            st.error(
                "The CSV must include these five tickers: "
                + ", ".join(TOP_FIVE_TICKERS)
                + ". Missing: "
                + ", ".join(missing_tickers)
            )
            st.stop()

        with st.spinner("Training three models for each of the five stocks..."):
            top_five_results = compare_top_stocks(multi_stock_data)
    except (ImportError, ValueError, pd.errors.ParserError) as error:
        st.error(f"Could not analyze the Kaggle CSV: {error}")
        st.stop()

    st.subheader("Top 5 Kaggle stock predictions")
    st.caption(
        "Each stock is trained separately. The test uses the latest 20% of dates "
        "without shuffling. A last-close baseline is also tested; Selected method "
        "uses whichever candidate had the lowest test RMSE. Forecasts refer to the "
        "day after each CSV's final date."
    )
    st.dataframe(top_five_results.round(3), hide_index=True, width="stretch")
    st.download_button(
        "Download top-5 predictions as CSV",
        data=top_five_results.to_csv(index=False).encode("utf-8"),
        file_name="top5_stock_predictions.csv",
        mime="text/csv",
    )
    st.warning(
        "The selected Kaggle dataset ends in July 2023. These are historical "
        "estimates, not current predictions, guarantees, or trading signals."
    )
    st.stop()

if not run_prediction:
    st.info("Enter a ticker, then select **Analyze stock**.")
    st.stop()

if not ticker:
    st.error("Enter a stock ticker symbol.")
    st.stop()

with st.spinner(f"Loading {ticker} history and training the models..."):
    try:
        if uploaded_file is None:
            history = get_stock_history(ticker, start_date.isoformat())
        else:
            history = read_uploaded_history(uploaded_file.getvalue(), ticker)
    except (ImportError, ValueError) as error:
        st.error(f"Could not load {ticker}: {error}")
        st.stop()

    training_data, feature_data = prepare_data(history)
    if len(training_data) < 30:
        st.error("Not enough usable data. Choose an earlier start date.")
        st.stop()

    split_at = int(len(training_data) * 0.8)
    if len(training_data) - split_at < 2:
        st.error("At least two test rows are needed to calculate R².")
        st.stop()

    train = training_data.iloc[:split_at]
    test = training_data.iloc[split_at:]
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["Target"], test["Target"]

    regressors = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=1,
        ),
    }
    predictions = {}
    metric_rows = []
    for model_name, model in regressors.items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        predictions[model_name] = predicted
        metric_rows.append(
            {
                "Model": model_name,
                "MAE": mean_absolute_error(y_test, predicted),
                "RMSE": mean_squared_error(y_test, predicted) ** 0.5,
                "R²": r2_score(y_test, predicted),
            }
        )

    direction_train = (train["Target"] > train["Close"]).astype(int)
    direction_test = (test["Target"] > test["Close"]).astype(int)
    direction_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    direction_model.fit(X_train, direction_train)
    direction_predictions = direction_model.predict(X_test)
    direction_accuracy = accuracy_score(direction_test, direction_predictions)

    latest_features = feature_data[FEATURES].iloc[[-1]]
    last_close = float(latest_features["Close"].iloc[0])
    latest_date = feature_data["Date"].iloc[-1].date()
    next_prices = {
        model_name: float(model.predict(latest_features)[0])
        for model_name, model in regressors.items()
    }
    next_direction = int(direction_model.predict(latest_features)[0])


st.subheader(f"{ticker} — latest observed close: {last_close:.2f} ({latest_date})")
price_columns = st.columns(2)
for column, (model_name, predicted_price) in zip(price_columns, next_prices.items()):
    column.metric(
        f"Next close estimate · {model_name}",
        f"{predicted_price:.2f}",
        f"{predicted_price - last_close:+.2f}",
    )

direction_label = "UP" if next_direction == 1 else "NOT UP"
st.metric("Decision Tree direction estimate", direction_label)
st.caption("“Not up” means tomorrow's close was not higher than today's close.")

st.subheader("Historical closing price")
st.line_chart(history.set_index("Date")["Close"])

st.subheader("Test-period model results")
st.caption("The latest 20% of dates are held out for testing; dates are not shuffled.")
st.dataframe(pd.DataFrame(metric_rows).round(3), hide_index=True, width="stretch")
st.metric("Direction classifier test accuracy", f"{direction_accuracy:.1%}")

actual_up_down = confusion_matrix(
    direction_test,
    direction_predictions,
    labels=[0, 1],
)
st.caption("Direction confusion matrix: rows are actual; columns are predicted.")
st.dataframe(
    pd.DataFrame(
        actual_up_down,
        index=["Actual not up", "Actual up"],
        columns=["Predicted not up", "Predicted up"],
    ),
    width="stretch",
)

st.subheader("Actual vs. predicted test prices")
prediction_figure, prediction_axis = plt.subplots(figsize=(12, 4))
prediction_axis.plot(test["Target_Date"], y_test, label="Actual", linewidth=2)
for model_name, predicted in predictions.items():
    prediction_axis.plot(test["Target_Date"], predicted, label=model_name)
prediction_axis.set_xlabel("Trading date")
prediction_axis.set_ylabel("Closing price")
prediction_axis.legend()
prediction_figure.tight_layout()
st.pyplot(prediction_figure)
plt.close(prediction_figure)

st.subheader("Random Forest feature importance")
importance = pd.Series(
    regressors["Random Forest"].feature_importances_,
    index=FEATURES,
).sort_values()
importance_figure, importance_axis = plt.subplots(figsize=(8, 4))
importance.plot(kind="barh", ax=importance_axis, color="teal")
importance_axis.set_xlabel("Relative importance")
importance_figure.tight_layout()
st.pyplot(importance_figure)
plt.close(importance_figure)

st.caption(
    "Historical price patterns are noisy and cannot account for all market "
    "events. These estimates are not final, guaranteed, or trading signals."
)
