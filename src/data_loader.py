"""Load historical stock prices from a CSV file or Yahoo Finance."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _standardize_columns(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a frame with the expected, consistently named stock columns."""
    # Recent yfinance versions may return a two-level column index.
    if isinstance(data.columns, pd.MultiIndex):
        ticker_level = next(
            (
                level
                for level in range(data.columns.nlevels)
                if ticker in data.columns.get_level_values(level)
            ),
            None,
        )
        if ticker_level is not None:
            data = data.xs(ticker, axis=1, level=ticker_level)
        elif data.columns.nlevels == 2 and data.columns.get_level_values(1).nunique() == 1:
            data.columns = data.columns.get_level_values(0)
        else:
            raise ValueError(
                "Could not identify the ticker in the downloaded column headers."
            )

    data = data.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]

    # yfinance places dates in the index, while CSV files usually have a Date column.
    if "date" in data.columns:
        data = data.rename(columns={"date": "Date"})
    elif isinstance(data.index, pd.DatetimeIndex) or (
        data.index.name and data.index.name.lower() in {"date", "datetime"}
    ):
        index_name = data.index.name or "index"
        data = data.reset_index().rename(columns={index_name: "Date"})
    else:
        raise ValueError("The data must include a 'Date' column or date index.")

    rename_map = {column.lower(): column for column in REQUIRED_COLUMNS[1:]}
    return data.rename(columns=rename_map)


def normalize_stock_data(raw_data: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """Validate and normalize an in-memory stock-price table."""
    data = _standardize_columns(raw_data, ticker)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data]
    if missing_columns:
        raise ValueError(
            "Stock data is missing required columns: "
            + ", ".join(missing_columns)
        )

    data = data[REQUIRED_COLUMNS].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    for column in REQUIRED_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values("Date").drop_duplicates(subset="Date", keep="last")
    data = data.reset_index(drop=True)
    if data.empty:
        raise ValueError("The stock data contains no rows with valid dates.")

    return data


def normalize_multi_stock_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Kaggle table containing daily prices for several companies."""
    data = raw_data.copy()
    data.columns = [str(column).strip().lower() for column in data.columns]
    rename_map = {
        "company": "Ticker",
        "ticker": "Ticker",
        "symbol": "Ticker",
        "date": "Date",
        "close/last": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    }
    data = data.rename(columns=rename_map)
    required_columns = ["Ticker", *REQUIRED_COLUMNS]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(
            "Multi-stock data is missing required columns: "
            + ", ".join(missing_columns)
        )

    data = data[required_columns].copy()
    data["Ticker"] = data["Ticker"].astype(str).str.strip().str.upper()
    data["Date"] = pd.to_datetime(data["Date"], format="mixed", errors="coerce")
    for column in REQUIRED_COLUMNS[1:]:
        values = data[column].astype(str)
        values = values.str.replace("$", "", regex=False).str.replace(",", "", regex=False)
        data[column] = pd.to_numeric(values, errors="coerce")

    data = data.dropna(subset=required_columns)
    data = data[data["Ticker"] != ""]
    data = (
        data.sort_values(["Ticker", "Date"])
        .drop_duplicates(subset=["Ticker", "Date"], keep="last")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError("The multi-stock CSV contains no valid stock-price rows.")

    return data


def load_stock_data(
    csv_path: str | Path | None = None,
    ticker: str = "AAPL",
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Load stock data from a local CSV, or download it from Yahoo Finance.

    Args:
        csv_path: Optional path to a CSV containing Date, Open, High, Low,
            Close, and Volume columns. If omitted, data is downloaded.
        ticker: Yahoo Finance ticker symbol used when downloading.
        start: First date to download, in YYYY-MM-DD format.
        end: Optional end date to download, in YYYY-MM-DD format.

    Returns:
        A DataFrame with Date, Open, High, Low, Close, and Volume columns,
        sorted in ascending date order.
    """
    if csv_path is not None:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Stock CSV file was not found: {path}")
        raw_data = pd.read_csv(path)
    else:
        import yfinance as yf

        raw_data = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )
        if raw_data.empty:
            raise ValueError(
                f"No stock data was downloaded for {ticker!r}. "
                "Check the ticker symbol and date range."
            )

    return normalize_stock_data(raw_data, ticker)
