# Stock Direction & Price Lab

A beginner-friendly Streamlit dashboard for one stock ticker at a time. It
estimates the next trading day's closing price with Linear Regression and
Random Forest, and predicts whether that close may be **up** or **not up** with
a Decision Tree classifier. Historical data is downloaded from Yahoo Finance.

> Educational estimates only—not final or guaranteed predictions, financial advice, or trading signals.

## Run locally

From the project folder, install packages and start the dashboard:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).
Choose or search by company name or ticker (for example, **Tesla (TSLA)**),
choose a start date, and select **Analyze stock**. Choose **Other (enter ticker)**
to type a different symbol.
The app downloads data automatically, or you can upload a CSV from the sidebar.
The dashboard shows forecasts, a chronological test score table, direction
accuracy/confusion matrix, closing-price chart, and Random Forest feature
importance. It caches downloaded data for one hour.

### Get Kaggle stock data

For one ticker, download and extract the
[Tesla historical dataset](https://www.kaggle.com/datasets/varpit94/tesla-stock-data-updated-till-28jun2021).
Choose its CSV file under **Optional Kaggle CSV**.

To compare five companies, download and extract Kaggle's
[historical data for ten companies](https://www.kaggle.com/datasets/khushipitroda/stock-market-historical-data-of-top-10-companies).
Select **Top 5 from Kaggle** and upload its `data.csv` file. The dashboard
trains separate models for AAPL, MSFT, AMZN, META, and TSLA, then displays
next-close estimates, direction estimates, and time-based test scores. The file
contains daily data from 2013 through July 2023, so those predictions are
historical learning examples, not current forecasts. A simple last-close
baseline is included so the ML results can be compared with a basic reference.

Both CSV uploads are read by the app and do not need to be committed to GitHub.

## Host it from GitHub

Streamlit Community Cloud runs Python apps; GitHub Pages does not. To deploy:

1. Put this project in a GitHub repository. Include `app.py`, `src/`,
   `requirements.txt`, and the folders' `.gitkeep` files. The `.gitignore`
   keeps local environments, downloaded data, and generated outputs out of Git.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. Choose **Create app**, select your repository and branch, set the main file
   to `app.py`, and deploy.
4. Share the Streamlit app URL when deployment finishes.

No API key is required; the app retrieves public historical prices through
`yfinance`. The host needs internet access to fetch stock data.

## Project layout

```text
app.py                  Streamlit web dashboard
src/main.py             Shared features and command-line model
src/data_loader.py      CSV/Yahoo Finance data loader
requirements.txt        App and model dependencies
```

## Dashboard ideas

These public projects are useful for studying dashboard organization and user
flows. This project uses its own implementation and keeps a chronological
train/test split:

- [StockStream](https://github.com/rohitmore1012/StockStream) — ticker selection
  and separating stock views.
- [AI Stock Dashboard](https://github.com/ErikThiart/ai-stock-dashboard) —
  organizing charts, indicators, and prediction sections.
- [Stock Price Prediction](https://github.com/vikasharma005/Stock-Price-Prediction)
  — a larger example of how a prediction project can grow into a platform.
