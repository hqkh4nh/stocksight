import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import CFG, DATA_DIR


def download_stock(ticker, start_date=None, retries=3):
    if start_date is None:
        start_date = CFG["data"]["start_date"]

    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                auto_adjust=True,
                progress=False,
            )

            if df.empty:
                print(f"  {ticker}: empty, retry {attempt + 1}/{retries}")
                time.sleep(2)
                continue

            # yfinance sometimes returns MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df = df.sort_values("date").reset_index(drop=True)
            return df

        except Exception as e:
            print(f"  {ticker} attempt {attempt + 1}/{retries}: {e}")
            time.sleep(5 * (attempt + 1))

    return pd.DataFrame()


def cache_path(ticker):
    return DATA_DIR / f"{ticker}.csv"


def load_cached(ticker):
    path = cache_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No cache for {ticker}. Run prepare_data first.")
    return pd.read_csv(path, parse_dates=["date"])


def download_all(force=False):
    tickers = CFG["tickers"]
    results = {}

    for i, ticker in enumerate(tickers, 1):
        path = cache_path(ticker)

        if path.exists() and not force:
            df = pd.read_csv(path)
            print(f"[{i}/{len(tickers)}] {ticker}: cached ({len(df)} rows) - skip")
            results[ticker] = len(df)
            continue

        print(f"[{i}/{len(tickers)}] {ticker}: downloading...")
        df = download_stock(ticker)

        if df.empty:
            print(f"  {ticker}: failed")
            results[ticker] = 0
            continue

        df.to_csv(path, index=False)
        print(f"  {ticker}: {len(df)} rows ({df['date'].min().date()} - {df['date'].max().date()})")
        results[ticker] = len(df)

    return results


if __name__ == "__main__":
    df = download_stock("AAPL")
    print(df.head())
    print("rows:", len(df))
