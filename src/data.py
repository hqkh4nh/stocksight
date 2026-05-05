"""Data loader: download OHLCV from yfinance and cache as CSV."""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import CFG, DATA_DIR

def download_stock(ticker: str, start_date: str = None, retries: int = 3) -> pd.DataFrame:
    """
    Download OHLCV data for a single ticker from yfinance.

    Args:
        ticker: Stock symbol, e.g. "AAPL".
        start_date: Start date in "YYYY-MM-DD" format. Defaults to value from config.
        retries: Number of retries on network error.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    if start_date is None:
        start_date = CFG["data"]["start_date"]

    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                auto_adjust=True,        # auto-adjust for stock splits and dividends
                progress=False,
            )

            if df.empty:
                print(f"  ⚠ {ticker}: empty data, retry {attempt + 1}/{retries}")
                time.sleep(2)
                continue

            # yfinance sometimes returns MultiIndex columns -> flatten
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Reset index (Date is the index -> turn into a column)
            df = df.reset_index()

            # Keep only the columns we need
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]

            # Sort ascending by date
            df = df.sort_values("date").reset_index(drop=True)

            return df

        except Exception as e:
            print(f"  ✗ {ticker} attempt {attempt + 1}/{retries} error: {e}")
            time.sleep(5 * (attempt + 1))

    return pd.DataFrame()

def cache_path(ticker: str) -> Path:
    """Return the CSV cache path for a ticker."""
    return DATA_DIR / f"{ticker}.csv"

def load_cached(ticker: str) -> pd.DataFrame:
    """Load cached CSV. Raises if not yet cached."""
    path = cache_path(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No cache for {ticker}. Run prepare_data first.")
    df = pd.read_csv(path, parse_dates=["date"])
    return df

def download_all(force: bool = False) -> dict:
    """
    Download all tickers listed in config and save them as CSV.

    Args:
        force: If True, re-download even if a cache exists.

    Returns:
        dict {ticker: rows_count}
    """
    tickers = CFG["tickers"]
    results = {}

    for i, ticker in enumerate(tickers, 1):
        path = cache_path(ticker)

        if path.exists() and not force:
            df = pd.read_csv(path)
            print(f"[{i}/{len(tickers)}] {ticker}: cached ({len(df)} rows) — skip")
            results[ticker] = len(df)
            continue

        print(f"[{i}/{len(tickers)}] {ticker}: downloading...")
        df = download_stock(ticker)

        if df.empty:
            print(f"  {ticker}: failed to download")
            results[ticker] = 0
            continue

        df.to_csv(path, index=False)
        print(f"  {ticker}: saved {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
        results[ticker] = len(df)

    return results

if __name__ == "__main__":
    # Test: download a single ticker
    print("Test download AAPL...")
    df = download_stock("AAPL")
    print(df.head())
    print(f"Total rows: {len(df)}")