"""yfinance fetch + parquet cache for stocks and macro indices.

Caches under data/raw/stocks/{TICKER}.parquet and
data/raw/macro/{label}.parquet so subsequent runs don't re-download.
"""
from pathlib import Path
import pandas as pd
import yfinance as yf

from src.config import CFG

RAW_DIR = Path(CFG["paths"]["data_raw_dir"])
STOCKS_DIR = RAW_DIR / "stocks"
MACRO_DIR = RAW_DIR / "macro"

# yfinance symbol -> short filename label
MACRO_LABELS = {
    "^VIX": "vix",
    "^TNX": "tnx",
    "CL=F": "oil",
    "DX-Y.NYB": "usd",
}


def _fetch(symbol: str, start: str) -> pd.DataFrame:
    # Tai du lieu gia tu Yahoo Finance. auto_adjust=True giup gia OHLC da
    # duoc dieu chinh theo chia co tuc/split, nen phu hop hon khi hoc tu lich su.
    df = yf.download(symbol, start=start, progress=False, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"No data for {symbol} from {start}")
    # yfinance tra ve Date la index; reset_index dua Date thanh cot de cac
    # buoc sau co the merge theo ngay giao dich.
    df = df.reset_index()
    # Chuan hoa ten cot ve chu thuong: Date -> date, Close -> close, ...
    # Neu yfinance tra MultiIndex cot, lay level dau tien roi chuyen chu thuong.
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_stock(ticker: str, refresh: bool = False) -> pd.DataFrame:
    # Cache parquet giup lan chay sau doc tu dia thay vi tai lai qua mang.
    STOCKS_DIR.mkdir(parents=True, exist_ok=True)
    path = STOCKS_DIR / f"{ticker}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = _fetch(ticker, CFG["data"]["start_date"])
    df.to_parquet(path, index=False)
    return df


def load_spy(refresh: bool = False) -> pd.DataFrame:
    return load_stock("SPY", refresh=refresh)


def load_macro(symbol: str, refresh: bool = False) -> pd.DataFrame:
    # Du lieu vi mo nhu VIX, lai suat 10 nam, dau, USD duoc luu rieng trong
    # data/raw/macro voi ten ngan gon de de ghep feature.
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    label = MACRO_LABELS[symbol]
    path = MACRO_DIR / f"{label}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = _fetch(symbol, CFG["data"]["start_date"])
    df.to_parquet(path, index=False)
    return df


def load_all_macro(refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {MACRO_LABELS[s]: load_macro(s, refresh=refresh) for s in CFG["macro_indices"]}


if __name__ == "__main__":
    print("Fetching stocks...")
    for tkr in CFG["tickers"]:
        df = load_stock(tkr)
        print(f"  {tkr}: {len(df)} rows  {df['date'].min().date()} -> {df['date'].max().date()}")
    print("\nFetching macro...")
    for sym in CFG["macro_indices"]:
        df = load_macro(sym)
        print(f"  {sym} ({MACRO_LABELS[sym]}): {len(df)} rows  {df['date'].min().date()} -> {df['date'].max().date()}")
