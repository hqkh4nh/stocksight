"""Feature engineering: common technicals + macro changes + sector-aware correlations."""
import numpy as np
import pandas as pd

from src.sector_config import sector_for, OIL_CORR_SECTORS, RATE_CORR_SECTORS


# === Group A: common technical (15 features) ===

def add_returns(df):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_lag_features(df):
    """Relative (stationary) close lags: today vs N days ago, expressed as % change.
    Replaces raw price-level lags which were OOD on the test set after the long backfill."""
    df = df.copy()
    df["close_pct_lag_1"] = df["close"] / df["close"].shift(1) - 1
    df["close_pct_lag_5"] = df["close"] / df["close"].shift(5) - 1
    return df


def add_moving_averages(df):
    df = df.copy()
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    return df


def add_rsi(df, period=14):
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    df = df.copy()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df


def add_volatility(df, window=21):
    df = df.copy()
    df["volatility_21"] = df["returns"].rolling(window).std()
    return df


def add_bollinger_bands(df, window=20, num_std=2.0):
    df = df.copy()
    bb_mid = df["close"].rolling(window).mean()
    bb_std = df["close"].rolling(window).std()
    df["bb_width"] = ((bb_mid + num_std * bb_std) - (bb_mid - num_std * bb_std)) / bb_mid
    return df


def add_volume_ratio(df, window=20):
    df = df.copy()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(window).mean()
    return df


# === Group B: macro changes (4 features, joined from macro df dict) ===

def add_macro_features(df, macro_df):
    """macro_df is a single dataframe with cols: date, vix_change, tnx_change, oil_change, usd_change."""
    df = df.copy()
    return df.merge(macro_df, on="date", how="left")


# === Group C: sector-aware correlations (1-2 features) ===

def add_oil_correlation(df, oil_returns, window=21):
    """Stock_Oil_Corr_21 + Vol_x_Oil. Requires oil_returns Series aligned to df.date."""
    df = df.copy()
    df["stock_oil_corr_21"] = df["returns"].rolling(window).corr(oil_returns)
    df["vol_x_oil"] = df["volatility_21"] * oil_returns.abs()
    return df


def add_rate_correlation(df, rate_returns, window=21):
    """Stock_Rate_Corr_21 only."""
    df = df.copy()
    df["stock_rate_corr_21"] = df["returns"].rolling(window).corr(rate_returns)
    return df


# === Public API ===

FEATURE_COLUMNS_BASE = [
    "close", "returns", "log_returns", "close_pct_lag_1", "close_pct_lag_5",
    "ma_5", "ma_20", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal",
    "volatility_21", "bb_width", "volume_ratio",
]

MACRO_COLUMNS = ["vix_change", "tnx_change", "oil_change", "usd_change"]


def feature_columns(ticker: str) -> list[str]:
    """Final ordered feature list for this ticker (drops 'close' since it's the price target reference)."""
    cols = [c for c in FEATURE_COLUMNS_BASE if c != "close"] + MACRO_COLUMNS
    sector = sector_for(ticker)
    if sector in OIL_CORR_SECTORS:
        cols += ["stock_oil_corr_21", "vol_x_oil"]
    elif sector in RATE_CORR_SECTORS:
        cols += ["stock_rate_corr_21"]
    return cols


def compute_features(stock_df: pd.DataFrame, macro_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Apply all feature groups. macro_df must have cols: date, vix_change, tnx_change, oil_change, usd_change."""
    df = stock_df.copy().sort_values("date").reset_index(drop=True)
    df = add_returns(df)
    df = add_lag_features(df)
    df = add_moving_averages(df)
    df = add_rsi(df, period=14)
    df = add_macd(df)
    df = add_volatility(df, window=21)
    df = add_bollinger_bands(df, window=20, num_std=2.0)
    df = add_volume_ratio(df, window=20)
    df = add_macro_features(df, macro_df)

    sector = sector_for(ticker)
    if sector in OIL_CORR_SECTORS:
        df = add_oil_correlation(df, df["oil_change"], window=21)
    elif sector in RATE_CORR_SECTORS:
        df = add_rate_correlation(df, df["tnx_change"], window=21)

    return df


if __name__ == "__main__":
    from src.data_loader import load_stock, load_all_macro

    macro = load_all_macro()
    macro_df = macro["vix"][["date"]].copy()
    macro_df["vix_change"] = macro["vix"]["close"].pct_change()
    macro_df["tnx_change"] = macro["tnx"]["close"].pct_change()
    macro_df["oil_change"] = macro["oil"]["close"].pct_change()
    macro_df["usd_change"] = macro["usd"]["close"].pct_change()

    stock = load_stock("BKR")
    feats = compute_features(stock, macro_df, ticker="BKR")
    feats = feats.dropna()
    print(f"BKR: {feats.shape}  cols={feature_columns('BKR')}")
