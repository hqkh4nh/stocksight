import numpy as np
import pandas as pd


# === Price & Returns ===

def add_returns(df):
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    return df


def add_lag_features(df):
    df = df.copy()
    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_5"] = df["close"].shift(5)
    return df


# === Moving averages ===

def add_moving_averages(df):
    df = df.copy()
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    return df


# === Momentum ===

def add_rsi(df, period=14):
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df):
    # needs ema_12 and ema_26 already computed
    df = df.copy()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df


# === Volatility & volume ===

def add_volatility(df, window=21):
    df = df.copy()
    df["volatility_21"] = df["returns"].rolling(window=window).std()
    return df


def add_bollinger_bands(df, window=20, num_std=2.0):
    df = df.copy()
    bb_mid = df["close"].rolling(window=window).mean()
    bb_std = df["close"].rolling(window=window).std()
    bb_upper = bb_mid + num_std * bb_std
    bb_lower = bb_mid - num_std * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    return df


def add_volume_ratio(df, window=20):
    df = df.copy()
    volume_ma = df["volume"].rolling(window=window).mean()
    df["volume_ratio"] = df["volume"] / volume_ma
    return df


FEATURE_COLUMNS = [
    "close", "returns", "log_returns", "close_lag_1", "close_lag_5",
    "ma_5", "ma_20", "ema_12", "ema_26",
    "rsi_14", "macd", "macd_signal",
    "volatility_21", "bb_width", "volume_ratio",
]


def compute_features(df):
    # Order matters: macd needs ema, volatility needs returns
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = add_returns(df)
    df = add_lag_features(df)
    df = add_moving_averages(df)
    df = add_rsi(df, period=14)
    df = add_macd(df)
    df = add_volatility(df, window=21)
    df = add_bollinger_bands(df, window=20, num_std=2.0)
    df = add_volume_ratio(df, window=20)
    return df


if __name__ == "__main__":
    from src.data import load_cached

    df = load_cached("AAPL")
    df_feat = compute_features(df)
    df_clean = df_feat.dropna()

    print("before:", df.shape, "after:", df_feat.shape, "clean:", df_clean.shape)
    print("rsi range:", round(df_clean["rsi_14"].min(), 2), "-", round(df_clean["rsi_14"].max(), 2))
    print("returns mean:", round(df_clean["returns"].mean(), 6))
    print("volume_ratio mean:", round(df_clean["volume_ratio"].mean(), 4))
