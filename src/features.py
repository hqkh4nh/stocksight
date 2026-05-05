import numpy as np
import pandas as pd

# Group 1: Price & Returns (4 features)

def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """returns = pct_change of close. log_returns = log(close[t]/close[t-1])."""
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """close_lag_1, close_lag_5: closing price N days ago."""
    df = df.copy()
    df["close_lag_1"] = df["close"].shift(1)
    df["close_lag_5"] = df["close"].shift(5)
    return df

# Group 2: Moving Averages (4 features)

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """ma_5, ma_20: simple moving average. ema_12, ema_26: exponential."""
    df = df.copy()
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    return df

# Group 3: Momentum (3 features)

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index): overbought/oversold indicator.
    RSI > 70: overbought. RSI < 30: oversold.
    Formula: 100 - 100 / (1 + avg_gain/avg_loss)
    """
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    MACD = EMA(12) - EMA(26)
    MACD signal = EMA(9) of MACD
    """
    df = df.copy()
    # Requires ema_12 and ema_26
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    return df

# Group 4: Volatility & Volume (3 features)

def add_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """volatility_21 = std of returns over 21 days."""
    df = df.copy()
    df["volatility_21"] = df["returns"].rolling(window=window).std()
    return df

def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands:
      bb_mid = MA(20)
      bb_upper = bb_mid + 2*std
      bb_lower = bb_mid - 2*std
      bb_width = (bb_upper - bb_lower) / bb_mid  <- the actual feature we use
    """
    df = df.copy()
    bb_mid = df["close"].rolling(window=window).mean()
    bb_std = df["close"].rolling(window=window).std()
    bb_upper = bb_mid + num_std * bb_std
    bb_lower = bb_mid - num_std * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    return df

def add_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """volume_ratio = volume / volume_ma_20. >1 means abnormally high volume."""
    df = df.copy()
    volume_ma = df["volume"].rolling(window=window).mean()
    df["volume_ratio"] = df["volume"] / volume_ma
    return df

# Compose

# Final list of all features (ordered by group)
FEATURE_COLUMNS = [
    # Price & Returns
    "close", "returns", "log_returns", "close_lag_1", "close_lag_5",
    # Moving Averages
    "ma_5", "ma_20", "ema_12", "ema_26",
    # Momentum
    "rsi_14", "macd", "macd_signal",
    # Volatility & Volume
    "volatility_21", "bb_width", "volume_ratio",
]

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all 15 features.

    Call order matters:
      1. add_returns -> needed by volatility, lag
      2. add_lag_features -> independent
      3. add_moving_averages -> needed by macd
      4. add_rsi -> independent
      5. add_macd -> needs ema_12, ema_26
      6. add_volatility -> needs returns
      7. add_bollinger_bands -> independent
      8. add_volume_ratio -> independent

    Args:
        df: DataFrame with columns date, open, high, low, close, volume.

    Returns:
        DataFrame with 14 additional feature columns (besides the existing 'close').
        Keeps 'date' as either index or column depending on your usage.
        Warning: NaN in early rows (due to rolling windows) -- not yet dropped.
    """
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
    # Test
    from src.data import load_cached

    df = load_cached("AAPL")
    print("Before features:", df.shape, "columns:", df.columns.tolist())

    df_feat = compute_features(df)
    print("After features:", df_feat.shape)
    print("Feature columns added:", [c for c in df_feat.columns if c not in df.columns])

    # Drop NaN caused by rolling windows
    df_clean = df_feat.dropna()
    print(f"After dropna: {df_feat.shape[0]} -> {df_clean.shape[0]} rows ({df_feat.shape[0] - df_clean.shape[0]} early rows lost)")

    # Sanity check
    print("\nSanity checks:")
    print(f"  RSI range: {df_clean['rsi_14'].min():.2f} -> {df_clean['rsi_14'].max():.2f}  (should be 0-100)")
    print(f"  MACD mean: {df_clean['macd'].mean():.4f}  (should be near 0)")
    print(f"  Returns mean: {df_clean['returns'].mean():.6f}  (should be near 0)")
    print(f"  Volume_ratio mean: {df_clean['volume_ratio'].mean():.4f}  (should be near 1)")
