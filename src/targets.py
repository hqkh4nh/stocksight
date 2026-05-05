import pandas as pd

def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 2 targets:
      - y_price: tomorrow's close price (regression)
      - y_direction: 1 if price goes up, 0 if it goes down (classification)

    Args:
        df: DataFrame with a 'close' column.

    Returns:
        DataFrame with 2 extra columns: y_price, y_direction.
        Warning: the last row will be NaN (we don't know tomorrow's price) -- drop before training.
    """
    df = df.copy()
    df["y_price"] = df["close"].shift(-1)
    returns_next = df["close"].shift(-1) / df["close"] - 1
    df["y_direction"] = (returns_next > 0).astype(int)
    return df

if __name__ == "__main__":
    from src.data import load_cached
    from src.features import compute_features

    df = load_cached("AAPL")
    df = compute_features(df)
    df = make_targets(df)
    df = df.dropna()

    print(f"Final shape: {df.shape}")
    print(f"\nDistribution of y_direction (0=Down, 1=Up):")
    print(df["y_direction"].value_counts(normalize=True).sort_index())

    print(f"\nSample (last 5 rows):")
    print(df[["date", "close", "y_price", "y_direction"]].tail())