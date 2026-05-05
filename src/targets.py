import pandas as pd


def make_targets(df):
    # y_price: tomorrow's close (regression)
    # y_direction: 1 if up, 0 if down (classification)
    # last row will be NaN -> drop before training
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
    df = make_targets(df).dropna()

    print("shape:", df.shape)
    print("y_direction balance:")
    print(df["y_direction"].value_counts(normalize=True).sort_index())
    print(df[["date", "close", "y_price", "y_direction"]].tail())
