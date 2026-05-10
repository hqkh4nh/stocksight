import pandas as pd


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """1-day-ahead targets for ML.

    DL targets (14-day return seq) are computed dynamically inside build_dl_sequences,
    not stored as columns, because horizon may change.
    """
    df = df.copy()
    df["y_return"] = df["close"].shift(-1) / df["close"] - 1
    df["y_direction"] = (df["y_return"] > 0).astype(int)
    return df


if __name__ == "__main__":
    from src.data_loader import load_stock

    df = load_stock("AAPL")
    df = make_targets(df).dropna()

    print("shape:", df.shape)
    print("y_direction balance:")
    print(df["y_direction"].value_counts(normalize=True).sort_index())
    print(df[["close", "y_return", "y_direction"]].tail())
