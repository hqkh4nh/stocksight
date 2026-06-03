import pandas as pd


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """1-day-ahead return target for ML.

    DL targets (14-day return seq) are computed dynamically inside build_dl_sequences,
    not stored as columns, because horizon may change.
    """
    df = df.copy()
    # y_return la bien can du doan cho ML: loi suat ngay mai so voi hom nay.
    # shift(-1) lay close cua ngay tiep theo, vi vay dong cuoi se thanh NaN
    # va duoc dropna o pipeline sau do.
    df["y_return"] = df["close"].shift(-1) / df["close"] - 1
    return df


if __name__ == "__main__":
    from src.data_loader import load_stock

    df = load_stock("AAPL")
    df = make_targets(df).dropna()

    print("shape:", df.shape)
    print(df[["close", "y_return"]].tail())
