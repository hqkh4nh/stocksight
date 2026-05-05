from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_price_train: np.ndarray
    y_price_test: np.ndarray
    y_direction_train: np.ndarray
    y_direction_test: np.ndarray
    feature_names: list
    scaler: StandardScaler
    train_dates: pd.Series
    test_dates: pd.Series


def prepare_ml_data(df: pd.DataFrame, train_ratio: float = 0.80) -> SplitData:
    df = df.sort_values("date").reset_index(drop=True)

    X = df[FEATURE_COLUMNS].values
    y_price = df["y_price"].values
    y_direction = df["y_direction"].values
    dates = df["date"]

    # Chronological split - DO NOT SHUFFLE
    split_idx = int(len(df) * train_ratio)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
    y_dir_train, y_dir_test = y_direction[:split_idx], y_direction[split_idx:]
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]

    # fit ONLY on train to avoid leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_price_train=y_price_train,
        y_price_test=y_price_test,
        y_direction_train=y_dir_train,
        y_direction_test=y_dir_test,
        feature_names=FEATURE_COLUMNS,
        scaler=scaler,
        train_dates=train_dates,
        test_dates=test_dates,
    )


def prepare_pipeline(ticker, train_ratio=0.8):
    from src.data import load_cached
    from src.features import compute_features
    from src.targets import make_targets

    df = load_cached(ticker)
    df = compute_features(df)
    df = make_targets(df)
    df = df.dropna().reset_index(drop=True)

    split = prepare_ml_data(df, train_ratio=train_ratio)
    return split, df


if __name__ == "__main__":
    split, df = prepare_pipeline("AAPL")

    print(f"total: {len(df)}")
    print(f"train: {len(split.X_train)} ({split.train_dates.min().date()} - {split.train_dates.max().date()})")
    print(f"test:  {len(split.X_test)} ({split.test_dates.min().date()} - {split.test_dates.max().date()})")
    print(f"X_train mean/std: {split.X_train.mean():.4f} / {split.X_train.std():.4f}")
    print(f"y_direction up%: {split.y_direction_train.mean():.3f}")
