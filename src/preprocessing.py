from dataclasses import dataclass
from os.path import split
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS

@dataclass
class SplitData:
    """Container for train/test data after split + scale."""
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

"""Preprocessing: chronological split + feature scaling."""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS


@dataclass
class SplitData:
    """Container for train/test data after split + scale."""
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
    """
    Prepare data for ML models.

    Args:
        df: DataFrame already containing 15 features + 2 targets, with NaN dropped.
        train_ratio: Train ratio. Default 80%.

    Returns:
        SplitData with X_train, X_test, y_*, scaler, ...
    """
    # Sort by date (just to be safe)
    df = df.sort_values("date").reset_index(drop=True)

    # Separate features and targets
    X = df[FEATURE_COLUMNS].values         # shape (N, 15)
    y_price = df["y_price"].values
    y_direction = df["y_direction"].values
    dates = df["date"]

    # Chronological split -- DO NOT SHUFFLE
    split_idx = int(len(df) * train_ratio)
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
    y_direction_train, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]
    train_dates, test_dates = dates[:split_idx], dates[split_idx:]

    # Scale features (fit only on train, transform on test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_price_train=y_price_train,
        y_price_test=y_price_test,
        y_direction_train=y_direction_train,
        y_direction_test=y_direction_test,
        feature_names=FEATURE_COLUMNS,
        scaler=scaler,
        train_dates=train_dates,
        test_dates=test_dates,
    )

def prepare_pipeline(ticker: str, train_ratio: float = 0.8) -> Tuple[SplitData, pd.DataFrame]:
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

    print(f"Total rows after preprocessing: {len(df)}")
    print(f"Train: {len(split.X_train)} rows ({split.train_dates.min().date()} -> {split.train_dates.max().date()})")
    print(f"Test:  {len(split.X_test)} rows ({split.test_dates.min().date()} -> {split.test_dates.max().date()})")
    print(f"\nFeature shape: {split.X_train.shape}")
    print(f"y_price_train range: ${split.y_price_train.min():.2f} -> ${split.y_price_train.max():.2f}")
    print(f"y_direction_train balance: {split.y_direction_train.mean():.3f} (proportion of 1=Up)")

    # Verify scaling: scaled train set should have mean ~0, std ~1
    print(f"\nAfter scaling:")
    print(f"  X_train mean: {split.X_train.mean():.4f} (should be ~0)")
    print(f"  X_train std:  {split.X_train.std():.4f} (should be ~1)")
    print(f"  X_test  mean: {split.X_test.mean():.4f} (close to 0 but not exactly)")