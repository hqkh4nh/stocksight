"""Preprocessing: build feature dataframes, split chronologically, slide windows for DL.

Key choices:
- MinMaxScaler (not StandardScaler) per reference, separate scalers for X and y
- Scalers fit on TRAIN ONLY then transform test (anti-leak)
- Sliding window lookback=60, horizon=14
- DL output shape (N, 14, 1) for return seq target
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import CFG, MODELS_DIR
from src.data_loader import load_stock, load_all_macro
from src.features import compute_features, feature_columns
from src.targets import make_targets_v2


@dataclass
class SplitData:
    """Tabular split for ML (1-day target)."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_return_train: np.ndarray
    y_return_test: np.ndarray
    y_direction_train: np.ndarray
    y_direction_test: np.ndarray
    close_train: np.ndarray
    close_test: np.ndarray
    feature_names: list
    scaler_X: MinMaxScaler
    train_dates: pd.Series
    test_dates: pd.Series


@dataclass
class DLData:
    """Sequence split for DL (14-day return seq target).

    Targets are RAW returns (typically ~ N(0, 0.02)) — kept unscaled so the
    sign-aware directional_loss can actually penalise wrong-sign predictions.
    """
    X_train_seq: np.ndarray             # (N, 60, n_features) scaled
    X_test_seq: np.ndarray
    y_return_seq_train: np.ndarray      # (N, 14, 1) RAW returns
    y_return_seq_test: np.ndarray       # (N, 14, 1) RAW returns
    close_anchor_train: np.ndarray      # (N,) close price at end of input window
    close_anchor_test: np.ndarray
    train_target_dates: pd.Series       # date of last predicted day t+14 for each sequence
    test_target_dates: pd.Series
    feature_names: list
    scaler_X: MinMaxScaler
    window: int
    horizon: int


def build_macro_df() -> pd.DataFrame:
    """Combine all macro indices into one dataframe with pct_change columns."""
    macros = load_all_macro()
    out = macros["vix"][["date"]].copy()
    out["vix_change"] = macros["vix"]["close"].pct_change()
    out["tnx_change"] = macros["tnx"]["close"].pct_change()
    out["oil_change"] = macros["oil"]["close"].pct_change()
    out["usd_change"] = macros["usd"]["close"].pct_change()
    return out


def prepare_ml_split(df: pd.DataFrame, feat_cols: list, train_ratio: float,
                     split_idx: int = None) -> SplitData:
    """Build the 1-day tabular split. If `split_idx` is given, use it directly so the ML
    test set begins on the SAME row as the first DL test anchor — making baseline
    comparisons strictly apples-to-apples. Otherwise fall back to `train_ratio`."""
    df = df.sort_values("date").reset_index(drop=True)
    X = df[feat_cols].values.astype(np.float32)
    y_return = df["y_return"].values.astype(np.float32)
    y_direction = df["y_direction"].values.astype(np.int32)
    close = df["close"].values.astype(np.float32)
    dates = df["date"]

    n = len(df)
    if split_idx is None:
        split_idx = int(n * train_ratio)
    split_idx = max(1, min(split_idx, n - 1))

    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X[:split_idx])
    X_test = scaler_X.transform(X[split_idx:])

    return SplitData(
        X_train=X_train,
        X_test=X_test,
        y_return_train=y_return[:split_idx],
        y_return_test=y_return[split_idx:],
        y_direction_train=y_direction[:split_idx],
        y_direction_test=y_direction[split_idx:],
        close_train=close[:split_idx],
        close_test=close[split_idx:],
        feature_names=feat_cols,
        scaler_X=scaler_X,
        train_dates=dates.iloc[:split_idx].reset_index(drop=True),
        test_dates=dates.iloc[split_idx:].reset_index(drop=True),
    )


def build_dl_sequences(df: pd.DataFrame, feat_cols: list, window: int, horizon: int,
                       train_ratio: float, ticker: str) -> tuple[DLData, SplitData]:
    """Build (X, y_return_seq) sliding windows.

    For each anchor index i, X[i] = features[i-window+1 .. i] (60 days),
    y_return_seq[i] = returns[i+1 .. i+horizon] (14 days ahead).
    Anchor close = close[i].
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)

    # Need at least window past + horizon future
    valid_idx = np.arange(window - 1, n - horizon)
    if len(valid_idx) <= 0:
        raise ValueError(f"Not enough rows ({n}) for window={window} + horizon={horizon}")

    n_train = int(len(valid_idx) * train_ratio)

    # Fit scalers on train slice of features and returns
    train_anchor_indices = valid_idx[:n_train]
    train_feat_rows = np.unique(np.concatenate([
        np.arange(i - window + 1, i + 1) for i in train_anchor_indices
    ]))
    train_feat_rows = train_feat_rows[(train_feat_rows >= 0) & (train_feat_rows < n)]

    scaler_X = MinMaxScaler()
    scaler_X.fit(df.iloc[train_feat_rows][feat_cols].values.astype(np.float32))

    # Build sequences for ALL valid anchors. Targets are RAW returns.
    X_full = scaler_X.transform(df[feat_cols].values.astype(np.float32))
    n_features = X_full.shape[1]

    X_seq = np.zeros((len(valid_idx), window, n_features), dtype=np.float32)
    y_ret_seq = np.zeros((len(valid_idx), horizon, 1), dtype=np.float32)
    close_anchor = np.zeros(len(valid_idx), dtype=np.float32)
    target_dates = []
    for k, i in enumerate(valid_idx):
        X_seq[k] = X_full[i - window + 1: i + 1]
        future_returns = df["returns"].iloc[i + 1: i + 1 + horizon].values.astype(np.float32)
        y_ret_seq[k, :, 0] = future_returns
        close_anchor[k] = df["close"].iloc[i]
        target_dates.append(df["date"].iloc[i + horizon])

    target_dates = pd.Series(target_dates).reset_index(drop=True)

    # Persist X scaler only (no scaler_y — targets are raw)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, MODELS_DIR / f"{ticker}_scaler_X.joblib")

    dl = DLData(
        X_train_seq=X_seq[:n_train],
        X_test_seq=X_seq[n_train:],
        y_return_seq_train=y_ret_seq[:n_train],
        y_return_seq_test=y_ret_seq[n_train:],
        close_anchor_train=close_anchor[:n_train],
        close_anchor_test=close_anchor[n_train:],
        train_target_dates=target_dates.iloc[:n_train].reset_index(drop=True),
        test_target_dates=target_dates.iloc[n_train:].reset_index(drop=True),
        feature_names=feat_cols,
        scaler_X=scaler_X,
        window=window,
        horizon=horizon,
    )

    # Sync ML split with DL anchors: ML test begins on the same row as the first DL test anchor.
    # First test anchor row = valid_idx[n_train]; that row's `y_return` (next-day) is the ML target.
    ml_split_idx = int(valid_idx[n_train]) if n_train < len(valid_idx) else int(valid_idx[-1])
    split = prepare_ml_split(df, feat_cols, train_ratio, split_idx=ml_split_idx)
    return dl, split


def prepare_dl_pipeline_v2(ticker: str, macro_df: pd.DataFrame,
                           window: int = 60, horizon: int = 14,
                           train_ratio: float = 0.85) -> tuple[DLData, SplitData]:
    stock = load_stock(ticker)
    feats = compute_features(stock, macro_df, ticker=ticker)
    feats = make_targets_v2(feats)
    feats = feats.dropna().reset_index(drop=True)
    feat_cols = feature_columns(ticker)
    return build_dl_sequences(feats, feat_cols, window, horizon, train_ratio, ticker)


if __name__ == "__main__":
    macro_df = build_macro_df()
    for tkr in CFG["tickers"][:3]:
        dl, split = prepare_dl_pipeline_v2(tkr, macro_df)
        print(f"{tkr}: X_train_seq={dl.X_train_seq.shape}  y_seq_train={dl.y_return_seq_train.shape}")
