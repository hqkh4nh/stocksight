"""Preprocessing: build features, chronological split, sliding windows for DL.

MinMaxScaler fit on train only. Window=60, horizon=14, DL output shape (N, 14, 1).
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import CFG
from src.data_loader import load_stock, load_all_macro
from src.features import compute_features, feature_columns
from src.targets import make_targets


@dataclass
class SplitData:
    """Tabular split for ML (1-day return target).

    The validation slice is the last `val_ratio` of the training rows (kept
    chronological so it mirrors a hold-out window). It is reported alongside
    test metrics; the model is still fit on the full train slice.
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_return_train: np.ndarray
    y_return_test: np.ndarray
    close_train: np.ndarray
    close_test: np.ndarray
    feature_names: list
    scaler_X: MinMaxScaler
    train_dates: pd.Series
    test_dates: pd.Series
    X_val: np.ndarray
    y_return_val: np.ndarray
    val_dates: pd.Series


@dataclass
class DLData:
    """Sequence split for DL (14-day return seq target).

    RAW returns kept unscaled so directional_loss can penalise wrong sign.
    """
    X_train_seq: np.ndarray             # (N, 60, n_features) scaled
    X_test_seq: np.ndarray
    y_return_seq_train: np.ndarray      # (N, 14, 1) RAW returns
    y_return_seq_test: np.ndarray       # (N, 14, 1) RAW returns
    close_anchor_train: np.ndarray      # (N,) close price at end of input window
    close_anchor_test: np.ndarray
    train_target_dates: pd.Series       # date of last predicted day t+14 for each sequence
    test_target_dates: pd.Series
    train_anchor_dates: pd.Series       # date of anchor (last input day) for each sequence
    test_anchor_dates: pd.Series
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
                     split_idx: int = None, val_ratio: float = 0.10) -> SplitData:
    """Build the 1-day tabular split with a chronological validation slice.

    The last `val_ratio` rows of the training partition become the validation set
    (still inside the train MinMax range, fit on full train).
    """
    df = df.sort_values("date").reset_index(drop=True)
    X = df[feat_cols].values.astype(np.float32)
    y_return = df["y_return"].values.astype(np.float32)
    close = df["close"].values.astype(np.float32)
    dates = df["date"]

    n = len(df)
    if split_idx is None:
        split_idx = int(n * train_ratio)
    split_idx = max(1, min(split_idx, n - 1))

    scaler_X = MinMaxScaler()
    X_train_full_scaled = scaler_X.fit_transform(X[:split_idx])
    X_test = scaler_X.transform(X[split_idx:])

    val_size = max(1, int(split_idx * val_ratio))
    train_end = split_idx - val_size

    return SplitData(
        X_train=X_train_full_scaled[:train_end],
        X_test=X_test,
        y_return_train=y_return[:train_end],
        y_return_test=y_return[split_idx:],
        close_train=close[:train_end],
        close_test=close[split_idx:],
        feature_names=feat_cols,
        scaler_X=scaler_X,
        train_dates=dates.iloc[:train_end].reset_index(drop=True),
        test_dates=dates.iloc[split_idx:].reset_index(drop=True),
        X_val=X_train_full_scaled[train_end:split_idx],
        y_return_val=y_return[train_end:split_idx],
        val_dates=dates.iloc[train_end:split_idx].reset_index(drop=True),
    )


def build_dl_sequences(df: pd.DataFrame, feat_cols: list, window: int, horizon: int,
                       train_ratio: float) -> tuple[DLData, SplitData]:
    """Build (X, y_return_seq) sliding windows.

    For each anchor index i, X[i] = features[i-window+1 .. i] (60 days),
    y_return_seq[i] = returns[i+1 .. i+horizon] (14 days ahead).
    Anchor close = close[i].
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)

    valid_idx = np.arange(window - 1, n - horizon)
    if len(valid_idx) <= 0:
        raise ValueError(f"Not enough rows ({n}) for window={window} + horizon={horizon}")

    n_train = int(len(valid_idx) * train_ratio)

    train_anchor_indices = valid_idx[:n_train]
    train_feat_rows = np.unique(np.concatenate([
        np.arange(i - window + 1, i + 1) for i in train_anchor_indices
    ]))
    train_feat_rows = train_feat_rows[(train_feat_rows >= 0) & (train_feat_rows < n)]

    scaler_X = MinMaxScaler()
    scaler_X.fit(df.iloc[train_feat_rows][feat_cols].values.astype(np.float32))

    X_full = scaler_X.transform(df[feat_cols].values.astype(np.float32))
    n_features = X_full.shape[1]

    X_seq = np.zeros((len(valid_idx), window, n_features), dtype=np.float32)
    y_ret_seq = np.zeros((len(valid_idx), horizon, 1), dtype=np.float32)
    close_anchor = np.zeros(len(valid_idx), dtype=np.float32)
    target_dates = []
    anchor_dates = []
    for k, i in enumerate(valid_idx):
        X_seq[k] = X_full[i - window + 1: i + 1]
        future_returns = df["returns"].iloc[i + 1: i + 1 + horizon].values.astype(np.float32)
        y_ret_seq[k, :, 0] = future_returns
        close_anchor[k] = df["close"].iloc[i]
        anchor_dates.append(df["date"].iloc[i])
        target_dates.append(df["date"].iloc[i + horizon])

    target_dates = pd.Series(target_dates).reset_index(drop=True)
    anchor_dates = pd.Series(anchor_dates).reset_index(drop=True)

    dl = DLData(
        X_train_seq=X_seq[:n_train],
        X_test_seq=X_seq[n_train:],
        y_return_seq_train=y_ret_seq[:n_train],
        y_return_seq_test=y_ret_seq[n_train:],
        close_anchor_train=close_anchor[:n_train],
        close_anchor_test=close_anchor[n_train:],
        train_target_dates=target_dates.iloc[:n_train].reset_index(drop=True),
        test_target_dates=target_dates.iloc[n_train:].reset_index(drop=True),
        train_anchor_dates=anchor_dates.iloc[:n_train].reset_index(drop=True),
        test_anchor_dates=anchor_dates.iloc[n_train:].reset_index(drop=True),
        feature_names=feat_cols,
        scaler_X=scaler_X,
        window=window,
        horizon=horizon,
    )

    ml_split_idx = int(valid_idx[n_train]) if n_train < len(valid_idx) else int(valid_idx[-1])
    split = prepare_ml_split(df, feat_cols, train_ratio, split_idx=ml_split_idx)
    return dl, split


def prepare_dl_pipeline(ticker: str, macro_df: pd.DataFrame,
                        window: int = 60, horizon: int = 14,
                        train_ratio: float = 0.85) -> tuple[DLData, SplitData]:
    stock = load_stock(ticker)
    feats = compute_features(stock, macro_df, ticker=ticker)
    feats = make_targets(feats)
    feats = feats.dropna().reset_index(drop=True)
    feat_cols = feature_columns(ticker)
    return build_dl_sequences(feats, feat_cols, window, horizon, train_ratio)


if __name__ == "__main__":
    macro_df = build_macro_df()
    for tkr in CFG["tickers"][:3]:
        dl, split = prepare_dl_pipeline(tkr, macro_df)
        print(f"{tkr}: X_train_seq={dl.X_train_seq.shape}  y_seq_train={dl.y_return_seq_train.shape}")
        print(f"     ml train={split.X_train.shape}  val={split.X_val.shape}  test={split.X_test.shape}")
