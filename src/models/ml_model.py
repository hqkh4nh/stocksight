"""Consolidated ML trainers: Ridge, Logistic, RF, XGB.

Plus recursive_forecast_14 wrapper reserved for Streamlit (deferred).
Offline eval uses 1-day metrics only.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from xgboost import XGBClassifier, XGBRegressor

from src.config import CFG, MODELS_DIR
from src.preprocessing import SplitData


def _reg_metrics(y_true, y_pred):
    return {
        "test_mae_return": float(mean_absolute_error(y_true, y_pred)),
        "test_rmse_return": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_ridge(split: SplitData):
    p = CFG["ml"]["ridge"]
    model = Ridge(alpha=p["alpha"], random_state=42)
    model.fit(split.X_train, split.y_return_train)
    y_pred = model.predict(split.X_test)
    return model, y_pred, _reg_metrics(split.y_return_test, y_pred)


def train_rf_reg(split: SplitData):
    p = CFG["ml"]["random_forest"]
    model = RandomForestRegressor(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        random_state=42, n_jobs=-1,
    )
    model.fit(split.X_train, split.y_return_train)
    y_pred = model.predict(split.X_test)
    return model, y_pred, _reg_metrics(split.y_return_test, y_pred)


def train_xgb_reg(split: SplitData):
    p = CFG["ml"]["xgboost"]
    model = XGBRegressor(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    model.fit(split.X_train, split.y_return_train)
    y_pred = model.predict(split.X_test)
    return model, y_pred, _reg_metrics(split.y_return_test, y_pred)


def train_logistic(split: SplitData):
    p = CFG["ml"]["logistic"]
    model = LogisticRegression(max_iter=p["max_iter"], class_weight=p["class_weight"],
                               random_state=42)
    model.fit(split.X_train, split.y_direction_train)
    y_pred = model.predict(split.X_test)
    y_proba = model.predict_proba(split.X_test)[:, 1]
    return model, y_pred, y_proba, _clf_metrics(split.y_direction_test, y_pred, y_proba)


def train_rf(split: SplitData):
    p = CFG["ml"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"], class_weight=p["class_weight"],
        random_state=42, n_jobs=-1,
    )
    model.fit(split.X_train, split.y_direction_train)
    y_pred = model.predict(split.X_test)
    y_proba = model.predict_proba(split.X_test)[:, 1]
    return model, y_pred, y_proba, _clf_metrics(split.y_direction_test, y_pred, y_proba)


def train_xgb(split: SplitData):
    p = CFG["ml"]["xgboost"]
    pos = int((split.y_direction_train == 1).sum())
    neg = int((split.y_direction_train == 0).sum())
    spw = neg / max(pos, 1)
    model = XGBClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], eval_metric="logloss",
        scale_pos_weight=spw,
        tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    model.fit(split.X_train, split.y_direction_train)
    y_pred = model.predict(split.X_test)
    y_proba = model.predict_proba(split.X_test)[:, 1]
    return model, y_pred, y_proba, _clf_metrics(split.y_direction_test, y_pred, y_proba)


def _clf_metrics(y_true, y_pred, y_proba):
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def save_ml_models(ticker: str, ridge, logistic, rf, xgb, scaler_X,
                   rf_reg=None, xgb_reg=None):
    joblib.dump(ridge, MODELS_DIR / f"{ticker}_ridge.pkl")
    joblib.dump(logistic, MODELS_DIR / f"{ticker}_logistic.pkl")
    joblib.dump(rf, MODELS_DIR / f"{ticker}_rf.pkl")
    joblib.dump(xgb, MODELS_DIR / f"{ticker}_xgb.pkl")
    joblib.dump(scaler_X, MODELS_DIR / f"{ticker}_scaler_X.joblib")
    if rf_reg is not None:
        joblib.dump(rf_reg, MODELS_DIR / f"{ticker}_rf_reg.pkl")
    if xgb_reg is not None:
        joblib.dump(xgb_reg, MODELS_DIR / f"{ticker}_xgb_reg.pkl")


# === Recursive multi-step forecast ===

def recursive_forecast_14(model, last_features_row: np.ndarray, feat_cols: list = None,
                          scaler_X=None, horizon: int = 14) -> np.ndarray:
    """Predict `horizon`-step returns by feeding each prediction back into time-dependent features.

    Updates after each step: `returns`, `log_returns`, `close_pct_lag_1`, `close_pct_lag_5`.
    Other features (RSI, MACD, MA, volatility, macro, sector corrs) are held constant — an
    intentional approximation: short-horizon recursion (~14 days) makes the moving-average drift
    second-order, while building exact handcrafted-feature update kernels would push scope.

    Args:
        model:              fitted sklearn-style estimator with .predict(X)
        last_features_row:  (n_features,) UNSCALED feature vector for anchor day
        feat_cols:          ordered list[str] matching last_features_row positions.
                            If None, falls back to legacy behaviour (no lag updates).
        scaler_X:           optional fitted MinMaxScaler — if provided, applied before model.predict.
        horizon:            number of steps to roll forward.

    Returns:
        np.ndarray (horizon,) of predicted 1-day returns.
    """
    feat_raw = last_features_row.astype(float).copy()
    preds = []

    if feat_cols is None:
        # Legacy behaviour for tests using dummy constant models
        feat = feat_raw.reshape(1, -1)
        for _ in range(horizon):
            preds.append(float(model.predict(feat)[0]))
        return np.array(preds)

    idx = {c: i for i, c in enumerate(feat_cols)}
    # Maintain rolling window of last 5 returns to recompute close_pct_lag_5
    # Seed with current 1-day lag if available, else 0
    recent = [feat_raw[idx["close_pct_lag_1"]]] * 5 if "close_pct_lag_1" in idx else [0.0] * 5

    for _ in range(horizon):
        x = feat_raw.reshape(1, -1)
        if scaler_X is not None:
            x = scaler_X.transform(x)
        r_pred = float(model.predict(x)[0])
        preds.append(r_pred)

        # Roll the recent-returns window
        recent.append(r_pred)
        recent = recent[-5:]

        if "returns" in idx:
            feat_raw[idx["returns"]] = r_pred
        if "log_returns" in idx:
            feat_raw[idx["log_returns"]] = np.log1p(r_pred)
        if "close_pct_lag_1" in idx:
            feat_raw[idx["close_pct_lag_1"]] = r_pred
        if "close_pct_lag_5" in idx:
            # Cumulative 5-day return from rolling window
            cum5 = float(np.prod([1 + r for r in recent]) - 1)
            feat_raw[idx["close_pct_lag_5"]] = cum5

    return np.array(preds)


if __name__ == "__main__":
    from src.preprocessing import prepare_dl_pipeline_v2, build_macro_df
    macro_df = build_macro_df()
    _, split = prepare_dl_pipeline_v2("BKR", macro_df)

    ridge, _, m_r = train_ridge(split)
    log, _, _, m_l = train_logistic(split)
    rf, _, _, m_rf = train_rf(split)
    xgb, _, _, m_x = train_xgb(split)

    print("Ridge   :", m_r)
    print("Logistic:", m_l)
    print("RF      :", m_rf)
    print("XGB     :", m_x)
