"""Consolidated ML trainers: Ridge, Logistic, RF, XGB.

Plus recursive_forecast_14 wrapper reserved for Streamlit (deferred).
Offline eval uses 1-day metrics only.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, roc_auc_score)
from xgboost import XGBClassifier

from src.config import CFG, MODELS_DIR
from src.preprocessing import SplitData


def train_ridge(split: SplitData):
    p = CFG["ml"]["ridge"]
    model = Ridge(alpha=p["alpha"], random_state=42)
    model.fit(split.X_train, split.y_return_train)
    y_pred = model.predict(split.X_test)
    metrics = {
        "test_mae_return": float(mean_absolute_error(split.y_return_test, y_pred)),
        "test_rmse_return": float(np.sqrt(mean_squared_error(split.y_return_test, y_pred))),
    }
    return model, y_pred, metrics


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
    model = XGBClassifier(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"], eval_metric="logloss",
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


def save_ml_models(ticker: str, ridge, logistic, rf, xgb, scaler_X):
    joblib.dump(ridge, MODELS_DIR / f"{ticker}_ridge.pkl")
    joblib.dump(logistic, MODELS_DIR / f"{ticker}_logistic.pkl")
    joblib.dump(rf, MODELS_DIR / f"{ticker}_rf.pkl")
    joblib.dump(xgb, MODELS_DIR / f"{ticker}_xgb.pkl")
    joblib.dump(scaler_X, MODELS_DIR / f"{ticker}_scaler_X.joblib")


# === Recursive multi-step (DEFERRED — for Streamlit only, not used in offline eval) ===

def recursive_forecast_14(model, last_features_row: np.ndarray, horizon: int = 14) -> np.ndarray:
    """Predict horizon steps ahead by feeding each prediction back as the next input.

    NOTE: Naive implementation — does NOT update lag features; assumes feature vector
    represents 'last known state'. For Streamlit demo only. Realistic recursive
    inference would require a feature update function which is beyond this plan's scope.

    Returns: np.ndarray shape (horizon,) of predictions.
    """
    preds = []
    feat = last_features_row.copy().reshape(1, -1)
    for _ in range(horizon):
        p = model.predict(feat)[0]
        preds.append(p)
        # Naive update: shift the close-related feature by predicted return — placeholder
        # Real implementation depends on feature ordering; left as TODO comment for Streamlit task.
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
