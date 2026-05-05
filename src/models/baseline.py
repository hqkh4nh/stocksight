import joblib
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score, roc_auc_score)

from src.config import CFG, MODELS_DIR
from src.preprocessing import SplitData


def train_ridge(split: SplitData):
    params = CFG["ml"]["ridge"]
    model = Ridge(alpha=params["alpha"], random_state=42)
    model.fit(split.X_train, split.y_price_train)

    y_pred_train = model.predict(split.X_train)
    y_pred_test = model.predict(split.X_test)

    metrics = {
        "train_mae": mean_absolute_error(split.y_price_train, y_pred_train),
        "train_rmse": np.sqrt(mean_squared_error(split.y_price_train, y_pred_train)),
        "test_mae": mean_absolute_error(split.y_price_test, y_pred_test),
        "test_rmse": np.sqrt(mean_squared_error(split.y_price_test, y_pred_test)),
        "test_mape": np.mean(np.abs((split.y_price_test - y_pred_test) / split.y_price_test)) * 100,
    }
    return model, y_pred_test, metrics


def train_logistic(split: SplitData):
    params = CFG["ml"]["logistic"]
    model = LogisticRegression(
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        random_state=42,
    )
    model.fit(split.X_train, split.y_direction_train)

    y_pred_test = model.predict(split.X_test)
    y_proba_test = model.predict_proba(split.X_test)[:, 1]

    metrics = {
        "test_accuracy": accuracy_score(split.y_direction_test, y_pred_test),
        "test_f1": f1_score(split.y_direction_test, y_pred_test),
        "test_roc_auc": roc_auc_score(split.y_direction_test, y_proba_test),
    }
    return model, y_pred_test, y_proba_test, metrics


def save_baseline_models(ticker, ridge_model, logistic_model, scaler):
    joblib.dump(ridge_model, MODELS_DIR / f"{ticker}_ridge.pkl")
    joblib.dump(logistic_model, MODELS_DIR / f"{ticker}_logistic.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{ticker}_scaler.pkl")


if __name__ == "__main__":
    from src.preprocessing import prepare_pipeline

    split, _ = prepare_pipeline("AAPL")

    ridge, _, ridge_m = train_ridge(split)
    print("Ridge:", {k: round(float(v), 4) for k, v in ridge_m.items()})

    logistic, _, _, log_m = train_logistic(split)
    print("Logistic:", {k: round(float(v), 4) for k, v in log_m.items()})

    save_baseline_models("AAPL", ridge, logistic, split.scaler)
