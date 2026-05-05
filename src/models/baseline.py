"""ML Model 1: Ridge (regression) + Logistic Regression (classification)."""
import joblib
from pathlib import Path

import numpy as np
from scipy.stats import Logistic
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, accuracy_score, f1_score, roc_auc_score)

from src.config import CFG, MODELS_DIR
from src.preprocessing import SplitData

def train_ridge(split: SplitData) -> tuple:
    """Train Ridge regression for price prediction."""
    params = CFG["ml"]["ridge"]
    model = Ridge(alpha=params["alpha"], random_state=42)
    model.fit(split.X_train, split.y_price_train)

    # Predict
    y_pred_train = model.predict(split.X_train)
    y_pred_test = model.predict(split.X_test)

    # Metrics
    metrics = {
        "model": "Ridge",
        "task": "regression",
        "train_mae": mean_absolute_error(split.y_price_train, y_pred_train),
        "train_rmse": np.sqrt(mean_squared_error(split.y_price_train, y_pred_train)),
        "test_mae": mean_absolute_error(split.y_price_test, y_pred_test),
        "test_rmse": np.sqrt(mean_squared_error(split.y_price_test, y_pred_test)),
        "test_mape": np.mean(np.abs((split.y_price_test - y_pred_test) / split.y_price_test)) * 100,
    }

    return model, y_pred_test, metrics

def train_logistic(split: SplitData) -> tuple:
    """Train Logistic Regression for direction classification."""
    params = CFG["ml"]["logistic"]
    model = LogisticRegression(
        max_iter=params["max_iter"],
        class_weight=params["class_weight"],
        random_state=42,
    )
    model.fit(split.X_train, split.y_direction_train)

    # Predict
    y_pred_test = model.predict(split.X_test)
    y_proba_test = model.predict_proba(split.X_test)[:, 1]    # P(Up)

    # Metrics
    metrics = {
        "model": "Logistic",
        "task": "classification",
        "test_accuracy": accuracy_score(split.y_direction_test, y_pred_test),
        "test_f1": f1_score(split.y_direction_test, y_pred_test),
        "test_roc_auc": roc_auc_score(split.y_direction_test, y_proba_test),
    }

    return model, y_pred_test, y_proba_test, metrics

def save_baseline_models(ticker: str, ridge_model, logistic_model, scaler) -> None:
    """Save both models plus the scaler."""
    joblib.dump(ridge_model, MODELS_DIR / f"{ticker}_ridge.pkl")
    joblib.dump(logistic_model, MODELS_DIR / f"{ticker}_logistic.pkl")
    joblib.dump(scaler, MODELS_DIR / f"{ticker}_scaler.pkl")


if __name__ == "__main__":
    from src.preprocessing import prepare_pipeline

    split, _ = prepare_pipeline("AAPL")

    ridge, _, ridge_metrics = train_ridge(split)
    print("Ridge:")
    for k, v in ridge_metrics.items():
        if isinstance(v, (int, float)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    logistic, _, _, log_metrics = train_logistic(split)
    print("\nLogistic:")
    for k, v in log_metrics.items():
        if isinstance(v, (int, float, np.floating)):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    save_baseline_models("AAPL", ridge, logistic, split.scaler)