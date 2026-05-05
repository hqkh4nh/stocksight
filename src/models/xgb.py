import joblib
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score, roc_auc_score)

from src.config import CFG, MODELS_DIR
from src.preprocessing import SplitData


def train_xgb_regressor(split: SplitData):
    params = CFG["ml"]["xgboost"]
    model = XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        random_state=42,
        n_jobs=-1,
    )
    # Train on returns (stationary) instead of raw price — trees can't extrapolate
    model.fit(split.X_train, split.y_return_train)

    pred_return = model.predict(split.X_test)
    y_pred = split.close_test * (1 + pred_return)

    metrics = {
        "test_mae": mean_absolute_error(split.y_price_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(split.y_price_test, y_pred)),
        "test_mape": np.mean(np.abs((split.y_price_test - y_pred) / split.y_price_test)) * 100,
        "feature_importances": dict(zip(split.feature_names, model.feature_importances_.tolist())),
    }
    return model, y_pred, metrics


def train_xgb_classifier(split: SplitData):
    params = CFG["ml"]["xgboost"]
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(split.X_train, split.y_direction_train)

    y_pred = model.predict(split.X_test)
    y_proba = model.predict_proba(split.X_test)[:, 1]
    metrics = {
        "test_accuracy": accuracy_score(split.y_direction_test, y_pred),
        "test_f1": f1_score(split.y_direction_test, y_pred),
        "test_roc_auc": roc_auc_score(split.y_direction_test, y_proba),
        "feature_importances": dict(zip(split.feature_names, model.feature_importances_.tolist())),
    }
    return model, y_pred, y_proba, metrics


def save_xgb_models(ticker, xgb_reg, xgb_clf):
    joblib.dump(xgb_reg, MODELS_DIR / f"{ticker}_xgb_reg.pkl")
    joblib.dump(xgb_clf, MODELS_DIR / f"{ticker}_xgb_clf.pkl")


if __name__ == "__main__":
    from src.preprocessing import prepare_pipeline

    split, _ = prepare_pipeline("AAPL")

    xgb_reg, _, reg_m = train_xgb_regressor(split)
    print(f"XGB reg  - MAE={reg_m['test_mae']:.3f}  RMSE={reg_m['test_rmse']:.3f}  MAPE={reg_m['test_mape']:.2f}%")

    xgb_clf, _, _, clf_m = train_xgb_classifier(split)
    print(f"XGB clf  - acc={clf_m['test_accuracy']:.4f}  f1={clf_m['test_f1']:.4f}  auc={clf_m['test_roc_auc']:.4f}")

    save_xgb_models("AAPL", xgb_reg, xgb_clf)
