import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, accuracy_score, f1_score, roc_auc_score)

from src.config import CFG, MODELS_DIR
from src.preprocessing import SplitData

def train_rf_regressor(split: SplitData):
    params = CFG["ml"]["random_forest"]
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=42,
        n_jobs=-1,
    )
    model.fit(split.X_train, split.y_price_train)

    y_pred = model.predict(split.X_test)
    metrics = {
        "test_mae": mean_absolute_error(split.y_price_test, y_pred),
        "test_rmse": np.sqrt(mean_squared_error(split.y_price_test, y_pred)),
        "test_mape": np.mean(np.abs((split.y_price_test - y_pred) / split.y_price_test)) * 100,
        "feature_importances": dict(zip(split.feature_names, model.feature_importances_.tolist())),
    }
    return model, y_pred, metrics

def train_rf_classifier(split: SplitData):
    params = CFG["ml"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        class_weight="balanced",
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

def save_rf_models(ticker, rf_reg, rf_clf):
    joblib.dump(rf_reg, MODELS_DIR / f"{ticker}_rf_reg.pkl")
    joblib.dump(rf_clf, MODELS_DIR / f"{ticker}_rf_clf.pkl")

if __name__ == "__main__":
    from src.preprocessing import prepare_pipeline

    split, _ = prepare_pipeline("AAPL")

    rf_reg, _, reg_m = train_rf_regressor(split)
    print(f"RF reg  - MAE={reg_m['test_mae']:.3f}  RMSE={reg_m['test_rmse']:.3f}  MAPE={reg_m['test_mape']:.2f}%")

    rf_clf, _, _, clf_m = train_rf_classifier(split)
    print(f"RF clf  - acc={clf_m['test_accuracy']:.4f}  f1={clf_m['test_f1']:.4f}  auc={clf_m['test_roc_auc']:.4f}")

    top = sorted(clf_m["feature_importances"].items(), key=lambda x: -x[1])[:5]
    print("top features:", top)

    save_rf_models("AAPL", rf_reg, rf_clf)