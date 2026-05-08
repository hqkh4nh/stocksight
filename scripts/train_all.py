import time

from src.config import CFG
from src.preprocessing import prepare_pipeline
from src.models.baseline import train_ridge, train_logistic, save_baseline_models
from src.models.random_forest import train_rf_regressor, train_rf_classifier, save_rf_models
from src.models.xgb import train_xgb_regressor, train_xgb_classifier, save_xgb_models
from src.evaluate import append_result, save_feature_importances, reset_summary

def train_one_ticker(ticker):
    split, _ = prepare_pipeline(ticker)

    ridge, _, m = train_ridge(split)
    append_result(ticker, "Ridge", "regression", m)

    logistic, _, _, m = train_logistic(split)
    append_result(ticker, "Logistic", "classification", m)

    save_baseline_models(ticker, ridge, logistic, split.scaler)

    rf_reg, _, m = train_rf_regressor(split)
    append_result(ticker, "RandomForest", "regression", m)
    save_feature_importances(ticker, "RF_reg", m["feature_importances"])

    rf_clf, _, _, m = train_rf_classifier(split)
    append_result(ticker, "RandomForest", "classification", m)
    save_feature_importances(ticker, "RF_clf", m["feature_importances"])

    save_rf_models(ticker, rf_reg, rf_clf)

    xgb_reg, _, m = train_xgb_regressor(split)
    append_result(ticker, "XGBoost", "regression", m)
    save_feature_importances(ticker, "XGB_reg", m["feature_importances"])

    xgb_clf, _, _, m = train_xgb_classifier(split)
    append_result(ticker, "XGBoost", "classification", m)
    save_feature_importances(ticker, "XGB_clf", m["feature_importances"])

    save_xgb_models(ticker, xgb_reg, xgb_clf)

def main():
    reset_summary()

    for ticker in CFG["tickers"]:
        t0 = time.time()
        try:
            train_one_ticker(ticker)
            print(f"{ticker}  ok ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"{ticker}  FAILED: {e}")


if __name__ == "__main__":
    main()