"""ML regressors: Ridge, RF, XGB. Each returns (model, y_pred_test, y_pred_val)."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.config import CFG, MODELS_DIR
from src.metrics import reg_metrics
from src.preprocessing import SplitData


def train_ridge(split: SplitData):
    # Ridge la hoi quy tuyen tinh co regularization L2. No phu hop lam mo
    # hinh nen vi don gian, it overfit hon linear regression thuong.
    p = CFG["ml"]["ridge"]
    model = Ridge(alpha=p["alpha"], random_state=42)
    model.fit(split.X_train, split.y_return_train)
    y_pred_test = model.predict(split.X_test)
    y_pred_val = model.predict(split.X_val)
    return model, y_pred_test, y_pred_val


def train_rf_reg(split: SplitData):
    # Random Forest gom nhieu cay quyet dinh. Moi cay hoc mot phan du lieu,
    # ket qua cuoi la trung binh, giup bat quan he phi tuyen.
    p = CFG["ml"]["random_forest"]
    model = RandomForestRegressor(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        min_samples_split=p["min_samples_split"],
        random_state=42, n_jobs=-1,
    )
    model.fit(split.X_train, split.y_return_train)
    y_pred_test = model.predict(split.X_test)
    y_pred_val = model.predict(split.X_val)
    return model, y_pred_test, y_pred_val


def train_xgb_reg(split: SplitData):
    # XGBoost la gradient boosting: cay sau hoc cach sua loi cua cay truoc.
    # Thuong manh tren du lieu bang nhieu feature.
    p = CFG["ml"]["xgboost"]
    model = XGBRegressor(
        n_estimators=p["n_estimators"], max_depth=p["max_depth"],
        learning_rate=p["learning_rate"], subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    model.fit(split.X_train, split.y_return_train)
    y_pred_test = model.predict(split.X_test)
    y_pred_val = model.predict(split.X_val)
    return model, y_pred_test, y_pred_val


def save_ml_models(ticker: str, ridge, rf_reg, xgb_reg, scaler_X):
    # Luu ca model va scaler. Khi du bao live, input moi phai scale bang dung
    # scaler da fit tren train, neu khong prediction se sai thang do.
    joblib.dump(ridge, MODELS_DIR / f"{ticker}_ridge.pkl")
    joblib.dump(rf_reg, MODELS_DIR / f"{ticker}_rf_reg.pkl")
    joblib.dump(xgb_reg, MODELS_DIR / f"{ticker}_xgb_reg.pkl")
    joblib.dump(scaler_X, MODELS_DIR / f"{ticker}_scaler_X.joblib")


# === Recursive multi-step forecast ===

def recursive_forecast_14(model, last_features_row: np.ndarray, feat_cols: list = None,
                          scaler_X=None, horizon: int = 14) -> np.ndarray:
    """Predict horizon-step returns by feeding predictions back into lag features.

    Updates `returns`, `log_returns`, `close_pct_lag_1`, `close_pct_lag_5` after each step.
    Other features (RSI, MACD, MA, volatility, macro, sector corrs) held constant.
    If `feat_cols` is None, falls back to legacy behaviour (no lag updates).
    """
    feat_raw = last_features_row.astype(float).copy()
    preds = []

    if feat_cols is None:
        # Che do cu: lap lai cung mot dong feature, khong cap nhat lag. Giu
        # lai de tuong thich voi test/code cu.
        feat = feat_raw.reshape(1, -1)
        for _ in range(horizon):
            preds.append(float(model.predict(feat)[0]))
        return np.array(preds)

    idx = {c: i for i, c in enumerate(feat_cols)}
    recent = [feat_raw[idx["close_pct_lag_1"]]] * 5 if "close_pct_lag_1" in idx else [0.0] * 5

    for _ in range(horizon):
        # Moi buoc du bao 1 ngay return. Neu co scaler thi transform feature
        # raw sang cung thang do voi luc train.
        x = feat_raw.reshape(1, -1)
        if scaler_X is not None:
            x = scaler_X.transform(x)
        r_pred = float(model.predict(x)[0])
        preds.append(r_pred)

        # Feeding back: return du bao cua hom nay tro thanh feature lag cho
        # buoc ngay mai. Cac feature phuc tap hon nhu RSI/MACD duoc giu nguyen.
        recent.append(r_pred)
        recent = recent[-5:]

        if "returns" in idx:
            feat_raw[idx["returns"]] = r_pred
        if "log_returns" in idx:
            feat_raw[idx["log_returns"]] = np.log1p(r_pred)
        if "close_pct_lag_1" in idx:
            feat_raw[idx["close_pct_lag_1"]] = r_pred
        if "close_pct_lag_5" in idx:
            cum5 = float(np.prod([1 + r for r in recent]) - 1)
            feat_raw[idx["close_pct_lag_5"]] = cum5

    return np.array(preds)


if __name__ == "__main__":
    from src.preprocessing import prepare_dl_pipeline, build_macro_df
    macro_df = build_macro_df()
    _, split = prepare_dl_pipeline("BKR", macro_df)

    _, yp_test, yp_val = train_ridge(split)
    print("Ridge val MAE:", reg_metrics(split.y_return_val, yp_val))
    print("Ridge test MAE:", reg_metrics(split.y_return_test, yp_test))
