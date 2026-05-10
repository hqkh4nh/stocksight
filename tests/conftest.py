import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_split():
    """Synthetic SplitData for ML/baseline unit tests."""
    from src.preprocessing import SplitData

    rng = np.random.default_rng(0)
    n_train, n_val, n_test, n_feat = 180, 20, 50, 8
    X_full = rng.standard_normal((n_train + n_val + n_test, n_feat)).astype(np.float32)
    y_full = (rng.standard_normal(n_train + n_val + n_test) * 0.02).astype(np.float32)
    close_full = np.cumprod(1 + y_full) * 100.0

    return SplitData(
        X_train=X_full[:n_train],
        X_test=X_full[n_train + n_val:],
        y_return_train=y_full[:n_train],
        y_return_test=y_full[n_train + n_val:],
        close_train=close_full[:n_train],
        close_test=close_full[n_train + n_val:],
        feature_names=[f"f{i}" for i in range(n_feat)],
        scaler_X=MinMaxScaler(),
        train_dates=None, test_dates=None,
        X_val=X_full[n_train:n_train + n_val],
        y_return_val=y_full[n_train:n_train + n_val],
        val_dates=None,
    )
