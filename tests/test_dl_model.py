import numpy as np
import pytest
import tensorflow as tf

from src.models.dl_model import (bounded_return, build_seq2seq_attention,
                                  directional_loss)


def test_directional_loss_zero_when_perfect():
    y_true = tf.constant([[0.01], [-0.02], [0.005]])
    y_pred = tf.constant([[0.01], [-0.02], [0.005]])
    assert float(directional_loss(y_true, y_pred)) == pytest.approx(0.0, abs=1e-7)


def test_directional_loss_higher_when_sign_wrong():
    y_true = tf.constant([[0.01]])
    y_pred_wrong = tf.constant([[-0.01]])
    y_pred_right = tf.constant([[0.03]])
    err_wrong = float(directional_loss(y_true, y_pred_wrong))
    err_right = float(directional_loss(y_true, y_pred_right))
    assert err_wrong > err_right
    assert err_wrong / err_right == pytest.approx(1.5, rel=1e-3)


def test_directional_loss_kicks_in_with_negative_targets():
    """Regression: scaler_y previously made all y_true >= 0, disabling the penalty.
    With raw returns (negative truths possible), the penalty must trigger.
    Use mirror predictions with identical |residual| so the only difference is the sign-penalty."""
    y_true = tf.constant([[-0.01]])
    y_pred_right_sign = tf.constant([[-0.03]])  # sign matches, residual = +0.02
    y_pred_wrong_sign = tf.constant([[0.01]])   # sign flipped, residual = -0.02 (same |err|)
    err_right = float(directional_loss(y_true, y_pred_right_sign))
    err_wrong = float(directional_loss(y_true, y_pred_wrong_sign))
    assert err_wrong == pytest.approx(err_right * 1.5, rel=1e-4)


def test_model_output_shape():
    cfg = {
        "conv_filters_1": 32, "conv_filters_2": 16,
        "bilstm_units": 16, "decoder_lstm_units": 32,
        "td_dense_units": 16, "dropout": 0.2,
    }
    model = build_seq2seq_attention(window=60, n_features=20, horizon=14, cfg=cfg)
    dummy = tf.zeros((4, 60, 20))
    out = model(dummy)
    assert tuple(out.shape) == (4, 14, 1)


def test_bounded_return_clamps_to_pm10pct():
    """Predictions must stay in [-0.1, +0.1] regardless of pre-activation magnitude."""
    huge = tf.constant([[-100.0], [-1.0], [0.0], [1.0], [100.0]])
    out = bounded_return(huge).numpy()
    assert out.min() >= -0.10001
    assert out.max() <= 0.10001
    # Around zero, behaves linearly with slope 0.1
    val = float(bounded_return(tf.constant(0.001)).numpy())
    assert val == pytest.approx(0.0001, abs=1e-6)


def test_model_output_bounded_at_inference():
    cfg = {
        "conv_filters_1": 16, "conv_filters_2": 8,
        "bilstm_units": 8, "decoder_lstm_units": 16,
        "td_dense_units": 8, "dropout": 0.0,
    }
    model = build_seq2seq_attention(window=60, n_features=20, horizon=14, cfg=cfg)
    rng = np.random.default_rng(0)
    big_input = rng.standard_normal((8, 60, 20)).astype("float32") * 5.0  # large feature values
    out = model(big_input).numpy()
    assert out.min() >= -0.10001 and out.max() <= 0.10001
