import numpy as np
import pytest
import tensorflow as tf

from src.config import CFG
from src.models.dl_model import (build_seq2seq_attention, make_bounded_return,
                                  make_directional_loss)


def test_directional_loss_zero_when_perfect():
    loss_fn = make_directional_loss(3.0)
    y_true = tf.constant([[0.01], [-0.02], [0.005]])
    y_pred = tf.constant([[0.01], [-0.02], [0.005]])
    assert float(loss_fn(y_true, y_pred)) == pytest.approx(0.0, abs=1e-7)


def test_directional_loss_penalty_ratio_matches_factor():
    """When |residual| is identical, wrong-sign loss should be penalty * right-sign loss."""
    for penalty in (1.5, 3.0, 5.0):
        loss_fn = make_directional_loss(penalty)
        y_true = tf.constant([[0.01]])
        y_pred_wrong = tf.constant([[-0.01]])
        y_pred_right = tf.constant([[0.03]])
        err_wrong = float(loss_fn(y_true, y_pred_wrong))
        err_right = float(loss_fn(y_true, y_pred_right))
        assert err_wrong / err_right == pytest.approx(penalty, rel=1e-3)


def test_directional_loss_kicks_in_with_negative_targets():
    loss_fn = make_directional_loss(3.0)
    y_true = tf.constant([[-0.01]])
    y_pred_right_sign = tf.constant([[-0.03]])
    y_pred_wrong_sign = tf.constant([[0.01]])
    err_right = float(loss_fn(y_true, y_pred_right_sign))
    err_wrong = float(loss_fn(y_true, y_pred_wrong_sign))
    assert err_wrong == pytest.approx(err_right * 3.0, rel=1e-4)


def test_default_loss_uses_config_penalty():
    """Default `directional_loss` must read penalty from config.yaml."""
    from src.models.dl_model import directional_loss
    expected = CFG["dl"].get("direction_penalty", 3.0)
    y_true = tf.constant([[0.01]])
    y_pred_wrong = tf.constant([[-0.01]])
    y_pred_right = tf.constant([[0.03]])
    ratio = float(directional_loss(y_true, y_pred_wrong)) / float(directional_loss(y_true, y_pred_right))
    assert ratio == pytest.approx(expected, rel=1e-3)


def test_model_output_shape():
    cfg = {
        "conv_filters_1": 32, "conv_filters_2": 16,
        "bilstm_units": 16, "decoder_lstm_units": 32,
        "td_dense_units": 16, "dropout": 0.2, "return_cap": 0.05,
    }
    model = build_seq2seq_attention(window=60, n_features=20, horizon=14, cfg=cfg)
    dummy = tf.zeros((4, 60, 20))
    out = model(dummy)
    assert tuple(out.shape) == (4, 14, 1)


def test_bounded_return_respects_cap():
    cap = 0.05
    bounded = make_bounded_return(cap)
    huge = tf.constant([[-100.0], [-1.0], [0.0], [1.0], [100.0]])
    out = bounded(huge).numpy()
    assert out.min() >= -cap - 1e-5
    assert out.max() <= cap + 1e-5
    val = float(bounded(tf.constant(0.001)).numpy())
    assert val == pytest.approx(0.001 * cap, rel=1e-4)


def test_model_output_bounded_at_inference():
    cap = 0.05
    cfg = {
        "conv_filters_1": 16, "conv_filters_2": 8,
        "bilstm_units": 8, "decoder_lstm_units": 16,
        "td_dense_units": 8, "dropout": 0.0, "return_cap": cap,
    }
    model = build_seq2seq_attention(window=60, n_features=20, horizon=14, cfg=cfg)
    rng = np.random.default_rng(0)
    big_input = rng.standard_normal((8, 60, 20)).astype("float32") * 5.0
    out = model(big_input).numpy()
    assert out.min() >= -cap - 1e-5 and out.max() <= cap + 1e-5
