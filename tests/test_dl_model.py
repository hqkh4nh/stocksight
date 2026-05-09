import numpy as np
import pytest
import tensorflow as tf

from src.models.dl_model import directional_loss, build_seq2seq_attention


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
