"""DL model: CNN+BiLSTM Encoder + RepeatVector + LSTM Decoder + Attention.

Architecture:
  Input (B, window, n_features)
  → Conv1D(128,3,same,relu) → BatchNorm → Dropout(0.2)
  → Conv1D(64,3,same,relu)
  → Bidirectional(LSTM(128, return_sequences=True))  → enc_seq (B, window, 256)
  → enc_last = enc_seq[:, -1, :]                     (B, 256)
  → RepeatVector(14)                                  (B, 14, 256)
  → LSTM(256, return_sequences=True)                  (B, 14, 256)  = dec_seq
  → Attention()([dec_seq, enc_seq])                   (B, 14, 256)  = context
  → Concatenate()([dec_seq, context])                 (B, 14, 512)
  → TimeDistributed(Dense(64, relu))
  → TimeDistributed(Dense(1, linear))                 (B, 14, 1)

Loss: directional_loss - penalises wrong-sign predictions 1.5x vs 1.0x.
"""
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Attention,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    LSTM,
    Lambda,
    RepeatVector,
    TimeDistributed,
)
from tensorflow.keras.optimizers import Adam

import src._tf_quiet  # noqa: F401
from src.config import CFG, MODELS_DIR
from src.preprocessing import DLData


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def directional_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """MSE with 1.5x penalty when sign(y_true) and sign(y_pred) are opposite.

    Uses `sign(y_true) * sign(y_pred) < 0` so a zero on either side does not trigger.
    """
    err2 = tf.square(y_true - y_pred)
    opposite = tf.less(tf.sign(y_true) * tf.sign(y_pred), 0.0)
    penalty = tf.where(opposite, 1.5, 1.0)
    return tf.reduce_mean(penalty * err2)


def bounded_return(x: tf.Tensor) -> tf.Tensor:
    """Constrain daily-return prediction to [-10%, +10%] to prevent runaway compounding."""
    return tf.tanh(x) * 0.1


def build_seq2seq_attention(
    window: int,
    n_features: int,
    horizon: int,
    cfg: dict,
) -> Model:
    """Build CNN+BiLSTM encoder -> RepeatVector -> LSTM decoder -> Attention.

    Output shape: (B, horizon, 1).
    """
    dropout_rate = cfg["dropout"]

    # --- Encoder ---
    inputs = Input(shape=(window, n_features), name="encoder_input")

    x = Conv1D(cfg["conv_filters_1"], 3, padding="same", activation="relu", name="conv1")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = Dropout(dropout_rate, name="drop1")(x)

    x = Conv1D(cfg["conv_filters_2"], 3, padding="same", activation="relu", name="conv2")(x)

    enc_seq = Bidirectional(
        LSTM(cfg["bilstm_units"], return_sequences=True), name="bilstm"
    )(x)
    # enc_seq shape: (B, window, bilstm_units*2)

    # Use Lambda for safe serialisation of the slice enc_seq[:, -1, :]
    enc_last = Lambda(lambda t: t[:, -1, :], name="enc_last")(enc_seq)
    # enc_last shape: (B, bilstm_units*2)

    # --- Decoder ---
    dec_input = RepeatVector(horizon, name="repeat")(enc_last)
    # dec_input shape: (B, horizon, bilstm_units*2)

    dec_seq = LSTM(cfg["decoder_lstm_units"], return_sequences=True, name="decoder_lstm")(dec_input)
    # dec_seq shape: (B, horizon, decoder_lstm_units)

    # Attention: query=dec_seq, value=enc_seq
    context = Attention(name="attention")([dec_seq, enc_seq])
    # context shape: (B, horizon, bilstm_units*2)

    merged = Concatenate(name="concat")([dec_seq, context])
    # merged shape: (B, horizon, decoder_lstm_units + bilstm_units*2)

    out = TimeDistributed(Dense(cfg["td_dense_units"], activation="relu"), name="td_dense")(merged)
    out = TimeDistributed(Dense(1, activation=bounded_return), name="return_out")(out)
    # out shape: (B, horizon, 1)

    model = Model(inputs=inputs, outputs=out, name="seq2seq_attention")
    return model


def train_dl(dl: DLData, verbose: int = 0) -> tuple:
    """Train the seq2seq model on DLData. Returns (model, history)."""
    cfg = CFG["dl"]
    set_seed(cfg["seed"])

    n_features = dl.X_train_seq.shape[2]
    model = build_seq2seq_attention(dl.window, n_features, dl.horizon, cfg)

    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"], clipnorm=cfg["clipnorm"]),
        loss=directional_loss,
        metrics=["mae"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,
            verbose=verbose,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg["reduce_lr_factor"],
            patience=cfg["reduce_lr_patience"],
            min_lr=cfg["min_lr"],
            verbose=verbose,
        ),
    ]

    history = model.fit(
        dl.X_train_seq,
        dl.y_return_seq_train,
        validation_split=cfg["val_split"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=verbose,
        shuffle=False,
    )

    return model, history


def save_dl_model(ticker: str, model: Model) -> None:
    """Save model to models/{ticker}_dl.keras."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{ticker}_dl.keras"
    model.save(path)
    print(f"Saved DL model -> {path}")


if __name__ == "__main__":
    from src.preprocessing import build_macro_df, prepare_dl_pipeline

    ticker = "BKR"
    print(f"Loading pipeline for {ticker}...")
    macro_df = build_macro_df()
    dl, _split = prepare_dl_pipeline(ticker, macro_df)
    print(f"  X_train_seq: {dl.X_train_seq.shape}  y_train: {dl.y_return_seq_train.shape}")

    print("Training model (verbose=1)...")
    model, history = train_dl(dl, verbose=1)

    epochs_run = len(history.history["loss"])
    final_val_loss = history.history["val_loss"][-1]
    print(f"epochs run: {epochs_run}  |  final val_loss: {final_val_loss:.6f}")

    save_dl_model(ticker, model)
