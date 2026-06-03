"""DL model: CNN+BiLSTM Encoder + RepeatVector + LSTM Decoder + Attention.

Architecture:
  Input (B, window, n_features)
  -> Conv1D(128,3,same,relu) -> BatchNorm -> Dropout(0.2)
  -> Conv1D(64,3,same,relu)
  -> Bidirectional(LSTM(128, return_sequences=True))  -> enc_seq (B, window, 256)
  -> enc_last = enc_seq[:, -1, :]                     (B, 256)
  -> RepeatVector(14)                                  (B, 14, 256)
  -> LSTM(256, return_sequences=True)                  (B, 14, 256)  = dec_seq
  -> Attention()([dec_seq, enc_seq])                   (B, 14, 256)  = context
  -> Concatenate()([dec_seq, context])                 (B, 14, 512)
  -> TimeDistributed(Dense(64, relu))
  -> TimeDistributed(Dense(1, linear))                 (B, 14, 1)

Loss: directional_loss with `direction_penalty` multiplier on wrong-sign predictions.
Output: tanh(x) * `return_cap` to bound predicted daily returns.
"""
import json
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
    # Dat seed de ket qua train lap lai tot hon giua cac lan chay.
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_directional_loss(penalty: float):
    """Factory: MSE with `penalty`x weight when sign(y_true) * sign(y_pred) < 0."""
    penalty_f = float(penalty)

    def directional_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # err2 la sai so binh phuong thong thuong cua bai toan hoi quy.
        err2 = tf.square(y_true - y_pred)
        # opposite=True khi model du doan sai huong tang/giam.
        opposite = tf.less(tf.sign(y_true) * tf.sign(y_pred), 0.0)
        # Neu sai huong, nhan loss len penalty lan de model uu tien hoc dau +/-
        # cua return, khong chi hoc gan gia tri trung binh.
        weight = tf.where(opposite, penalty_f, 1.0)
        return tf.reduce_mean(weight * err2)

    return directional_loss


def directional_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Default loss using the penalty configured in config.yaml."""
    return make_directional_loss(CFG["dl"].get("direction_penalty", 3.0))(y_true, y_pred)


def make_bounded_return(cap: float):
    """Factory: tanh activation scaled by `cap`."""
    cap_f = float(cap)

    def bounded_return(x: tf.Tensor) -> tf.Tensor:
        # tanh gioi han dau ra trong [-1, 1], nhan cap=0.05 de return moi
        # ngay nam trong khoang xap xi [-5%, +5%].
        return tf.tanh(x) * cap_f

    return bounded_return


def bounded_return(x: tf.Tensor) -> tf.Tensor:
    """Default activation using the cap configured in config.yaml."""
    return make_bounded_return(CFG["dl"].get("return_cap", 0.05))(x)


def build_seq2seq_attention(
    window: int,
    n_features: int,
    horizon: int,
    cfg: dict,
) -> Model:
    dropout_rate = cfg["dropout"]
    cap = cfg.get("return_cap", 0.05)

    inputs = Input(shape=(window, n_features), name="encoder_input")

    # Conv1D nhin cac mau cuc bo tren chuoi 60 ngay, giong bo loc phat hien
    # bien dong ngan han trong feature.
    x = Conv1D(cfg["conv_filters_1"], 3, padding="same", activation="relu", name="conv1")(inputs)
    x = BatchNormalization(name="bn1")(x)
    x = Dropout(dropout_rate, name="drop1")(x)

    x = Conv1D(cfg["conv_filters_2"], 3, padding="same", activation="relu", name="conv2")(x)

    # BiLSTM doc chuoi theo hai chieu trong window lich su de ma hoa boi canh
    # qua khu thanh enc_seq.
    enc_seq = Bidirectional(
        LSTM(cfg["bilstm_units"], return_sequences=True), name="bilstm"
    )(x)

    # Lay hidden state cuoi lam tom tat cua 60 ngay input.
    enc_last = Lambda(lambda t: t[:, -1, :], name="enc_last")(enc_seq)

    # Decoder can tao 14 buoc output, nen RepeatVector nhan ban tom tat
    # encoder thanh 14 vector dau vao cho LSTM decoder.
    dec_input = RepeatVector(horizon, name="repeat")(enc_last)
    dec_seq = LSTM(cfg["decoder_lstm_units"], return_sequences=True, name="decoder_lstm")(dec_input)

    # Attention cho moi buoc du bao nhin lai toan bo enc_seq, giup model tap
    # trung vao nhung ngay lich su co lien quan hon.
    context = Attention(name="attention")([dec_seq, enc_seq])
    merged = Concatenate(name="concat")([dec_seq, context])

    out = TimeDistributed(Dense(cfg["td_dense_units"], activation="relu"), name="td_dense")(merged)
    out = TimeDistributed(Dense(1, activation=make_bounded_return(cap)), name="return_out")(out)

    return Model(inputs=inputs, outputs=out, name="seq2seq_attention")


def train_dl(dl: DLData, verbose: int = 0) -> tuple:
    """Train the seq2seq model. Returns (model, history)."""
    cfg = CFG["dl"]
    set_seed(cfg["seed"])

    n_features = dl.X_train_seq.shape[2]
    model = build_seq2seq_attention(dl.window, n_features, dl.horizon, cfg)

    loss_fn = make_directional_loss(cfg.get("direction_penalty", 3.0))
    model.compile(
        optimizer=Adam(learning_rate=cfg["learning_rate"], clipnorm=cfg["clipnorm"]),
        loss=loss_fn,
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
        # shuffle=False bat buoc voi chuoi thoi gian: khong tron thu tu lich su
        # khi cat validation_split phan cuoi cua train.
        shuffle=False,
    )

    return model, history


def save_dl_model(ticker: str, model: Model) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{ticker}_dl.keras"
    model.save(path)
    print(f"Saved DL model -> {path}")


def save_dl_history(ticker: str, history) -> None:
    """Persist per-epoch loss/val_loss/mae/val_mae to JSON for inspection."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{ticker}_dl_history.json"
    payload = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    path.write_text(json.dumps(payload, indent=2))


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
    save_dl_history(ticker, history)
