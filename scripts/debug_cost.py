"""Investigate why COST MAPE_14d = 130% while BKR/FITB are 5%/4%.

Hypotheses to check:
  H1: pred returns have non-zero mean -> 14-day cumprod biases price up/down
  H2: pred returns have outliers -> divergent compound
  H3: COST close anchor near zero in some test rows -> divides by tiny denominator
  H4: COST has price-regime shift train->test that breaks the unscaled return distribution
"""
import numpy as np
import tensorflow as tf

from src.models.dl_model import bounded_return, directional_loss
from src.preprocessing import build_macro_df, prepare_dl_pipeline

TICKERS = ["BKR", "COST", "FITB"]


def load_model(ticker):
    return tf.keras.models.load_model(
        f"models/{ticker}_dl.keras",
        custom_objects={"directional_loss": directional_loss,
                        "bounded_return": bounded_return},
        compile=False,
        safe_mode=False,
    )


def main():
    macro = build_macro_df()
    for tkr in TICKERS:
        print(f"\n{'='*70}\n{tkr}\n{'='*70}")
        dl, _ = prepare_dl_pipeline(tkr, macro)
        model = load_model(tkr)
        pred = model.predict(dl.X_test_seq, verbose=0)[..., 0]      # (N, 14)
        true = dl.y_return_seq_test[..., 0]                          # (N, 14)
        anchor = dl.close_anchor_test                                # (N,)

        print(f"Test sequences: {pred.shape}")
        print(f"Anchor close range: [{anchor.min():.2f}, {anchor.max():.2f}]  "
              f"mean={anchor.mean():.2f}")

        print(f"\nPRED returns: mean={pred.mean():+.6f}  std={pred.std():.6f}  "
              f"min={pred.min():+.4f}  max={pred.max():+.4f}")
        print(f"TRUE returns: mean={true.mean():+.6f}  std={true.std():.6f}  "
              f"min={true.min():+.4f}  max={true.max():+.4f}")

        # Cumulative 14-day return per sequence
        pred_cum14 = np.prod(1 + pred, axis=1) - 1                   # (N,)
        true_cum14 = np.prod(1 + true, axis=1) - 1
        print(f"\nPRED cum 14d: mean={pred_cum14.mean():+.4f}  std={pred_cum14.std():.4f}  "
              f"max|.|={np.abs(pred_cum14).max():.4f}")
        print(f"TRUE cum 14d: mean={true_cum14.mean():+.4f}  std={true_cum14.std():.4f}  "
              f"max|.|={np.abs(true_cum14).max():.4f}")

        # Per-day MAPE breakdown
        pred_prices = anchor[:, None] * np.cumprod(1 + pred, axis=1)
        true_prices = anchor[:, None] * np.cumprod(1 + true, axis=1)
        mape_per_day = np.mean(np.abs((true_prices - pred_prices) / true_prices), axis=0) * 100
        print(f"\nMAPE per day (1..14):")
        print("  " + "  ".join(f"d{i+1}={m:.1f}%" for i, m in enumerate(mape_per_day)))

        # Worst sequences
        worst_idx = np.argsort(-np.abs((true_prices - pred_prices) / true_prices).mean(axis=1))[:3]
        print(f"\nTop-3 worst test sequences (by per-row mean MAPE):")
        for k in worst_idx:
            row_mape = np.mean(np.abs((true_prices[k] - pred_prices[k]) / true_prices[k])) * 100
            print(f"  row {k}: anchor={anchor[k]:.2f}  pred_cum14={pred_cum14[k]:+.4f}  "
                  f"true_cum14={true_cum14[k]:+.4f}  row_mape={row_mape:.1f}%")
            print(f"    pred returns: {np.array2string(pred[k], precision=4)}")
            print(f"    true returns: {np.array2string(true[k], precision=4)}")


if __name__ == "__main__":
    main()
