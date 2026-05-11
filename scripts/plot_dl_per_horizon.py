"""Plot DL per-horizon MAE growth (d1 -> d7 -> d14) across all tickers.

Two panels:
  Left:  Each ticker normalized to its d1 MAE; mean +- 1 std overlay.
         Shows how forecast error grows RELATIVE to day-1 as horizon extends.
  Right: Absolute MAE in price units per ticker.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config import RESULTS_DIR


def main():
    df = pd.read_csv(RESULTS_DIR / "dl_summary.csv")
    horizons = [1, 7, 14]
    mae_cols = [f"test_mae_price_d{h}" for h in horizons]

    df = df[df["test_mae_price_d1"] > 0].copy()
    norm = df[mae_cols].div(df["test_mae_price_d1"], axis=0)
    norm.columns = horizons

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for _, row in norm.iterrows():
        ax1.plot(horizons, row.values, color="tab:blue",
                 alpha=0.25, linewidth=0.8)
    mean = norm.mean()
    std = norm.std()
    ax1.plot(horizons, mean.values, color="tab:red", linewidth=2.5,
             label=f"Mean across {len(df)} tickers", marker="o")
    ax1.fill_between(horizons, (mean - std).values, (mean + std).values,
                     color="tab:red", alpha=0.15, label="±1 std")
    ax1.set_xlabel("Forecast horizon (days ahead)")
    ax1.set_ylabel("MAE / MAE(d1)")
    ax1.set_title("Relative error growth (normalized to day-1)")
    ax1.set_xticks(horizons)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    for _, row in df.iterrows():
        ax2.plot(horizons, [row[c] for c in mae_cols],
                 alpha=0.5, linewidth=0.9,
                 marker="o", markersize=3, label=row["ticker"])
    ax2.set_xlabel("Forecast horizon (days ahead)")
    ax2.set_ylabel("MAE (price units, USD)")
    ax2.set_title("Absolute MAE per ticker")
    ax2.set_xticks(horizons)
    ax2.legend(loc="upper left", fontsize=6, ncol=3)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "dl_per_horizon_error.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved: {out}")

    growth = (norm[14] / norm[1]).describe()
    print(f"\nError growth ratio d14/d1: mean={growth['mean']:.2f}, "
          f"median={growth['50%']:.2f}, std={growth['std']:.2f}")


if __name__ == "__main__":
    main()
