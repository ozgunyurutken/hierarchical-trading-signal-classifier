"""
V5 Decision Gate 1.5 detail — 9 derived stage 2 feature time-series plot.

Output: reports/v5_corr_recheck_derived_features.png

Each panel: derived feature for BTC-aligned period (2014-09-17 → 2025-12-30)
with train/val/test split shading.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from src.utils.config import cfg

plt.rcParams.update({"figure.dpi": 130, "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

config = cfg()
TRAIN_FRAC = config["training"]["train_size"]
VAL_FRAC = config["training"]["validation_size"]

# Stage 2 derived feature config (label, unit, color, semilog?)
PANELS = [
    ("VIX_zscore_long", "VIX z-score (25-yıl baseline)", "C1", False),
    ("DXY_zscore_long", "DXY z-score (25-yıl baseline)", "C2", False),
    ("SP500_log_return_5d", "S&P 500 5-gün log return", "C0", False),
    ("Gold_log_return_20d", "Gold 20-gün log return", "C3", False),
    ("FEDFUNDS_change_60d", "FEDFUNDS 60-gün % değişim", "C6", False),
    ("UNRATE_change_180d", "UNRATE 180-gün % değişim", "C8", False),
    ("CPI_yoy_change", "CPI YoY (252-gün) %", "C7", False),
    ("M2_yoy_change", "M2 YoY (252-gün) %", "C9", False),
    ("Yield_Curve_10Y_2Y", "Yield Curve (US10Y - US2Y)", "C4", False),
]


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    # Use FULL pre-train derived (2000-2025) — Stage 2 K-Means fit data
    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True)

    # Crypto era markers
    crypto_start_btc = pd.Timestamp("2014-09-17")
    crypto_start_eth = pd.Timestamp("2017-11-09")

    # Train/val/test split dates from BTC-aligned (4094 days)
    btc_aligned = pd.read_csv(proc / "btc_aligned_v5.csv",
                              index_col=0, parse_dates=True)
    n = len(btc_aligned)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_end = btc_aligned.index[n_train - 1]
    val_end = btc_aligned.index[n_train + n_val - 1]

    # Plot 9-panel grid (5x2, last cell legend)
    fig, axes = plt.subplots(5, 2, figsize=(16, 14))
    axes = axes.flatten()
    btc_macro = pretrain   # use pre-train as primary plot data

    for i, (col, title, color, semilog) in enumerate(PANELS):
        ax = axes[i]
        s = btc_macro[col].dropna()
        if semilog:
            ax.semilogy(s.index, s.values, lw=0.85, color=color)
        else:
            ax.plot(s.index, s.values, lw=0.85, color=color)
        # Reference lines
        if "zscore" in col:
            ax.axhline(0, color="black", lw=0.6, alpha=0.5)
            ax.axhline(1, color="red", lw=0.5, ls=":", alpha=0.5)
            ax.axhline(-1, color="red", lw=0.5, ls=":", alpha=0.5)
            ax.axhline(2, color="red", lw=0.5, ls="--", alpha=0.5, label="±2σ")
            ax.axhline(-2, color="red", lw=0.5, ls="--", alpha=0.5)
        elif "return" in col or "change" in col or "yoy" in col:
            ax.axhline(0, color="black", lw=0.6, alpha=0.5)

        # Pre-train period: 2000 → 2014-09 = K-Means fit-only window (gri)
        ax.axvspan(s.index.min(), crypto_start_btc, color="#e0e0e0", alpha=0.30)
        # Crypto era split shading: train/val/test
        ax.axvspan(crypto_start_btc, train_end, color="#cfe5ff", alpha=0.20)
        ax.axvspan(train_end, val_end, color="#ffe5b4", alpha=0.22)
        ax.axvspan(val_end, s.index.max(), color="#ffb3b3", alpha=0.22)
        # Crypto era boundary line
        ax.axvline(crypto_start_btc, color="black", lw=0.8, alpha=0.6)

        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8.5)

    # Last cell — legend
    ax = axes[-1]
    ax.axis("off")
    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, fc="#e0e0e0", alpha=0.6,
                      label="Pre-train (2000 → 2014-09, K-Means fit only)"),
        plt.Rectangle((0, 0), 1, 1, fc="#cfe5ff", alpha=0.6,
                      label="Crypto Train (2014-09 → 2022-08, 70%)"),
        plt.Rectangle((0, 0), 1, 1, fc="#ffe5b4", alpha=0.6,
                      label="Crypto Val   (2022-08 → 2024-04, 15%)"),
        plt.Rectangle((0, 0), 1, 1, fc="#ffb3b3", alpha=0.6,
                      label="Crypto Test  (2024-04 → 2025-12, 15%)"),
        plt.Line2D([], [], color="black", lw=0.8, label="Crypto era başlangıcı"),
        plt.Line2D([], [], color="red", lw=0.5, ls=":", label="±1σ (z-score)"),
        plt.Line2D([], [], color="red", lw=0.5, ls="--", label="±2σ (z-score)"),
        plt.Line2D([], [], color="black", lw=0.5, label="Zero line (returns / changes)"),
    ]
    ax.legend(handles=legend_elems, loc="center", fontsize=9.5, frameon=False,
              title="Legend & Color Code", title_fontsize=11)

    n_pretrain = len(pretrain)
    fig.suptitle(
        f"V5 Decision Gate 1.5 — 9 Derived Stage 2 Macro Features (Pre-train + Crypto era)\n"
        f"Pre-train: {pretrain.index.min().date()} → 2014-09-17 (K-Means fit window)  "
        f"·  Crypto: 2014-09-17 → {pretrain.index.max().date()} (Stage 2 inference)\n"
        f"Total {n_pretrain} bday — 9 derived: 2 long-term z-score, 2 log-return, "
        f"2 RoC, 2 YoY, 1 spread  ·  FRED lag: CPI 14g, UNRATE 7g (BLS realistic)",
        fontsize=11, fontweight="bold", y=0.998
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PROJECT_ROOT / "reports" / "v5_corr_recheck_derived_features.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out.relative_to(PROJECT_ROOT)}")
    print(f"\nSummary stats per derived feature:")
    print(btc_macro[[p[0] for p in PANELS]].describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
