"""V5 Phase 3 — Stage 1 trend label comparison: SMA / Forward-return / ZigZag.

Output: reports/Phase3/v5_p3_label_comparison_btc.png
        reports/Phase3/v5_p3_label_comparison_eth.png

Üç yöntem aynı price serisi üzerinde karşılaştırılır:
  Panel 1 — SMA crossover (TAUTOLOJI: feature space ile lineer ilişki)
  Panel 2 — Forward-return + adaptive threshold (h=10, k=0.5)
  Panel 3 — ZigZag piecewise segmentation (dev=10%, min=10d, amp=5%)

Anlatım: SMA crossover label MA_slope_20/MA_slope_50 feature'larıyla
yarı-çizgisel ilişkili → model trivially öğrenir, prediction skill yok.
ZigZag (price geometry) ve forward-return (gelecek bilgi) feature space
ile ortogonal → gerçek bir öğrenme görevi yaratır.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.labels.v5_trend_labels import forward_return_trend_label, zigzag_trend_label, label_distribution
from src.features.v5_trend_features import sma_crossover_trend_label

LABEL_COLORS = {"uptrend": "#7ec27e", "downtrend": "#e07e7e", "range": "#f0c870"}


def shade(ax, label_series):
    s = label_series.dropna()
    if s.empty:
        return
    cur, start = s.iloc[0], s.index[0]
    for i in range(1, len(s)):
        v = s.iloc[i]
        if v != cur:
            ax.axvspan(start, s.index[i], color=LABEL_COLORS.get(cur, "white"), alpha=0.30, lw=0)
            cur, start = v, s.index[i]
    ax.axvspan(start, s.index[-1], color=LABEL_COLORS.get(cur, "white"), alpha=0.30, lw=0)


def plot_comparison(close, asset_label, color_close, out_path):
    sma_lab = sma_crossover_trend_label(close, fast_window=20, slow_window=50)
    fr_df = forward_return_trend_label(close, horizon=10, rolling_window=252, threshold_k=0.5)
    zz_df = zigzag_trend_label(close, deviation_pct=0.10, min_segment_days=10, range_amplitude=0.05)

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    shade(axes[0], sma_lab)
    axes[0].semilogy(close.index, close.values, color=color_close, lw=1.0)
    axes[0].set_ylabel(f"{asset_label} (log)", fontsize=11)
    sma_dist = label_distribution(sma_lab)
    axes[0].set_title(
        f"Method 1 — SMA(20)/SMA(50) crossover  [TAUTOLOGY: ~linear in features]\n"
        f"  up {sma_dist['uptrend']:.1%}  /  down {sma_dist['downtrend']:.1%}  /  range {sma_dist['range']:.1%}",
        fontsize=11, fontweight="bold")

    shade(axes[1], fr_df["label"])
    axes[1].semilogy(close.index, close.values, color=color_close, lw=1.0)
    axes[1].set_ylabel(f"{asset_label} (log)", fontsize=11)
    fr_dist = label_distribution(fr_df["label"])
    axes[1].set_title(
        f"Method 2 — Forward-return (h=10, k=0.5)  [orthogonal to features]\n"
        f"  up {fr_dist['uptrend']:.1%}  /  down {fr_dist['downtrend']:.1%}  /  range {fr_dist['range']:.1%}",
        fontsize=11, fontweight="bold")

    shade(axes[2], zz_df["label"])
    axes[2].semilogy(close.index, close.values, color=color_close, lw=1.0)
    axes[2].set_ylabel(f"{asset_label} (log)", fontsize=11)
    zz_dist = label_distribution(zz_df["label"])
    n_segs = zz_df["segment_id"].nunique()
    axes[2].set_title(
        f"Method 3 — ZigZag (dev=10%, min=10d, amp=5%)  [orthogonal, {n_segs} segments]\n"
        f"  up {zz_dist['uptrend']:.1%}  /  down {zz_dist['downtrend']:.1%}  /  range {zz_dist['range']:.1%}",
        fontsize=11, fontweight="bold")

    handles = [Patch(facecolor=c, alpha=0.5, label=l) for l, c in LABEL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=11)
    fig.suptitle(f"Phase 3 — Stage 1 trend label comparison ({asset_label})",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    out_dir = PROJECT_ROOT / "reports" / "Phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    plot_comparison(btc["Close"], "BTC", "#F7931A",
                    out_dir / "v5_p3_label_comparison_btc.png")
    plot_comparison(eth["Close"], "ETH", "#627EEA",
                    out_dir / "v5_p3_label_comparison_eth.png")


if __name__ == "__main__":
    main()
