"""V5 Phase 3 — Stage 1 features timeline grid (BTC + ETH).

Outputs:
  reports/Phase3/v5_p3_stage1_features_btc.png  (4×4 grid, 14 features)
  reports/Phase3/v5_p3_stage1_features_eth.png  (4×4 grid, 14 features)

14 BTC/ETH OHLCV-based features grouped into 6 categories. Color-coded
by category for visual grouping.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt

from src.features.v5_trend_features import build_stage1_features, FEATURE_GROUPS, STAGE1_FEATURE_COLS

GROUP_COLORS = {
    "Returns":         "#3a6fb0",
    "Trend strength":  "#2a7a2a",
    "Momentum":        "#bf6e1d",
    "Mean reversion":  "#9c4dcf",
    "Volatility":      "#cc4444",
    "Volume":          "#557755",
}


def feature_to_group(feature: str) -> str:
    for g, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return g
    return ""


def plot_features(df, asset_label, color_close, out_path):
    feats = build_stage1_features(df[["Open", "High", "Low", "Close", "Volume"]])
    feats = feats.dropna()

    fig, axes = plt.subplots(4, 4, figsize=(20, 12), sharex=True)
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    axes_list = list(axes.flat)

    for ax, col in zip(axes_list[:14], STAGE1_FEATURE_COLS):
        group = feature_to_group(col)
        c = GROUP_COLORS.get(group, "black")
        ax.plot(feats.index, feats[col], color=c, lw=0.8)
        ax.set_title(f"{col}  ({group})", fontsize=10, color=c, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    ax15, ax16 = axes_list[14], axes_list[15]
    close = df["Close"].reindex(feats.index)
    ax15.semilogy(close.index, close.values, color=color_close, lw=1.0)
    ax15.set_title(f"{asset_label} Close (log) — reference", fontsize=10, fontweight="bold")
    ax15.tick_params(labelsize=8); ax15.grid(True, alpha=0.3)
    ax16.set_yticks([]); ax16.set_xticks([]); ax16.grid(False)
    for sp in ax16.spines.values(): sp.set_visible(False)
    ax16.text(0.5, 0.5, "(reserved)", ha="center", va="center",
              transform=ax16.transAxes, fontsize=11, color="#aaa", style="italic")

    fig.suptitle(f"Phase 3 — Stage 1 Trend Classifier features ({asset_label}, 14 OHLCV-only features)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    out_dir = PROJECT_ROOT / "reports" / "Phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    plot_features(btc, "BTC", "#F7931A", out_dir / "v5_p3_stage1_features_btc.png")
    plot_features(eth, "ETH", "#627EEA", out_dir / "v5_p3_stage1_features_eth.png")


if __name__ == "__main__":
    main()
