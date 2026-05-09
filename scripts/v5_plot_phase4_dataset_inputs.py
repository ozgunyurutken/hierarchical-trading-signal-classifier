"""V5 Phase 4 — Stage 3 dataset input visualization.

Outputs to reports/Phase4.0_inputs/:
  v5_p4_label_timeline.png         — price + signal label shading (BTC, ETH)
  v5_p4_label_distribution.png     — Buy/Hold/Sell counts + share, BTC vs ETH
  v5_p4_forward_return_hist.png    — forward return distribution per class + eps band
  v5_p4_feature_correlation.png    — 16x16 feature correlation heatmap (BTC, ETH)
  v5_p4_stage1_raw_vs_smooth.png   — P_uptrend raw vs 10d smoothed timeline
  v5_p4_stage2_regime_timeline.png — Stage 2 regime label + days_since_last_transition
  v5_p4_oscillator_distributions.png — 6 oscillator histograms per asset
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.features.v5_stage3_features import STAGE3_FEATURE_COLS, FEATURE_GROUPS

ASSETS = ["btc", "eth"]
ASSET_COLORS = {"btc": "#F7931A", "eth": "#627EEA"}
SIGNAL_COLORS = {"Buy": "#3a8a3a", "Hold": "#999999", "Sell": "#cc4444"}
REGIME_COLORS = {"Bull": "#7ec27e", "Neutral": "#f0c870", "Bear": "#e07e7e"}


def _load(asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    proc = PROJECT_ROOT / "data" / "processed"
    feat = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                       index_col=0, parse_dates=True)
    ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                        index_col=0, parse_dates=True)
    return feat, ohlcv


def _shade_label(ax, label_series, colors, alpha=0.30):
    """Shade x-axis spans by sequential label values."""
    s = label_series.dropna()
    if s.empty: return
    cur, start = s.iloc[0], s.index[0]
    for i in range(1, len(s)):
        v = s.iloc[i]
        if v != cur:
            ax.axvspan(start, s.index[i], color=colors.get(cur, "white"),
                       alpha=alpha, lw=0)
            cur, start = v, s.index[i]
    ax.axvspan(start, s.index[-1], color=colors.get(cur, "white"),
               alpha=alpha, lw=0)


def plot_label_timeline(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    for ax, asset in zip(axes, ASSETS):
        feat, ohlcv = _load(asset)
        close = ohlcv["Close"].reindex(feat.index)
        _shade_label(ax, feat["signal_label"], SIGNAL_COLORS, alpha=0.25)
        ax.semilogy(close.index, close.values, color=ASSET_COLORS[asset], lw=1.0)
        ax.set_title(f"{asset.upper()} — price + Stage 3 signal labels (h=5, k=0.5)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{asset.upper()} (log $)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(feat.index.min(), feat.index.max())

    handles = [Patch(facecolor=c, alpha=0.5, label=l) for l, c in SIGNAL_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.005), fontsize=10)
    fig.suptitle("Phase 4 — Stage 3 signal labels timeline\n"
                 "Buy/Hold/Sell determined by causal forward-return + adaptive threshold",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out / "v5_p4_label_timeline.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {(out / 'v5_p4_label_timeline.png').relative_to(PROJECT_ROOT)}")


def plot_label_distribution(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, asset in zip(axes, ASSETS):
        feat, _ = _load(asset)
        counts = feat["signal_label"].value_counts().reindex(["Buy", "Hold", "Sell"], fill_value=0)
        share = counts / counts.sum()
        bars = ax.bar(counts.index, counts.values,
                      color=[SIGNAL_COLORS[c] for c in counts.index],
                      edgecolor="black", linewidth=0.5)
        for bar, v, s in zip(bars, counts.values, share):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v}\n({s:.1%})",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(f"{asset.upper()} — n={counts.sum()}, span "
                     f"{feat.index.min().date()} → {feat.index.max().date()}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(counts.values) * 1.18)

    fig.suptitle("Phase 4 — Stage 3 signal-label class distribution",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "v5_p4_label_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {(out / 'v5_p4_label_distribution.png').relative_to(PROJECT_ROOT)}")


def plot_forward_return_hist(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, asset in zip(axes, ASSETS):
        feat, _ = _load(asset)
        for cls, color in SIGNAL_COLORS.items():
            sub = feat[feat["signal_label"] == cls]["forward_return"]
            ax.hist(sub, bins=60, alpha=0.55, color=color, label=f"{cls} (n={len(sub)})",
                    edgecolor="black", linewidth=0.3)
        # Median eps as reference dashed lines (median of |eps|)
        med_eps = feat["eps_threshold"].median()
        ax.axvline(+med_eps, color="black", ls="--", lw=0.9, label=f"±median eps = ±{med_eps:.3f}")
        ax.axvline(-med_eps, color="black", ls="--", lw=0.9)
        ax.axvline(0, color="grey", ls=":", lw=0.6)
        ax.set_xlim(-0.25, 0.25)
        ax.set_xlabel("5-day forward return")
        ax.set_ylabel("count")
        ax.set_title(f"{asset.upper()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 4 — Forward-return distribution by signal class\n"
                 "Buy/Sell are the tails beyond ±k·rolling_std; Hold is the center band",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "v5_p4_forward_return_hist.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {(out / 'v5_p4_forward_return_hist.png').relative_to(PROJECT_ROOT)}")


def plot_feature_correlation(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, asset in zip(axes, ASSETS):
        feat, _ = _load(asset)
        X = feat[STAGE3_FEATURE_COLS]
        corr = X.corr().values
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(STAGE3_FEATURE_COLS)))
        ax.set_yticks(range(len(STAGE3_FEATURE_COLS)))
        ax.set_xticklabels(STAGE3_FEATURE_COLS, fontsize=7, rotation=70, ha="right")
        ax.set_yticklabels(STAGE3_FEATURE_COLS, fontsize=7)
        # Cell text only for |r| > 0.3
        for i in range(len(STAGE3_FEATURE_COLS)):
            for j in range(len(STAGE3_FEATURE_COLS)):
                v = corr[i, j]
                if abs(v) > 0.3 and i != j:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if abs(v) < 0.7 else "white")
        # Group separator lines
        boundaries = [3, 6, 10]  # after Stage1 raw, smoothed, regime
        for b in boundaries:
            ax.axhline(b - 0.5, color="black", lw=0.7, alpha=0.5)
            ax.axvline(b - 0.5, color="black", lw=0.7, alpha=0.5)
        ax.set_title(f"{asset.upper()}", fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    fig.suptitle("Phase 4 — Stage 3 feature correlation matrix\n"
                 "Black lines separate feature groups (S1 raw / S1 smooth / S2 / Oscillator)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "v5_p4_feature_correlation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {(out / 'v5_p4_feature_correlation.png').relative_to(PROJECT_ROOT)}")


def plot_stage1_raw_vs_smooth(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False)
    for ax, asset in zip(axes, ASSETS):
        feat, _ = _load(asset)
        ax.plot(feat.index, feat["P1_up"], color="#5a9a5a", lw=0.8, alpha=0.65,
                label="P1_up RAW (RF tuned)")
        ax.plot(feat.index, feat["P1_up_smooth10"], color="#1a5a1a", lw=1.4,
                label="P1_up SMOOTHED (10d rolling)")
        ax.set_title(f"{asset.upper()} — Stage 1 P(uptrend) raw vs 10-day smoothed",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("P(uptrend)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")
        ax.set_xlim(feat.index.min(), feat.index.max())

    fig.suptitle("Phase 4 — Stage 1 OOF: raw vs smoothed (Phase 3.5 segment-level finding)\n"
                 "Smoothed posterior reduces daily noise while preserving regime structure",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out / "v5_p4_stage1_raw_vs_smooth.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {(out / 'v5_p4_stage1_raw_vs_smooth.png').relative_to(PROJECT_ROOT)}")


def plot_stage2_regime_timeline(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    proc = PROJECT_ROOT / "data" / "processed"
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=False,
                             gridspec_kw={"height_ratios": [1.2, 1.0]})
    asset = "btc"
    feat, _ = _load(asset)
    s2 = pd.read_csv(proc / f"{asset}_regime_labels_composite_macro_v5_v5.csv",
                     index_col=0, parse_dates=True)
    s2 = s2.reindex(feat.index)

    # Panel 1: regime label shading + days_since
    _shade_label(axes[0], s2["regime_label"], REGIME_COLORS, alpha=0.45)
    axes[0].plot(feat.index, feat["regime_age_days"], color="black", lw=1.0)
    axes[0].set_title(f"{asset.upper()} — Stage 2 regime + days_since_last_transition",
                       fontsize=11, fontweight="bold")
    axes[0].set_ylabel("regime age (days)")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(feat.index.min(), feat.index.max())

    # Panel 2: ETH regime timeline
    asset = "eth"
    feat_e, _ = _load(asset)
    s2_e = pd.read_csv(proc / f"{asset}_regime_labels_composite_macro_v5_v5.csv",
                       index_col=0, parse_dates=True).reindex(feat_e.index)
    _shade_label(axes[1], s2_e["regime_label"], REGIME_COLORS, alpha=0.45)
    axes[1].plot(feat_e.index, feat_e["regime_age_days"], color="black", lw=1.0)
    axes[1].set_title(f"{asset.upper()} — Stage 2 regime + days_since_last_transition",
                       fontsize=11, fontweight="bold")
    axes[1].set_ylabel("regime age (days)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(feat_e.index.min(), feat_e.index.max())

    handles = [Patch(facecolor=c, alpha=0.6, label=l) for l, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.005), fontsize=10)
    fig.suptitle("Phase 4 — Stage 2 regime label + regime tenure feature",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out / "v5_p4_stage2_regime_timeline.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {(out / 'v5_p4_stage2_regime_timeline.png').relative_to(PROJECT_ROOT)}")


def plot_oscillator_distributions(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    osc_cols = FEATURE_GROUPS["Oscillator"]
    fig, axes = plt.subplots(2, 6, figsize=(20, 7))
    for ri, asset in enumerate(ASSETS):
        feat, _ = _load(asset)
        for ci, col in enumerate(osc_cols):
            ax = axes[ri, ci]
            data = feat[col].dropna()
            ax.hist(data, bins=50, color=ASSET_COLORS[asset], edgecolor="black",
                    linewidth=0.3, alpha=0.85)
            ax.set_title(f"{col}", fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{asset.upper()}\ncount", fontsize=10, fontweight="bold")
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(True, alpha=0.3)
            ax.text(0.97, 0.97, f"μ={data.mean():.2f}\nσ={data.std():.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("Phase 4 — Oscillator feature distributions (BTC vs ETH)",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    fig.savefig(out / "v5_p4_oscillator_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {(out / 'v5_p4_oscillator_distributions.png').relative_to(PROJECT_ROOT)}")


def main():
    out = PROJECT_ROOT / "reports" / "Phase4.0_inputs"
    out.mkdir(parents=True, exist_ok=True)

    plot_label_timeline(out)
    plot_label_distribution(out)
    plot_forward_return_hist(out)
    plot_feature_correlation(out)
    plot_stage1_raw_vs_smooth(out)
    plot_stage2_regime_timeline(out)
    plot_oscillator_distributions(out)

    print()
    print(f"All plots in {out.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
