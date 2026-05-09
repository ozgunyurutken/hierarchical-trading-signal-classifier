"""
V5 Phase 2.4 — Az-feature K-Means k=3 (no anchors, pure unsupervised).

Önceki Phase 2.1-2.3 ablation'larında 9-feature K-Means/Sparse/GMM cluster
geometry'sinin curse-of-dimensionality + outlier features (FFR_change_60d,
M2_yoy 2020 spike) nedeniyle semantic ayrım yapamadığı görüldü. Manuel
anchor mekanizması ile zorlama "supervised classification"a dönüştü.

Phase 2.4 çözümü: Sadece 2 feature ile K-Means.
  - VIX_zscore_long  (volatility regime)
  - SP500_log_return_5d  (return momentum)

2D feature space → semantic ayrım net, anchor gerekmez.
Klasik volatility-return regime detection (Hamilton-Susmel 1994 SWARCH §3).

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_minimal_kmeans_v5.csv
  reports/Phase2/v5_p2.4_minimal_kmeans_centroids.png
  reports/Phase2/v5_p2.4_minimal_kmeans_timeline.png
  reports/Phase2/v5_p2.4_minimal_kmeans_distribution.png
  reports/Phase2/v5_p2.4_minimal_kmeans_diagnostics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.labels.v5_regime_labels import (
    MinimalKMeans, MINIMAL_FEATURES, BULL_BEAR_LABELS,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

# Bull = green, Neutral = yellow, Bear = red
REGIME_COLORS = {"Bull": "#7ec27e", "Neutral": "#f0c870", "Bear": "#e07e7e"}


def _shade_regimes(ax, price, regime, log=False, lw=1.0):
    if log:
        ax.semilogy(price.index, price.values, color="black", lw=lw, zorder=3)
    else:
        ax.plot(price.index, price.values, color="black", lw=lw, zorder=3)
    if regime.empty:
        return
    cur, start = regime.iloc[0], regime.index[0]
    for i in range(1, len(regime)):
        v = regime.iloc[i]
        if v != cur:
            if pd.notna(cur):
                ax.axvspan(start, regime.index[i],
                           color=REGIME_COLORS.get(cur, "white"),
                           alpha=0.30, lw=0, zorder=2)
            cur, start = v, regime.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime.index[-1],
                   color=REGIME_COLORS.get(cur, "white"),
                   alpha=0.30, lw=0, zorder=2)


def plot_centroids(model, out: Path):
    cent = model.centroid_summary()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(MINIMAL_FEATURES))
    width = 0.27
    for i, regime in enumerate(BULL_BEAR_LABELS):
        ax.bar(x + (i - 1) * width, cent.loc[regime].values, width, label=regime,
               color=REGIME_COLORS[regime], edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(MINIMAL_FEATURES, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Centroid value (original units)")
    ax.set_title("V5 Phase 2.4 — Az-feature K-Means k=3 Centroids\n"
                 "Pure unsupervised, 2 features (VIX z-score + SP500 5d return)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_timeline(btc_close, eth_close, btc_r, eth_r, pre_r, out: Path):
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    raw_dir = PROJECT_ROOT / "data" / "raw"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    sp500 = pd.read_csv(raw_dir / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    derived = pd.read_csv(proc_dir / "macro_derived_pretrain_v5.csv",
                          index_col=0, parse_dates=True)
    vix_z = derived["VIX_zscore_long"].dropna()

    # Panel 0 — pretrain context S&P 500 + VIX z-score + regime
    ax = axes[0]
    _shade_regimes(ax, pd.Series(np.nan, index=pre_r.index),
                   pre_r["regime_label"], log=False, lw=0)
    ax.plot(sp500.index, sp500.values, color="#1f77b4", lw=0.9,
            label="S&P 500 (left, linear)", zorder=3)
    ax.set_ylabel("S&P 500", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(vix_z.index, vix_z.values, color="#d62728", lw=0.7, alpha=0.8,
             label="VIX z-score long (right)", zorder=4)
    ax2.axhline(0, color="#d62728", ls=":", lw=0.4, alpha=0.5)
    ax2.axhline(2, color="#d62728", ls="--", lw=0.4, alpha=0.5)
    ax2.set_ylabel("VIX z-score (25-yıl baseline)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Pre-train context (2000-2025) — S&P 500 + VIX z-score + regime shading",
                 fontsize=10, fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

    # Panel 1 — BTC
    _shade_regimes(axes[1], btc_close, btc_r["regime_label"], log=True, lw=1.0)
    axes[1].set_ylabel("BTC (log)")
    axes[1].set_title("BTC + Az-feature K-Means regime",
                      fontsize=10, fontweight="bold")

    # Panel 2 — ETH
    _shade_regimes(axes[2], eth_close, eth_r["regime_label"], log=True, lw=1.0)
    axes[2].set_ylabel("ETH (log)")
    axes[2].set_title("ETH + Az-feature K-Means regime",
                      fontsize=10, fontweight="bold")

    # Panel 3 — pretrain band only
    pre_close = pd.Series(1.0, index=pre_r.index)
    _shade_regimes(axes[3], pre_close, pre_r["regime_label"], log=False, lw=0.2)
    axes[3].set_yticks([])
    axes[3].grid(False)
    axes[3].set_title("Pre-train regime band only (compact)",
                      fontsize=10, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("V5 Phase 2.4 — Az-feature K-Means k=3 Regime Timeline "
                 "(Bull / Neutral / Bear, no anchors)",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_distribution(btc_r, eth_r, pre_r, out: Path):
    rows = []
    for label, df in [("Pre-train", pre_r), ("BTC", btc_r), ("ETH", eth_r)]:
        c = df["regime_label"].value_counts(normalize=True) * 100
        for r in BULL_BEAR_LABELS:
            rows.append({"period": label, "regime": r, "pct": float(c.get(r, 0))})
    pivot = pd.DataFrame(rows).pivot(index="period", columns="regime",
                                      values="pct")[BULL_BEAR_LABELS]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[REGIME_COLORS[r] for r in BULL_BEAR_LABELS],
               edgecolor="black", lw=0.5)
    ax.set_xlabel("% of days")
    ax.set_xlim(0, 100)
    ax.set_title("V5 Phase 2.4 — Az-feature K-Means Regime Distribution",
                 fontsize=12, fontweight="bold")
    for i, period in enumerate(pivot.index):
        cum = 0
        for r in BULL_BEAR_LABELS:
            v = pivot.loc[period, r]
            if v > 4:
                ax.text(cum + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold")
            cum += v
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    print("V5 Phase 2.4 — Az-feature K-Means k=3 (Bull / Neutral / Bear)")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    (reports / "Phase2").mkdir(parents=True, exist_ok=True)

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True).dropna(subset=MINIMAL_FEATURES)
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print(f"\n[1] Fit K-Means k=3 on {len(MINIMAL_FEATURES)} features: {MINIMAL_FEATURES}")
    model = MinimalKMeans(n_clusters=3, random_state=42).fit(pretrain)
    print(f"  cluster → regime: {model.cluster_to_regime_}")
    print(f"\nCentroids (original units):")
    print(model.centroid_summary().round(3).to_string())

    plot_centroids(model, reports / "Phase2" / "v5_p2.4_minimal_kmeans_centroids.png")

    print(f"\n[2] Inference (BTC/ETH via pretrain reindex)")
    pre_r = model.predict(pretrain)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_minimal_kmeans_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_minimal_kmeans_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_minimal_kmeans_v5.csv")

    print(f"\n[3] Plots")
    plot_timeline(btc["Close"].loc[btc_r.index],
                  eth["Close"].loc[eth_r.index],
                  btc_r, eth_r, pre_r,
                  reports / "Phase2" / "v5_p2.4_minimal_kmeans_timeline.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "Phase2" / "v5_p2.4_minimal_kmeans_distribution.png")

    print("\n" + "=" * 70)
    print("Distribution:")
    distribution = {}
    for label, df, key in [("Pre-train", pre_r, "pretrain"),
                            ("BTC era", btc_r, "btc"),
                            ("ETH era", eth_r, "eth")]:
        c = df["regime_label"].value_counts()
        total = c.sum()
        print(f"  {label:12s} ", end="")
        distribution[key] = {}
        for r in BULL_BEAR_LABELS:
            v = int(c.get(r, 0))
            pct = float(v / total * 100) if total else 0.0
            distribution[key][r] = pct
            print(f"{r}: {v} ({pct:.1f}%)  ", end="")
        print()

    diagnostics = {
        "phase": "V5 Phase 2.4 — Az-feature K-Means k=3",
        "method": "sklearn KMeans, no anchors",
        "feature_subset": list(MINIMAL_FEATURES),
        "n_clusters": 3,
        "random_state": 42,
        "cluster_to_regime": {int(k): v for k, v in model.cluster_to_regime_.items()},
        "centroids_original_units": model.centroid_summary().round(4).to_dict(),
        "distribution_pct": distribution,
    }
    diag_path = reports / "Phase2" / "v5_p2.4_minimal_kmeans_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.4 complete.")


if __name__ == "__main__":
    main()
