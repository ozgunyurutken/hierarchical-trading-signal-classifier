"""
V5 Faz 2.2 Phase 2 — Stage 2 GMM (soft posterior) + Decision Gate 2 v2.

Phase 1 (UNRATE çıkar) yetersizdi: Risk-Off cluster yine tek event'e
specialize oldu (eskiden COVID 180g, şimdi 2022 hike 168g).

Phase 2: Gaussian Mixture Model + soft posterior. Multi-severity capture.
Reference: Li et al. 2024 JMLR [LR6] rare-event mixture model.

Outputs:
  data/processed/btc_regime_labels_gmm_v5.csv
  data/processed/eth_regime_labels_gmm_v5.csv
  data/processed/macro_pretrain_regime_labels_gmm_v5.csv
  reports/v5_p2_gmm_centroids.png
  reports/v5_p2_gmm_timeline.png
  reports/v5_p2_gmm_distribution.png
  reports/v5_p2_gmm_softprob_timeline.png  (NEW: soft probability time series)
"""
from __future__ import annotations

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
    SemanticGMM, STAGE2_FEATURES, REGIME_LABELS,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {"Risk-On": "#7ec27e", "Risk-Off": "#e07e7e", "Neutral": "#f0c870"}


def plot_centroids(model, out: Path):
    cent = model.centroid_summary()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(STAGE2_FEATURES))
    width = 0.27
    for i, regime in enumerate(REGIME_LABELS):
        vals = cent.loc[regime].values
        ax.bar(x + (i - 1) * width, vals, width, label=regime,
               color=REGIME_COLORS[regime], edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(STAGE2_FEATURES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Centroid mean (original units)")
    ax.set_title(f"V5 Phase 2 — GMM Cluster Centroids (n_components=3, covariance=full)\n"
                 f"Soft posterior model — {len(STAGE2_FEATURES)} features (UNRATE_change_180d removed)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_timeline(btc_close, eth_close, btc_regime, eth_regime, pretrain_regime, out: Path):
    """Hard-label regime timeline (argmax)."""
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    pretrain_close = pd.Series(1.0, index=pretrain_regime.index)
    _shade(axes[0], pretrain_close, pretrain_regime["regime_label"], log=False, lw=0.2)
    axes[0].set_ylabel("Pre-train regime"); axes[0].set_yticks([]); axes[0].grid(False)
    axes[0].set_title("Pre-train (2000-09 → 2025-12) — argmax label", fontsize=10, fontweight="bold")

    _shade(axes[1], btc_close, btc_regime["regime_label"], log=True, lw=1.0)
    axes[1].set_ylabel("BTC (USD, log)")
    axes[1].set_title("BTC + Stage 2 GMM regime (argmax)", fontsize=10, fontweight="bold")

    _shade(axes[2], eth_close, eth_regime["regime_label"], log=True, lw=1.0)
    axes[2].set_ylabel("ETH (USD, log)")
    axes[2].set_title("ETH + Stage 2 GMM regime (argmax)", fontsize=10, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("V5 Phase 2 — GMM Regime Timeline (hard argmax)",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_softprob_timeline(pretrain_regime, out: Path):
    """SOFT posterior P(regime | x) time series — Phase 2'nin asıl katma değeri."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    for ax, regime in zip(axes, REGIME_LABELS):
        col = f"P_{regime}"
        ax.fill_between(pretrain_regime.index, 0, pretrain_regime[col],
                        color=REGIME_COLORS[regime], alpha=0.5, lw=0)
        ax.plot(pretrain_regime.index, pretrain_regime[col],
                color=REGIME_COLORS[regime], lw=0.7)
        ax.set_ylabel(f"P({regime})")
        ax.set_ylim(0, 1)
        ax.set_title(f"Soft posterior probability: {regime}",
                     fontsize=10.5, fontweight="bold")
        # Crisis annotations
        crises = [
            ("2002-07", "dot-com"),
            ("2008-10", "Lehman"),
            ("2020-04", "COVID"),
            ("2022-06", "Fed hike"),
        ]
        for date, label in crises:
            d = pd.Timestamp(date)
            if d in pretrain_regime.index or any(abs((d - pretrain_regime.index).days) < 30):
                p_val = pretrain_regime[col].asof(d)
                if pd.notna(p_val) and p_val > 0.3:
                    ax.annotate(f"{label}\n{p_val:.2f}", xy=(d, p_val),
                                xytext=(0, 8), textcoords="offset points",
                                fontsize=8, ha="center",
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.4))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")
    fig.suptitle("V5 Phase 2 — GMM Soft Posterior P(Regime | x) Time Series\n"
                 "Pre-train period (2000-2025) — multi-severity Risk-Off capture\n"
                 "Crisis dates annotated where P > 0.3",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_distribution(btc_regime, eth_regime, pretrain_regime, out: Path):
    rows = []
    for label, df in [("Pre-train (2000-2025)", pretrain_regime),
                       ("BTC crypto era (2014-2025)", btc_regime),
                       ("ETH crypto era (2017-2025)", eth_regime)]:
        counts = df["regime_label"].value_counts(normalize=True) * 100
        for r in REGIME_LABELS:
            rows.append({"period": label, "regime": r, "pct": float(counts.get(r, 0))})
    df_long = pd.DataFrame(rows)
    pivot = df_long.pivot(index="period", columns="regime", values="pct")[REGIME_LABELS]
    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[REGIME_COLORS[r] for r in REGIME_LABELS], edgecolor="black", lw=0.5)
    ax.set_xlabel("% of days"); ax.set_xlim(0, 100)
    ax.set_title("V5 Phase 2 — GMM Regime Class Distribution (argmax)",
                 fontsize=12, fontweight="bold")
    for i, period in enumerate(pivot.index):
        cum = 0
        for r in REGIME_LABELS:
            v = pivot.loc[period, r]
            if v > 4:
                ax.text(cum + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")
            cum += v
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def _shade(ax, price_series, regime_series, log=False, lw=1.0):
    if log:
        ax.semilogy(price_series.index, price_series.values, color="black", lw=lw)
    else:
        ax.plot(price_series.index, price_series.values, color="black", lw=lw)
    if regime_series.empty:
        return
    cur, start = regime_series.iloc[0], regime_series.index[0]
    for i in range(1, len(regime_series)):
        val = regime_series.iloc[i]
        if val != cur:
            if pd.notna(cur):
                ax.axvspan(start, regime_series.index[i],
                           color=REGIME_COLORS.get(cur, "white"), alpha=0.30, lw=0)
            cur, start = val, regime_series.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime_series.index[-1],
                   color=REGIME_COLORS.get(cur, "white"), alpha=0.30, lw=0)


def main():
    print("V5 Phase 2 — Stage 2 GMM (soft posterior)")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True).dropna(subset=STAGE2_FEATURES)
    btc_macro = pd.read_csv(proc / "btc_features_macro_v5.csv",
                            index_col=0, parse_dates=True)
    eth_macro = pd.read_csv(proc / "eth_features_macro_v5.csv",
                            index_col=0, parse_dates=True)
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print(f"\n[1] Fit GMM (n_components=3, covariance=full)")
    model = SemanticGMM(n_components=3, covariance_type="full", random_state=42).fit(pretrain)
    print(f"  cluster → regime: {model.cluster_to_regime_}")
    print(f"\n  Centroids (original units):")
    print(model.centroid_summary().round(3).to_string())

    plot_centroids(model, reports / "v5_p2_gmm_centroids.png")

    print(f"\n[2] Inference (soft posterior)")
    pretrain_regime = model.predict(pretrain)
    btc_regime = model.predict(btc_macro)
    eth_regime = model.predict(eth_macro)

    pretrain_regime.to_csv(proc / "macro_pretrain_regime_labels_gmm_v5.csv")
    btc_regime.to_csv(proc / "btc_regime_labels_gmm_v5.csv")
    eth_regime.to_csv(proc / "eth_regime_labels_gmm_v5.csv")

    print(f"\n[3] Plots")
    plot_timeline(btc["Close"].loc[btc_regime.index],
                  eth["Close"].loc[eth_regime.index],
                  btc_regime, eth_regime, pretrain_regime,
                  reports / "v5_p2_gmm_timeline.png")
    plot_distribution(btc_regime, eth_regime, pretrain_regime,
                      reports / "v5_p2_gmm_distribution.png")
    plot_softprob_timeline(pretrain_regime,
                           reports / "v5_p2_gmm_softprob_timeline.png")

    print("\n" + "=" * 70)
    print("V5 Phase 2 complete.")
    print("\nDistribution (argmax):")
    for label, df in [("Pre-train", pretrain_regime),
                       ("BTC crypto era", btc_regime),
                       ("ETH crypto era", eth_regime)]:
        counts = df["regime_label"].value_counts()
        total = counts.sum()
        print(f"  {label:25s} ", end="")
        for r in REGIME_LABELS:
            v = counts.get(r, 0)
            print(f"{r}: {v} ({v/total*100:.1f}%)  ", end="")
        print()

    # Soft posterior diagnostics for known crises
    print("\nSoft posterior diagnostics — P(Risk-Off) at key crisis dates:")
    for date_str in ["2002-07-15", "2008-10-15", "2011-08-15", "2020-04-15", "2022-06-15"]:
        d = pd.Timestamp(date_str)
        if d in pretrain_regime.index:
            p_off = pretrain_regime.loc[d, "P_Risk-Off"]
            p_on = pretrain_regime.loc[d, "P_Risk-On"]
            p_neu = pretrain_regime.loc[d, "P_Neutral"]
            print(f"  {date_str}:  Risk-Off={p_off:.2f}  Risk-On={p_on:.2f}  Neutral={p_neu:.2f}")
        else:
            # Find nearest
            nearest = pretrain_regime.index[pretrain_regime.index.get_indexer([d], method="nearest")[0]]
            p_off = pretrain_regime.loc[nearest, "P_Risk-Off"]
            p_on = pretrain_regime.loc[nearest, "P_Risk-On"]
            p_neu = pretrain_regime.loc[nearest, "P_Neutral"]
            print(f"  {date_str} ({nearest.date()} nearest):  Risk-Off={p_off:.2f}  Risk-On={p_on:.2f}  Neutral={p_neu:.2f}")


if __name__ == "__main__":
    main()
