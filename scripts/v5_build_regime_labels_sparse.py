"""
V5 Phase 3 — Sparse K-Means (Witten & Tibshirani 2010 [LR1]).

L1 feature weighting otomatik UNRATE/FFR dominance'ını kontrol eder.
Hyperparameter scan: s ∈ [1.5, sqrt(p)/2, sqrt(p)/1.5, sqrt(p)/1.2, sqrt(p)*0.95]

Outputs:
  data/processed/btc_regime_labels_sparse_v5.csv
  data/processed/eth_regime_labels_sparse_v5.csv
  data/processed/macro_pretrain_regime_labels_sparse_v5.csv
  reports/v5_p3_sparse_feature_weights.png
  reports/v5_p3_sparse_centroids.png
  reports/v5_p3_sparse_timeline.png
  reports/v5_p3_sparse_distribution.png
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
    SemanticSparseKMeans, STAGE2_FEATURES, REGIME_LABELS,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {"Risk-On": "#7ec27e", "Risk-Off": "#e07e7e", "Neutral": "#f0c870"}


def plot_feature_weights(model, out: Path):
    weights = model.feature_weights_summary()
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(weights)), weights.values, color="C0", edgecolor="black", lw=0.5)
    ax.set_yticks(range(len(weights)))
    ax.set_yticklabels(weights.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Learned weight w_j (||w||_2 = 1, ||w||_1 ≤ s)")
    ax.axvline(1/np.sqrt(len(weights)), color="red", ls="--", lw=0.8,
               label=f"Uniform baseline 1/√p = {1/np.sqrt(len(weights)):.3f}")
    for i, v in enumerate(weights.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title(f"V5 Phase 3 — Sparse K-Means Learned Feature Weights\n"
                 f"L1 bound s={model.s_:.3f} (max √p={np.sqrt(len(weights)):.2f}); "
                 f"weight=0 → feature dropped",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_centroids(model, out: Path):
    cent = model.centroid_summary()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(STAGE2_FEATURES))
    width = 0.27
    for i, regime in enumerate(REGIME_LABELS):
        ax.bar(x + (i - 1) * width, cent.loc[regime].values, width, label=regime,
               color=REGIME_COLORS[regime], edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(STAGE2_FEATURES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Centroid value (original units)")
    ax.set_title(f"V5 Phase 3 — Sparse K-Means Centroids per Regime\n"
                 f"Witten & Tibshirani 2010 [LR1] L1-weighted clustering",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def _shade(ax, price, regime, log=False, lw=1.0):
    if log:
        ax.semilogy(price.index, price.values, color="black", lw=lw)
    else:
        ax.plot(price.index, price.values, color="black", lw=lw)
    if regime.empty:
        return
    cur, start = regime.iloc[0], regime.index[0]
    for i in range(1, len(regime)):
        v = regime.iloc[i]
        if v != cur:
            if pd.notna(cur):
                ax.axvspan(start, regime.index[i],
                           color=REGIME_COLORS.get(cur, "white"), alpha=0.30, lw=0)
            cur, start = v, regime.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime.index[-1],
                   color=REGIME_COLORS.get(cur, "white"), alpha=0.30, lw=0)


def plot_timeline(btc_close, eth_close, btc_r, eth_r, pre_r, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    pre_close = pd.Series(1.0, index=pre_r.index)
    _shade(axes[0], pre_close, pre_r["regime_label"], log=False, lw=0.2)
    axes[0].set_yticks([]); axes[0].grid(False)
    axes[0].set_title("Pre-train (2000-2025)", fontsize=10, fontweight="bold")
    _shade(axes[1], btc_close, btc_r["regime_label"], log=True, lw=1.0)
    axes[1].set_ylabel("BTC (log)")
    axes[1].set_title("BTC + Sparse K-Means regime", fontsize=10, fontweight="bold")
    _shade(axes[2], eth_close, eth_r["regime_label"], log=True, lw=1.0)
    axes[2].set_ylabel("ETH (log)")
    axes[2].set_title("ETH + Sparse K-Means regime", fontsize=10, fontweight="bold")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("V5 Phase 3 — Sparse K-Means Regime Timeline",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_distribution(btc_r, eth_r, pre_r, out: Path):
    rows = []
    for label, df in [("Pre-train", pre_r), ("BTC", btc_r), ("ETH", eth_r)]:
        c = df["regime_label"].value_counts(normalize=True) * 100
        for r in REGIME_LABELS:
            rows.append({"period": label, "regime": r, "pct": float(c.get(r, 0))})
    pivot = pd.DataFrame(rows).pivot(index="period", columns="regime", values="pct")[REGIME_LABELS]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[REGIME_COLORS[r] for r in REGIME_LABELS], edgecolor="black", lw=0.5)
    ax.set_xlabel("% of days"); ax.set_xlim(0, 100)
    ax.set_title("V5 Phase 3 — Sparse K-Means Regime Distribution", fontsize=12, fontweight="bold")
    for i, period in enumerate(pivot.index):
        cum = 0
        for r in REGIME_LABELS:
            v = pivot.loc[period, r]
            if v > 4:
                ax.text(cum + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold")
            cum += v
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def diagnostics(model, pre_r):
    """Crisis date P(Risk-Off) check (hard label, since SparseKMeans is hard-cluster)."""
    print("\nCrisis date diagnostics (regime_label):")
    for date_str in ["2002-07-15", "2008-10-15", "2011-08-15", "2015-08-25",
                     "2018-02-05", "2020-04-15", "2022-06-15"]:
        d = pd.Timestamp(date_str)
        if d in pre_r.index:
            label = pre_r.loc[d, "regime_label"]
        else:
            nearest = pre_r.index[pre_r.index.get_indexer([d], method="nearest")[0]]
            label = pre_r.loc[nearest, "regime_label"]
            date_str = f"{date_str}({nearest.date()})"
        print(f"  {date_str}:  {label}")


def main():
    print("V5 Phase 3 — Sparse K-Means (Witten & Tibshirani 2010)")
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

    p = len(STAGE2_FEATURES)
    sqrt_p = np.sqrt(p)
    print(f"\np={p} features, max ||w||_1 = sqrt(p) = {sqrt_p:.3f}")
    print("Hyperparameter scan: s candidates")

    # Hyperparameter scan
    s_candidates = [1.5, sqrt_p / 2, sqrt_p / 1.5, sqrt_p / 1.2, sqrt_p * 0.95]
    best_model = None
    best_balance = -np.inf

    for s in s_candidates:
        m = SemanticSparseKMeans(n_clusters=3, s=s, random_state=42).fit(pretrain)
        pre_r = m.predict(pretrain)
        counts = pre_r["regime_label"].value_counts(normalize=True)
        risk_off_pct = counts.get("Risk-Off", 0) * 100
        # Score: balance toward Risk-Off ≥ 5%, penalize too narrow (<3%) or too wide (>30%)
        if 5 <= risk_off_pct <= 25:
            balance_score = 100 - abs(risk_off_pct - 12)  # target ~12%
        else:
            balance_score = -abs(risk_off_pct - 12)
        weights = m.feature_weights_summary()
        nz = (weights > 0.01).sum()
        print(f"  s={s:.3f}  Risk-Off={risk_off_pct:.1f}%  active features={nz}/{p}  "
              f"balance_score={balance_score:.1f}")
        if balance_score > best_balance:
            best_balance = balance_score
            best_model = m
            best_s = s

    print(f"\n[1] Best s = {best_s:.3f} (balance_score={best_balance:.1f})")
    print(f"\nFeature weights:")
    print(best_model.feature_weights_summary().round(4).to_string())
    print(f"\nCentroids:")
    print(best_model.centroid_summary().round(3).to_string())

    plot_feature_weights(best_model, reports / "v5_p3_sparse_feature_weights.png")
    plot_centroids(best_model, reports / "v5_p3_sparse_centroids.png")

    print(f"\n[2] Inference")
    pre_r = best_model.predict(pretrain)
    btc_r = best_model.predict(btc_macro)
    eth_r = best_model.predict(eth_macro)

    pre_r.to_csv(proc / "macro_pretrain_regime_labels_sparse_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_sparse_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_sparse_v5.csv")

    print(f"\n[3] Plots")
    plot_timeline(btc["Close"].loc[btc_r.index],
                  eth["Close"].loc[eth_r.index],
                  btc_r, eth_r, pre_r,
                  reports / "v5_p3_sparse_timeline.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "v5_p3_sparse_distribution.png")

    print("\n" + "=" * 70)
    print("V5 Phase 3 complete.")
    print("\nDistribution:")
    for label, df in [("Pre-train", pre_r), ("BTC era", btc_r), ("ETH era", eth_r)]:
        c = df["regime_label"].value_counts()
        total = c.sum()
        print(f"  {label:12s} ", end="")
        for r in REGIME_LABELS:
            v = c.get(r, 0)
            print(f"{r}: {v} ({v/total*100:.1f}%)  ", end="")
        print()

    diagnostics(best_model, pre_r)


if __name__ == "__main__":
    main()
