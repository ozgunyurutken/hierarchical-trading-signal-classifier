"""
V5 Faz 2.2 — Stage 2 Macro Regime Labels (Decision Gate 2).

1. Fit K-Means k=3 on macro_derived_pretrain_v5.csv (2000-2025)
2. 4-method validation: Elbow + Silhouette + Gap + Calinski-Harabasz
3. Semantic relabeling Risk-On/Off/Neutral
4. Inference on BTC + ETH crypto-aligned macro features
5. Outputs:
   - app/models/stage2_kmeans_v5.joblib
   - data/processed/btc_regime_labels_v5.csv
   - data/processed/eth_regime_labels_v5.csv
   - data/processed/macro_pretrain_regime_labels_v5.csv
   - reports/Phase2/v5_p2_kmeans_validation.png
   - reports/Phase2/v5_p2_kmeans_centroids.png
   - reports/Phase2/v5_p2_regime_timeline.png
   - reports/Phase2/v5_p2_regime_distribution.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.labels.v5_regime_labels import (
    SemanticKMeans, validate_k, STAGE2_FEATURES, REGIME_LABELS,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {"Risk-On": "#7ec27e", "Risk-Off": "#e07e7e", "Neutral": "#f0c870"}


def plot_validation(val_result, out: Path):
    """4-panel: Elbow, Silhouette, Gap, Calinski-Harabasz."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(val_result.k_range, val_result.inertia, "o-", color="C0", lw=1.5)
    ax.axvline(3, color="red", ls=":", lw=1, alpha=0.7, label="k=3 (proposal)")
    ax.set_xlabel("k (number of clusters)"); ax.set_ylabel("Inertia (within-cluster SSE)")
    ax.set_title("1. Elbow Method (Thorndike 1953)", fontweight="bold")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(val_result.k_range, val_result.silhouette, "s-", color="C2", lw=1.5)
    ax.axvline(3, color="red", ls=":", lw=1, alpha=0.7)
    ax.axhline(0.5, color="gray", ls="--", lw=0.5, alpha=0.5, label="0.5 reasonable")
    ax.axhline(0.25, color="gray", ls=":", lw=0.5, alpha=0.5, label="0.25 weak")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette score")
    ax.set_title("2. Silhouette (Rousseeuw 1987)", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    gap = np.array(val_result.gap)
    se = np.array(val_result.gap_se)
    ax.errorbar(val_result.k_range, gap, yerr=se, fmt="o-", color="C3", lw=1.5,
                capsize=4, label="Gap ± SE")
    ax.axvline(3, color="red", ls=":", lw=1, alpha=0.7)
    # Tibshirani rule: smallest k such that Gap(k) >= Gap(k+1) - SE(k+1)
    chosen_k = None
    for i in range(len(val_result.k_range) - 1):
        if gap[i] >= gap[i + 1] - se[i + 1]:
            chosen_k = val_result.k_range[i]
            break
    if chosen_k:
        ax.axvline(chosen_k, color="green", ls="--", lw=1, alpha=0.7,
                   label=f"Tibshirani rule: k={chosen_k}")
    ax.set_xlabel("k"); ax.set_ylabel("Gap statistic")
    ax.set_title("3. Gap Statistic (Tibshirani 2001)", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(val_result.k_range, val_result.calinski_harabasz, "^-", color="C4", lw=1.5)
    ax.axvline(3, color="red", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("k"); ax.set_ylabel("Calinski-Harabasz index")
    ax.set_title("4. Variance Ratio (Caliński & Harabasz 1974)", fontweight="bold")

    fig.suptitle("V5 Decision Gate 2 — Stage 2 K-Means Cluster Validation\n"
                 "Macro pre-train data (2000-2025, 6521 bday) — proposal k=3",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_centroids(model, out: Path):
    """Bar chart: 9-feature centroids per regime."""
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
    ax.set_ylabel("Centroid value (original units)")
    ax.set_title("V5 Decision Gate 2 — K-Means Cluster Centroids per Regime\n"
                 "Semantic relabeling: lowest VIX_z + highest SP500 → Risk-On; "
                 "highest VIX_z + lowest SP500 → Risk-Off",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_regime_timeline(btc_close, eth_close, btc_regime, eth_regime,
                         pretrain_regime, out: Path):
    """Top: pre-train multi-asset (S&P+VIX+BTC) overlay + regime shading.
    Middle: BTC. Bottom: ETH."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    # 0. Pre-train context — S&P 500 (left, linear) + VIX z-score (right twin)
    ax = axes[0]
    raw_dir = PROJECT_ROOT / "data" / "raw"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    sp500 = pd.read_csv(raw_dir / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    derived = pd.read_csv(proc_dir / "macro_derived_pretrain_v5.csv",
                          index_col=0, parse_dates=True)
    vix_z = derived["VIX_zscore_long"].dropna()
    # Paint regime first (background)
    _shade_regime(ax, pd.Series(np.nan, index=pretrain_regime.index),
                  pretrain_regime["regime_label"], log=False, lw=0)
    ax.plot(sp500.index, sp500.values, color="#1f77b4", lw=0.9, label="S&P 500 (left, linear)")
    ax.set_ylabel("S&P 500", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(vix_z.index, vix_z.values, color="#d62728", lw=0.7, alpha=0.8,
             label="VIX z-score long (right)")
    ax2.axhline(0, color="#d62728", ls=":", lw=0.4, alpha=0.5)
    ax2.axhline(2, color="#d62728", ls="--", lw=0.4, alpha=0.5)
    ax2.set_ylabel("VIX z-score (25-yıl baseline)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False); ax2.spines["top"].set_visible(False)
    ax.set_title("Pre-train context (2000-2025) — S&P 500 (linear) + VIX z-score + regime shading",
                 fontsize=10, fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

    # 1. BTC
    ax = axes[1]
    _shade_regime(ax, btc_close, btc_regime["regime_label"], log=True, lw=1.0)
    ax.set_ylabel("BTC (USD, log)")
    ax.set_title("BTC price + Stage 2 regime shading (2014-09 → 2025-12)",
                 fontsize=10, fontweight="bold")

    # 2. ETH
    ax = axes[2]
    _shade_regime(ax, eth_close, eth_regime["regime_label"], log=True, lw=1.0)
    ax.set_ylabel("ETH (USD, log)")
    ax.set_title("ETH price + Stage 2 regime shading (2017-11 → 2025-12)",
                 fontsize=10, fontweight="bold")

    # 3. Pre-train regime band only (compact view)
    ax = axes[3]
    pretrain_close = pd.Series(1.0, index=pretrain_regime.index)
    _shade_regime(ax, pretrain_close, pretrain_regime["regime_label"], log=False, lw=0.2)
    ax.set_ylabel("Regime band"); ax.set_yticks([]); ax.grid(False)
    ax.set_title("Pre-train regime band only (compact, 2000-09 → 2025-12, 6521 bday)",
                 fontsize=10, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Date")

    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("V5 Decision Gate 2 — Stage 2 Regime Timeline\n"
                 "K-Means fit on pre-train (2000-2025), inference on crypto era",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def _shade_regime(ax, price_series, regime_series, log=False, lw=1.0):
    """Helper: draw price + paint background by regime runs."""
    if log:
        ax.semilogy(price_series.index, price_series.values, color="black", lw=lw)
    else:
        ax.plot(price_series.index, price_series.values, color="black", lw=lw)
    # Paint contiguous runs
    if regime_series.empty:
        return
    cur, start = regime_series.iloc[0], regime_series.index[0]
    for i in range(1, len(regime_series)):
        val = regime_series.iloc[i]
        if val != cur:
            if pd.notna(cur):
                ax.axvspan(start, regime_series.index[i],
                           color=REGIME_COLORS.get(cur, "white"), alpha=0.35, lw=0)
            cur, start = val, regime_series.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime_series.index[-1],
                   color=REGIME_COLORS.get(cur, "white"), alpha=0.35, lw=0)


def plot_regime_distribution(btc_regime, eth_regime, pretrain_regime, out: Path):
    """Stacked bar: regime % per period (pretrain / btc / eth)."""
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
    ax.set_title("V5 Decision Gate 2 — Regime Class Distribution per Period",
                 fontsize=12, fontweight="bold")
    # Annotate %
    for i, period in enumerate(pivot.index):
        cum = 0
        for r in REGIME_LABELS:
            v = pivot.loc[period, r]
            if v > 4:
                ax.text(cum + v / 2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="black")
            cum += v
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    print("V5 Faz 2.2 — Stage 2 Macro Regime Labels (Decision Gate 2)")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    models = PROJECT_ROOT / "app" / "models"
    models.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True).dropna(subset=STAGE2_FEATURES)
    btc_macro = pd.read_csv(proc / "btc_features_macro_v5.csv",
                            index_col=0, parse_dates=True)
    eth_macro = pd.read_csv(proc / "eth_features_macro_v5.csv",
                            index_col=0, parse_dates=True)
    btc = pd.read_csv(proc / "btc_aligned_v5.csv",
                      index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv",
                      index_col=0, parse_dates=True)
    print(f"  pre-train: {pretrain.shape}, BTC macro: {btc_macro.shape}, ETH macro: {eth_macro.shape}")

    # ---- Cluster validation ----
    print("\n[1] Cluster validation (k=2..8) on pre-train")
    X_pretrain = pretrain[STAGE2_FEATURES].values
    val_result = validate_k(X_pretrain, k_range=[2, 3, 4, 5, 6, 7, 8])
    print(f"  k_range:    {val_result.k_range}")
    print(f"  inertia:    {[f'{v:.1f}' for v in val_result.inertia]}")
    print(f"  silhouette: {[f'{v:.3f}' for v in val_result.silhouette]}")
    print(f"  gap:        {[f'{v:.3f}' for v in val_result.gap]}")
    print(f"  CH:         {[f'{v:.0f}' for v in val_result.calinski_harabasz]}")

    plot_validation(val_result, reports / "Phase2" / "v5_p2_kmeans_validation.png")

    # ---- Fit final K-Means k=3 ----
    print("\n[2] Fit K-Means k=3 + semantic relabel")
    model = SemanticKMeans(n_clusters=3, random_state=42).fit(pretrain)
    print(f"  cluster → regime mapping: {model.cluster_to_regime_}")
    print("\n  Centroids (original units):")
    print(model.centroid_summary().round(3).to_string())

    plot_centroids(model, reports / "Phase2" / "v5_p2_kmeans_centroids.png")

    # ---- Inference on pre-train + BTC + ETH ----
    print("\n[3] Inference (pre-train + BTC + ETH)")
    pretrain_regime = model.predict(pretrain)
    btc_regime = model.predict(btc_macro)
    eth_regime = model.predict(eth_macro)

    # Save labels
    pretrain_regime.to_csv(proc / "macro_pretrain_regime_labels_v5.csv")
    btc_regime.to_csv(proc / "btc_regime_labels_v5.csv")
    eth_regime.to_csv(proc / "eth_regime_labels_v5.csv")
    joblib.dump({"model": model, "feature_names": STAGE2_FEATURES},
                models / "stage2_kmeans_v5.joblib")
    print(f"  saved labels + model to data/processed/ + app/models/")

    # ---- Plots ----
    print("\n[4] Regime timeline plot (BTC + ETH price + regime shading)")
    plot_regime_timeline(
        btc["Close"].loc[btc_regime.index],
        eth["Close"].loc[eth_regime.index],
        btc_regime, eth_regime, pretrain_regime,
        reports / "Phase2" / "v5_p2_regime_timeline.png"
    )

    print("\n[5] Regime distribution plot")
    plot_regime_distribution(btc_regime, eth_regime, pretrain_regime,
                             reports / "Phase2" / "v5_p2_regime_distribution.png")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("V5 Faz 2.2 complete — Decision Gate 2 ready for review.")
    print("\nRegime class distribution:")
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


if __name__ == "__main__":
    main()
