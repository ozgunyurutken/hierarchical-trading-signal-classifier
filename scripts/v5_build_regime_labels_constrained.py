"""
V5 Phase 2.1 — Constrained K-Means (Wagstaff 2001 [LR2]).

Crisis date priors (must-link anchors): NBER recessions + major events
all forced into same cluster → Risk-Off.

Anchor periods (Risk-Off must-link):
  2001-03 → 2001-11   NBER dot-com recession
  2008-09 → 2009-06   NBER GFC recession (Lehman aftermath)
  2011-08 → 2011-10   US debt ceiling + Eurozone
  2015-08 → 2016-02   China devaluation + oil crash
  2018-10 → 2018-12   Q4 2018 sell-off
  2020-02 → 2020-04   COVID crash
  2022-02 → 2022-10   Fed hike + Ukraine war

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_constrained_v5.csv
  reports/Phase2/v5_p2.1_constrained_centroids.png
  reports/Phase2/v5_p2.1_constrained_timeline.png
  reports/Phase2/v5_p2.1_constrained_distribution.png
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
    SemanticConstrainedKMeans, STAGE2_FEATURES, REGIME_LABELS, CRISIS_DATE_RANGES,
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
        ax.bar(x + (i - 1) * width, cent.loc[regime].values, width, label=regime,
               color=REGIME_COLORS[regime], edgecolor="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(STAGE2_FEATURES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Centroid value (original units)")
    ax.set_title(f"V5 Phase 2.1 — Constrained K-Means Centroids per Regime\n"
                 f"COP-KMeans with {model.n_anchors_} crisis-date must-link anchors "
                 f"({len(CRISIS_DATE_RANGES)} crisis ranges)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def _shade_anchors(ax, crisis_ranges=CRISIS_DATE_RANGES):
    """Subplot height boyu silik darkred fill + başlangıç/bitiş dashed çizgileri.

    Phase 2.1+ anchor visualization: renkler üst üste gözüksün, anchor
    başlangıç ve bitiş tarihleri net olsun (alpha=0.10 fill + alpha=0.6 lines)."""
    for start, end, _label in crisis_ranges:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(s, e, color="darkred", alpha=0.10, zorder=0)
        ax.axvline(s, color="darkred", alpha=0.6, lw=0.8, ls="--", zorder=1)
        ax.axvline(e, color="darkred", alpha=0.6, lw=0.8, ls="--", zorder=1)


def _shade(ax, price, regime, log=False, lw=1.0):
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
                           alpha=0.25, lw=0, zorder=2)
            cur, start = v, regime.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime.index[-1],
                   color=REGIME_COLORS.get(cur, "white"),
                   alpha=0.25, lw=0, zorder=2)


def plot_timeline(btc_close, eth_close, btc_r, eth_r, pre_r, out: Path):
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    # 0. Pre-train context — S&P 500 (left, linear) + VIX z-score (right twin)
    raw_dir = PROJECT_ROOT / "data" / "raw"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    sp500 = pd.read_csv(raw_dir / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    derived = pd.read_csv(proc_dir / "macro_derived_pretrain_v5.csv",
                          index_col=0, parse_dates=True)
    vix_z = derived["VIX_zscore_long"].dropna()
    ax = axes[0]
    _shade_anchors(ax)
    _shade(ax, pd.Series(np.nan, index=pre_r.index),
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
    ax2.grid(False); ax2.spines["top"].set_visible(False)
    ax.set_title("Pre-train context (2000-2025) — S&P 500 + VIX z-score + "
                 "regime shading + anchor periods (silik darkred)",
                 fontsize=10, fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

    # 1. BTC
    _shade_anchors(axes[1])
    _shade(axes[1], btc_close, btc_r["regime_label"], log=True, lw=1.0)
    axes[1].set_ylabel("BTC (log)")
    axes[1].set_title("BTC + Constrained K-Means regime + anchor periods",
                      fontsize=10, fontweight="bold")

    # 2. ETH
    _shade_anchors(axes[2])
    _shade(axes[2], eth_close, eth_r["regime_label"], log=True, lw=1.0)
    axes[2].set_ylabel("ETH (log)")
    axes[2].set_title("ETH + Constrained K-Means regime + anchor periods",
                      fontsize=10, fontweight="bold")

    # 3. Pre-train regime band only
    _shade_anchors(axes[3])
    pre_close = pd.Series(1.0, index=pre_r.index)
    _shade(axes[3], pre_close, pre_r["regime_label"], log=False, lw=0.2)
    axes[3].set_yticks([]); axes[3].grid(False)
    axes[3].set_title("Pre-train regime band only (compact)",
                      fontsize=10, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    handles.append(Patch(facecolor="darkred", alpha=0.4,
                         label="Crisis anchor periods (must-link)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("V5 Phase 2.1 — Constrained K-Means Regime Timeline",
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
    pivot = pd.DataFrame(rows).pivot(index="period", columns="regime",
                                      values="pct")[REGIME_LABELS]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[REGIME_COLORS[r] for r in REGIME_LABELS],
               edgecolor="black", lw=0.5)
    ax.set_xlabel("% of days"); ax.set_xlim(0, 100)
    ax.set_title("V5 Phase 2.1 — Constrained K-Means Regime Distribution",
                 fontsize=12, fontweight="bold")
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


def diagnostics(pre_r, btc_r):
    """Crisis date capture check."""
    print("\nCrisis capture diagnostics:")
    for start, end, label in CRISIS_DATE_RANGES:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        slice_ = pre_r[(pre_r.index >= s) & (pre_r.index <= e)]
        if len(slice_) == 0:
            continue
        risk_off_pct = (slice_["regime_label"] == "Risk-Off").mean() * 100
        print(f"  {label[:35]:35s} ({start} → {end}): {risk_off_pct:.1f}% Risk-Off ({len(slice_)} bday)")


def main():
    print("V5 Phase 2.1 — Constrained K-Means (Wagstaff 2001 [LR2])")
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

    print(f"\n[1] Fit Constrained K-Means k=3 with {len(CRISIS_DATE_RANGES)} crisis ranges")
    model = SemanticConstrainedKMeans(n_clusters=3, random_state=42).fit(pretrain)
    print(f"  Anchors: {model.n_anchors_} bdays must-link → Risk-Off cluster")
    print(f"  cluster → regime: {model.cluster_to_regime_}")
    print(f"\nCentroids:")
    print(model.centroid_summary().round(3).to_string())

    plot_centroids(model, reports / "Phase2" / "v5_p2.1_constrained_centroids.png")

    print(f"\n[2] Inference (BTC/ETH via pretrain reindex — avoids warm-up NaN + weekends)")
    # CRITICAL: Use get_pretrain_labels() not predict() — predict() bypasses
    # must-link constraints and causes anchor leakage (~34% leak in V5 P2.1
    # diagnostics: 2018Q4 → 97% Risk-On, 2022 hike → 86% Neutral).
    pre_r = model.get_pretrain_labels(pretrain)
    # BTC/ETH regime: reindex pretrain regime to crypto trading days, ffill
    # weekends → Friday's regime (macro doesn't change on weekends).
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_constrained_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_constrained_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_constrained_v5.csv")

    print(f"\n[3] Plots")
    plot_timeline(btc["Close"].loc[btc_r.index],
                  eth["Close"].loc[eth_r.index],
                  btc_r, eth_r, pre_r,
                  reports / "Phase2" / "v5_p2.1_constrained_timeline.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "Phase2" / "v5_p2.1_constrained_distribution.png")

    print("\n" + "=" * 70)
    print("V5 Phase 2.1 complete.")
    print("\nDistribution:")
    for label, df in [("Pre-train", pre_r), ("BTC era", btc_r), ("ETH era", eth_r)]:
        c = df["regime_label"].value_counts()
        total = c.sum()
        print(f"  {label:12s} ", end="")
        for r in REGIME_LABELS:
            v = c.get(r, 0)
            print(f"{r}: {v} ({v/total*100:.1f}%)  ", end="")
        print()

    diagnostics(pre_r, btc_r)


if __name__ == "__main__":
    main()
