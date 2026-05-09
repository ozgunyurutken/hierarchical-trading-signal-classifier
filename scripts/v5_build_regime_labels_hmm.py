"""
V5 Phase 2.5 — 3-state Gaussian HMM (Hamilton 1989 [N6]).

Markov-chain regime transitions with built-in persistence (smooth state
changes). Soft posterior P(regime | x) ile Stage 3 soft fusion için ideal.

Feature subset (Phase 2.4 ile aynı):
  - VIX_zscore_long
  - SP500_log_return_5d

Hamilton-Susmel 1994 SWARCH §4 — 3-state regime switching benchmark.
Bouri-Vo-Saeed 2020 — 3-regime BTC volatility classification.

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_hmm_v5.csv
  reports/Phase2/v5_p2.5_hmm_centroids.png
  reports/Phase2/v5_p2.5_hmm_transmat.png
  reports/Phase2/v5_p2.5_hmm_timeline.png
  reports/Phase2/v5_p2.5_hmm_softprob_timeline.png
  reports/Phase2/v5_p2.5_hmm_distribution.png
  reports/Phase2/v5_p2.5_hmm_diagnostics.json
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
    MinimalGaussianHMM, MINIMAL_FEATURES, BULL_BEAR_LABELS,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

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
    ax.set_ylabel("State mean (original units)")
    ax.set_title("V5 Phase 2.5 — 3-state Gaussian HMM State Means\n"
                 "Hamilton 1989 [N6], 2 features (VIX z + SP500 5d return)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_transmat(model, out: Path):
    tm = model.transition_summary()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(tm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(BULL_BEAR_LABELS)))
    ax.set_xticklabels(BULL_BEAR_LABELS)
    ax.set_yticks(range(len(BULL_BEAR_LABELS)))
    ax.set_yticklabels(BULL_BEAR_LABELS)
    ax.set_xlabel("To regime")
    ax.set_ylabel("From regime")
    for i in range(len(BULL_BEAR_LABELS)):
        for j in range(len(BULL_BEAR_LABELS)):
            v = tm.values[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("V5 Phase 2.5 — HMM Transition Matrix P(s_{t+1} | s_t)",
                 fontsize=11.5, fontweight="bold")
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
    ax2.set_ylabel("VIX z-score", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False); ax2.spines["top"].set_visible(False)
    ax.set_title("Pre-train context — S&P 500 + VIX z-score + HMM regime shading",
                 fontsize=10, fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

    _shade_regimes(axes[1], btc_close, btc_r["regime_label"], log=True, lw=1.0)
    axes[1].set_ylabel("BTC (log)")
    axes[1].set_title("BTC + HMM regime", fontsize=10, fontweight="bold")

    _shade_regimes(axes[2], eth_close, eth_r["regime_label"], log=True, lw=1.0)
    axes[2].set_ylabel("ETH (log)")
    axes[2].set_title("ETH + HMM regime", fontsize=10, fontweight="bold")

    pre_close = pd.Series(1.0, index=pre_r.index)
    _shade_regimes(axes[3], pre_close, pre_r["regime_label"], log=False, lw=0.2)
    axes[3].set_yticks([]); axes[3].grid(False)
    axes[3].set_title("Pre-train regime band only (compact)",
                      fontsize=10, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("V5 Phase 2.5 — Gaussian HMM 3-state Regime Timeline "
                 "(Bull / Neutral / Bear, Viterbi hard label)",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_softprob_timeline(pre_r, out: Path):
    """Soft posterior P(regime | x) over pretrain timeline (stacked area)."""
    df = pre_r[[f"P_{r}" for r in BULL_BEAR_LABELS]].dropna()
    fig, ax = plt.subplots(figsize=(15, 4.5))
    ax.stackplot(df.index, [df[f"P_{r}"].values for r in BULL_BEAR_LABELS],
                 labels=BULL_BEAR_LABELS,
                 colors=[REGIME_COLORS[r] for r in BULL_BEAR_LABELS],
                 alpha=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(regime | x)")
    ax.set_xlim(df.index.min(), df.index.max())
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_title("V5 Phase 2.5 — HMM Soft Posterior P(regime | x) Timeline",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
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
    ax.set_title("V5 Phase 2.5 — HMM Regime Distribution (Viterbi argmax)",
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
    print("V5 Phase 2.5 — 3-state Gaussian HMM (Hamilton 1989 [N6])")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    (reports / "Phase2").mkdir(parents=True, exist_ok=True)

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True).dropna(subset=MINIMAL_FEATURES)
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print(f"\n[1] Fit 3-state Gaussian HMM on {len(MINIMAL_FEATURES)} features: {MINIMAL_FEATURES}")
    model = MinimalGaussianHMM(n_states=3, random_state=42).fit(pretrain)
    print(f"  state → regime: {model.state_to_regime_}")
    print(f"\nState means (original units):")
    print(model.centroid_summary().round(3).to_string())
    print(f"\nTransition matrix P(s_{{t+1}} | s_t):")
    print(model.transition_summary().round(4).to_string())

    plot_centroids(model, reports / "Phase2" / "v5_p2.5_hmm_centroids.png")
    plot_transmat(model, reports / "Phase2" / "v5_p2.5_hmm_transmat.png")

    print(f"\n[2] Inference (Viterbi + soft posterior)")
    pre_r = model.predict(pretrain)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_hmm_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_hmm_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_hmm_v5.csv")

    print(f"\n[3] Plots")
    plot_timeline(btc["Close"].loc[btc_r.index],
                  eth["Close"].loc[eth_r.index],
                  btc_r, eth_r, pre_r,
                  reports / "Phase2" / "v5_p2.5_hmm_timeline.png")
    plot_softprob_timeline(pre_r,
                           reports / "Phase2" / "v5_p2.5_hmm_softprob_timeline.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "Phase2" / "v5_p2.5_hmm_distribution.png")

    print("\n" + "=" * 70)
    print("Distribution (Viterbi):")
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
        "phase": "V5 Phase 2.5 — 3-state Gaussian HMM (Hamilton 1989)",
        "method": "hmmlearn GaussianHMM full covariance",
        "feature_subset": list(MINIMAL_FEATURES),
        "n_states": 3,
        "covariance_type": "full",
        "random_state": 42,
        "state_to_regime": {int(k): v for k, v in model.state_to_regime_.items()},
        "centroids_original_units": model.centroid_summary().round(4).to_dict(),
        "transition_matrix": model.transition_summary().round(6).to_dict(),
        "distribution_pct": distribution,
    }
    diag_path = reports / "Phase2" / "v5_p2.5_hmm_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.5 complete.")


if __name__ == "__main__":
    main()
