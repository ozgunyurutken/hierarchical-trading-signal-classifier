"""
V5 Phase 2.3 — Sparse K-Means k=3 (Witten & Tibshirani 2010 [LR1]).

Phase 2.2'de saf K-Means denendi → FFR_change_60d feature 2022 Fed hike
döneminde extreme z-score (+2.98) outlier oluşturduğu için cluster yapısını
domine etti (Risk-Off cluster'ı tek başına 2022 capture etti, COVID/GFC
Neutral'a düştü).

Sparse K-Means çözümü:
  - L1-penalized feature weight w_j ∈ [0,1] iteratif öğrenilir
  - Algoritma extreme outlier feature'a düşük weight verebilir
  - Hyperparameter scan: s ∈ {1.5, sqrt(p)/2, sqrt(p)/1.5, sqrt(p)/1.2, sqrt(p)*0.95}
  - Best s seçimi: Risk-Off cluster ~12% target balance (5-25% kabul edilebilir)

Anchor mekanizmasi training'e DAHIL DEĞIL — 8 crisis period'u POST-HOC
diagnostics olarak raporda gösterilir.

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_sparse_v5.csv
  reports/Phase2/v5_p2.3_sparse_centroids.png
  reports/Phase2/v5_p2.3_sparse_feature_weights.png
  reports/Phase2/v5_p2.3_sparse_timeline.png
  reports/Phase2/v5_p2.3_sparse_distribution.png
  reports/Phase2/v5_p2.3_sparse_diagnostics.json
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
    SemanticSparseKMeans, STAGE2_FEATURES, REGIME_LABELS, CRISIS_DATE_RANGES,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {"Risk-On": "#7ec27e", "Risk-Off": "#e07e7e", "Neutral": "#f0c870"}


def _shade_anchors(ax, crisis_ranges=CRISIS_DATE_RANGES):
    """Subplot height boyu silik darkred fill + başlangıç/bitiş dashed çizgileri.

    Phase 2.2+ anchor visualization: renkler üst üste gözüksün, anchor
    başlangıç ve bitiş tarihleri net olsun."""
    for start, end, _label in crisis_ranges:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(s, e, color="darkred", alpha=0.10, zorder=0)
        ax.axvline(s, color="darkred", alpha=0.6, lw=0.8, ls="--", zorder=1)
        ax.axvline(e, color="darkred", alpha=0.6, lw=0.8, ls="--", zorder=1)


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
                           alpha=0.25, lw=0, zorder=2)
            cur, start = v, regime.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime.index[-1],
                   color=REGIME_COLORS.get(cur, "white"),
                   alpha=0.25, lw=0, zorder=2)


def plot_feature_weights(model, out: Path):
    weights = model.feature_weights_summary()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(weights)), weights.values, color="C0",
            edgecolor="black", lw=0.5)
    ax.set_yticks(range(len(weights)))
    ax.set_yticklabels(weights.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Learned weight w_j (||w||_2 = 1, ||w||_1 ≤ s)")
    ax.axvline(1 / np.sqrt(len(weights)), color="red", ls="--", lw=0.8,
               label=f"Uniform baseline 1/√p = {1/np.sqrt(len(weights)):.3f}")
    for i, v in enumerate(weights.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_title(f"V5 Phase 2.3 — Sparse K-Means Learned Feature Weights\n"
                 f"L1 bound s={model.s_:.3f} (max √p={np.sqrt(len(weights)):.2f}); "
                 f"weight≈0 → feature dropped",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
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
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE2_FEATURES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Centroid value (original units)")
    ax.set_title(f"V5 Phase 2.3 — Sparse K-Means Centroids per Regime\n"
                 f"Witten & Tibshirani 2010 [LR1] L1-weighted clustering, "
                 f"best s={model.s_:.3f}",
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

    # Panel 0 — pretrain context S&P 500 + VIX z-score + regime + anchors
    ax = axes[0]
    _shade_anchors(ax)
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
    ax.set_title("Pre-train context (2000-2025) — S&P 500 + VIX z-score + "
                 "regime shading + anchor periods (silik darkred)",
                 fontsize=10, fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=8)

    # Panel 1 — BTC
    _shade_anchors(axes[1])
    _shade_regimes(axes[1], btc_close, btc_r["regime_label"], log=True, lw=1.0)
    axes[1].set_ylabel("BTC (log)")
    axes[1].set_title("BTC + Sparse K-Means regime + anchor periods",
                      fontsize=10, fontweight="bold")

    # Panel 2 — ETH
    _shade_anchors(axes[2])
    _shade_regimes(axes[2], eth_close, eth_r["regime_label"], log=True, lw=1.0)
    axes[2].set_ylabel("ETH (log)")
    axes[2].set_title("ETH + Sparse K-Means regime + anchor periods",
                      fontsize=10, fontweight="bold")

    # Panel 3 — pretrain regime band only
    _shade_anchors(axes[3])
    pre_close = pd.Series(1.0, index=pre_r.index)
    _shade_regimes(axes[3], pre_close, pre_r["regime_label"], log=False, lw=0.2)
    axes[3].set_yticks([])
    axes[3].grid(False)
    axes[3].set_title("Pre-train regime band only (compact)",
                      fontsize=10, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    handles.append(Patch(facecolor="darkred", alpha=0.25,
                         label="Crisis anchor periods (validation only)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("V5 Phase 2.3 — Sparse K-Means k=3 Regime Timeline "
                 "(L1 feature-weighted, anchors = post-hoc validation)",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
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
    ax.set_xlabel("% of days")
    ax.set_xlim(0, 100)
    ax.set_title("V5 Phase 2.3 — Sparse K-Means Regime Distribution",
                 fontsize=12, fontweight="bold")
    for i, period in enumerate(pivot.index):
        cum = 0
        for r in REGIME_LABELS:
            v = pivot.loc[period, r]
            if v > 4:
                ax.text(cum + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold")
            cum += v
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def compute_anchor_diagnostics(pre_r) -> list[dict]:
    """Her anchor period için cluster oranlarını hesapla (post-hoc validation)."""
    rows = []
    for start, end, label in CRISIS_DATE_RANGES:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        slice_ = pre_r[(pre_r.index >= s) & (pre_r.index <= e)]
        if len(slice_) == 0:
            continue
        regimes = slice_["regime_label"]
        rows.append({
            "label": label,
            "start": start,
            "end": end,
            "n_days": int(len(slice_)),
            "risk_off_pct": float((regimes == "Risk-Off").mean() * 100),
            "neutral_pct": float((regimes == "Neutral").mean() * 100),
            "risk_on_pct": float((regimes == "Risk-On").mean() * 100),
        })
    return rows


def print_anchor_diagnostics(rows: list[dict]):
    print("\nAnchor diagnostics (post-hoc validation, NOT training constraint):")
    print(f"  {'Period':40s}  {'Days':>5}  {'Risk-Off':>9}  {'Neutral':>8}  {'Risk-On':>8}")
    print(f"  {'-'*40}  {'-'*5}  {'-'*9}  {'-'*8}  {'-'*8}")
    for r in rows:
        print(f"  {r['label'][:40]:40s}  {r['n_days']:>5}  "
              f"{r['risk_off_pct']:>8.1f}%  "
              f"{r['neutral_pct']:>7.1f}%  "
              f"{r['risk_on_pct']:>7.1f}%")


def main():
    print("V5 Phase 2.3 — Sparse K-Means k=3 (Witten-Tibshirani 2010 [LR1])")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    (reports / "Phase2").mkdir(parents=True, exist_ok=True)

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True).dropna(subset=STAGE2_FEATURES)
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    p = len(STAGE2_FEATURES)
    sqrt_p = np.sqrt(p)
    print(f"\np={p} features, max ||w||_1 = sqrt(p) = {sqrt_p:.3f}")

    # Hyperparameter scan
    print(f"\n[1] Hyperparameter scan: best s by Risk-Off balance")
    s_candidates = [1.5, sqrt_p / 2, sqrt_p / 1.5, sqrt_p / 1.2, sqrt_p * 0.95]
    best_model = None
    best_balance = -np.inf
    best_s = None

    for s in s_candidates:
        m = SemanticSparseKMeans(n_clusters=3, s=s, random_state=42).fit(pretrain)
        pre_r_tmp = m.predict(pretrain)
        counts = pre_r_tmp["regime_label"].value_counts(normalize=True)
        risk_off_pct = counts.get("Risk-Off", 0) * 100
        # Score: target Risk-Off ~12%, accept 5-25% range
        if 5 <= risk_off_pct <= 25:
            balance_score = 100 - abs(risk_off_pct - 12)
        else:
            balance_score = -abs(risk_off_pct - 12)
        weights = m.feature_weights_summary()
        nz = (weights > 0.01).sum()
        print(f"  s={s:.3f}  Risk-Off={risk_off_pct:.1f}%  "
              f"active features={nz}/{p}  balance_score={balance_score:.1f}")
        if balance_score > best_balance:
            best_balance = balance_score
            best_model = m
            best_s = s

    print(f"\n  Best s = {best_s:.3f} (balance_score={best_balance:.1f})")
    print(f"\n[2] Best model summary")
    print(f"  cluster → regime: {best_model.cluster_to_regime_}")
    print(f"\nFeature weights (sorted desc):")
    print(best_model.feature_weights_summary().round(4).to_string())
    print(f"\nCentroids (original units):")
    print(best_model.centroid_summary().round(3).to_string())

    plot_centroids(best_model, reports / "Phase2" / "v5_p2.3_sparse_centroids.png")
    plot_feature_weights(best_model, reports / "Phase2" / "v5_p2.3_sparse_feature_weights.png")

    print(f"\n[3] Inference (BTC/ETH via pretrain reindex — weekend ffill)")
    pre_r = best_model.predict(pretrain)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_sparse_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_sparse_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_sparse_v5.csv")
    print(f"  saved 3 CSV files in data/processed/")

    print(f"\n[4] Plots")
    plot_timeline(btc["Close"].loc[btc_r.index],
                  eth["Close"].loc[eth_r.index],
                  btc_r, eth_r, pre_r,
                  reports / "Phase2" / "v5_p2.3_sparse_timeline.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "Phase2" / "v5_p2.3_sparse_distribution.png")

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
        for r in REGIME_LABELS:
            v = int(c.get(r, 0))
            pct = float(v / total * 100) if total else 0.0
            distribution[key][r] = pct
            print(f"{r}: {v} ({pct:.1f}%)  ", end="")
        print()

    anchor_diag = compute_anchor_diagnostics(pre_r)
    print_anchor_diagnostics(anchor_diag)

    diagnostics = {
        "phase": "V5 Phase 2.3 — Sparse K-Means k=3 (Witten-Tibshirani 2010)",
        "method": "L1-penalized weighted K-Means",
        "n_clusters": 3,
        "random_state": 42,
        "l1_bound_s_best": float(best_s),
        "l1_bound_max_sqrt_p": float(sqrt_p),
        "n_outer_iter": int(best_model.n_outer_iter),
        "feature_weights": best_model.feature_weights_summary().round(6).to_dict(),
        "cluster_to_regime": {int(k): v for k, v in best_model.cluster_to_regime_.items()},
        "centroids_original_units": best_model.centroid_summary().round(4).to_dict(),
        "distribution_pct": distribution,
        "anchor_diagnostics": anchor_diag,
        "anchor_validation_summary": {
            "n_anchors": len(anchor_diag),
            "anchors_above_50pct_risk_off": sum(1 for r in anchor_diag
                                                 if r["risk_off_pct"] >= 50.0),
            "mean_risk_off_pct_across_anchors": float(np.mean(
                [r["risk_off_pct"] for r in anchor_diag])) if anchor_diag else 0.0,
        },
    }
    diag_path = reports / "Phase2" / "v5_p2.3_sparse_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.3 complete.")


if __name__ == "__main__":
    main()
