"""
V5 Faz 2.1 — Feature engineering orchestrator + Decision Gate 1.5 (corr re-check).

Outputs:
  data/processed/btc_features_macro_v5.csv      (10 stage 2 features)
  data/processed/eth_features_macro_v5.csv      (10 stage 2 features)
  data/processed/btc_features_stage1_v5.csv     (11 stage 1 features)
  data/processed/eth_features_stage1_v5.csv     (11 stage 1 features)
  data/processed/btc_features_stage3_v5.csv     (15 stage 3 features)
  data/processed/eth_features_stage3_v5.csv     (15 stage 3 features)

  reports/v5_p1.5_corr_recheck_stage2.png  (Decision Gate 1.5)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.v5_macro_features import build_macro_features, PRETRAIN_START
from src.features.v5_technical_features import build_stage1_features, build_stage3_features

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# Stage 2 final features (config.yaml > stage2_features)
# NOT: VIX raw çıkarıldı — VIX_zscore_long ile lineer collinear (corr=1.0).
# 2026-05-09 Phase 1 fix: UNRATE_change_180d çıkarıldı (Risk-Off COVID-only fix)
STAGE2_FEATURES = [
    "VIX_zscore_long",
    "SP500_log_return_5d",
    "DXY_zscore_long",
    "Gold_log_return_20d",
    "FEDFUNDS_change_60d",
    "CPI_yoy_change",
    "Yield_Curve_10Y_2Y",
    "M2_yoy_change",
]


def plot_corr_recheck(btc_macro, eth_macro, out: Path):
    """Decision Gate 1.5: stage 2 features collinearity plot."""
    common_idx = btc_macro.index.intersection(eth_macro.index)
    # Add BTC + ETH Close (read aligned)
    btc_aligned = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned_v5.csv",
                              index_col=0, parse_dates=True)
    eth_aligned = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "eth_aligned_v5.csv",
                              index_col=0, parse_dates=True)

    merged = pd.concat([
        btc_aligned.loc[common_idx, "Close"].rename("BTC"),
        eth_aligned.loc[common_idx, "Close"].rename("ETH"),
        btc_macro.loc[common_idx, STAGE2_FEATURES],
    ], axis=1).dropna()

    fig, ax = plt.subplots(figsize=(11, 9.5))
    corr = merged.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Comparison stat: max absolute off-diag corr (excluding BTC↔ETH which is target leakage)
    off_diag = corr.where(~np.eye(len(corr)).astype(bool))
    macro_only = off_diag.iloc[2:, 2:]   # exclude BTC, ETH rows/cols
    max_macro_corr = macro_only.abs().max().max()
    median_macro_corr = macro_only.abs().median().median()

    ax.set_title(
        f"V5 Decision Gate 1.5 — Stage 2 Macro Feature Correlation (post-engineering)\n"
        f"BTC + ETH + 10 derived stage 2 features  ({common_idx.min().date()} → "
        f"{common_idx.max().date()}, {len(merged)} gün)\n"
        f"Max |off-diag macro corr|: {max_macro_corr:.2f}  ·  "
        f"Median |off-diag macro corr|: {median_macro_corr:.2f}\n"
        f"Faz 1 raw seviyede 0.95 küme vardı; derived features sonrası gerçek azalma kontrolü",
        fontsize=10.5, fontweight="bold"
    )
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")
    print(f"  Max |off-diag macro corr|: {max_macro_corr:.3f}")
    print(f"  Median |off-diag macro corr|: {median_macro_corr:.3f}")
    return max_macro_corr, median_macro_corr


def main():
    print("V5 Faz 2.1 — Feature Engineering Orchestrator")
    print("=" * 60)

    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"

    btc_aligned = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth_aligned = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    # ---- Macro features (BTC + ETH share macro panel) ----
    print("\n[1] Macro features (10 stage 2 derived)")
    btc_macro = build_macro_features(btc_aligned, baseline_start=PRETRAIN_START)
    eth_macro = build_macro_features(eth_aligned, baseline_start=PRETRAIN_START)
    btc_macro.to_csv(proc / "btc_features_macro_v5.csv")
    eth_macro.to_csv(proc / "eth_features_macro_v5.csv")
    print(f"  BTC macro: {btc_macro.shape}, ETH macro: {eth_macro.shape}")

    # ---- Stage 1 technical features (per coin) ----
    print("\n[2] Stage 1 technical features (11 trend features per coin)")
    btc_s1 = build_stage1_features(btc_aligned)
    eth_s1 = build_stage1_features(eth_aligned)
    btc_s1.to_csv(proc / "btc_features_stage1_v5.csv")
    eth_s1.to_csv(proc / "eth_features_stage1_v5.csv")
    print(f"  BTC stage1: {btc_s1.shape}, ETH stage1: {eth_s1.shape}")

    # ---- Stage 3 oscillator + volume features (per coin) ----
    print("\n[3] Stage 3 technical features (15 oscillator + volume per coin)")
    btc_s3 = build_stage3_features(btc_aligned)
    eth_s3 = build_stage3_features(eth_aligned)
    btc_s3.to_csv(proc / "btc_features_stage3_v5.csv")
    eth_s3.to_csv(proc / "eth_features_stage3_v5.csv")
    print(f"  BTC stage3: {btc_s3.shape}, ETH stage3: {eth_s3.shape}")

    # ---- Decision Gate 1.5: corr re-check ----
    print("\n[4] Decision Gate 1.5 — Stage 2 corr re-check plot")
    max_corr, median_corr = plot_corr_recheck(btc_macro, eth_macro,
                                              reports / "v5_p1.5_corr_recheck_stage2.png")

    print("\n" + "=" * 60)
    print(f"V5 Faz 2.1 complete.")
    print(f"  Stage 2 collinearity DROPPED from raw 0.95 cluster to:")
    print(f"    max |off-diag| = {max_corr:.3f}")
    print(f"    median |off-diag| = {median_corr:.3f}")
    if max_corr < 0.85 and median_corr < 0.40:
        print(f"  STATUS: ✓ Acceptable for K-Means cluster fitting")
    else:
        print(f"  STATUS: ⚠ High collinearity remains, K-Means may struggle")


if __name__ == "__main__":
    main()
