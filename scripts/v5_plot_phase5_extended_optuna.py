"""V5 Phase 5.2 — Extended Optuna (60 trial) vs Phase 4.5 (30 trial) compare.

Reads:
  reports/Phase4.5_after_tune/v5_p4_stage3_overall_tuned.csv          (30 trial)
  reports/Phase5.2_extended_optuna/v5_p5_extended_overall.csv          (60 trial)
  reports/Phase5.2_extended_optuna/v5_p5_extended_backtest.csv

Outputs:
  reports/Phase5.2_extended_optuna/v5_p5_extended_compare.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
MODEL_COLORS = {"xgboost":"#cc4444", "lightgbm":"#3a6fb0",
                "random_forest":"#2a7a2a", "mlp":"#bf6e1d"}


def main():
    rep30 = PROJECT_ROOT / "reports" / "Phase4.5_after_tune"
    rep60 = PROJECT_ROOT / "reports" / "Phase5.2_extended_optuna"

    df30 = pd.read_csv(rep30 / "v5_p4_stage3_overall_tuned.csv")
    df60 = pd.read_csv(rep60 / "v5_p5_extended_overall.csv")

    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(MODELS))
    w = 0.36

    for ax, asset in zip(axes, ASSETS):
        s30 = df30[df30["asset"] == asset].set_index("model").reindex(MODELS)
        s60 = df60[df60["asset"] == asset].set_index("model").reindex(MODELS)

        bars30 = ax.bar(x - w/2, s30["f1_macro"], width=w,
                        color="#888888", edgecolor="black", linewidth=0.5,
                        label="30 trial (Phase 4.5)")
        bars60 = ax.bar(x + w/2, s60["f1_macro"], width=w,
                        color="#3a8a3a", edgecolor="black", linewidth=0.5,
                        label="60 trial wider HP (Phase 5.2)")
        for bars in [bars30, bars60]:
            for bar in bars:
                v = bar.get_height()
                if pd.isna(v): continue
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.axhline(0.333, color="grey", ls=":", lw=0.6, label="chance (0.33)")
        ax.axhline(0.40, color="red", ls="--", lw=0.8, label="V5_PLAN gate (0.40)")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, fontsize=10)
        ax.set_ylabel("F1 macro")
        ax.set_title(f"{asset.upper()}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 0.43)

    fig.suptitle("Phase 5.2 — Extended Optuna comparison (30 vs 60 trial, wider HP space)\n"
                 "Marginal lift: BTC ~0 (saturated), ETH RF +0.013. F1m fundamental ~0.37 ceiling.",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = rep60 / "v5_p5_extended_compare.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
