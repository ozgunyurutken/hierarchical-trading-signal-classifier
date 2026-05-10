"""V5 Phase 4.6 — k-threshold Ablation visualization.

Reads:
  reports/Phase4.6_k_ablation/v5_p4_k_ablation_overall.csv
  reports/Phase4.6_k_ablation/v5_p4_k_ablation_backtest.csv
  reports/Phase4.6_k_ablation/v5_p4_k_ablation_label_dist.csv

Outputs to reports/Phase4.6_k_ablation/:
  v5_p4_k_label_distribution.png   class share vs k
  v5_p4_k_metrics.png              F1m, Sharpe, Return, MaxDD vs k (line, per model)
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
ASSET_COLORS = {"btc": "#F7931A", "eth": "#627EEA"}
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
MODEL_COLORS = {
    "xgboost":       "#cc4444",
    "lightgbm":      "#3a6fb0",
    "random_forest": "#2a7a2a",
    "mlp":           "#bf6e1d",
}
RULES = ["stateful", "defensive", "prob_weighted"]
SIGNAL_COLORS = {"Buy": "#3a8a3a", "Hold": "#999999", "Sell": "#cc4444"}


def plot_label_dist(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    df = pd.read_csv(out / "v5_p4_k_ablation_label_dist.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, asset in zip(axes, ASSETS):
        sub = df[df["asset"] == asset]
        pivot = sub.pivot_table(index="k", columns="class", values="share")
        pivot = pivot[["Buy", "Hold", "Sell"]]
        bottom = np.zeros(len(pivot))
        for cls in ["Buy", "Hold", "Sell"]:
            ax.bar(pivot.index.astype(str), pivot[cls], bottom=bottom,
                   color=SIGNAL_COLORS[cls], edgecolor="black", linewidth=0.4,
                   label=cls)
            for i, (k, v) in enumerate(zip(pivot.index, pivot[cls])):
                ax.text(i, bottom[i] + v/2, f"{v:.0%}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white" if v > 0.10 else "black")
            bottom += pivot[cls].values
        ax.set_title(f"{asset.upper()}", fontsize=11, fontweight="bold")
        ax.set_xlabel("threshold k (eps = k * rolling_std_20)")
        ax.set_ylabel("class share")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)

    fig.suptitle("Phase 4.6 — Signal label class distribution vs threshold k\n"
                 "k=0.5 is V5_PLAN spec (current). Higher k -> more Hold, fewer Buy/Sell.",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out / "v5_p4_k_label_distribution.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_metrics(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    overall = pd.read_csv(out / "v5_p4_k_ablation_overall.csv")
    bt = pd.read_csv(out / "v5_p4_k_ablation_backtest.csv")

    # 4 panels: F1m / Sharpe / Return / MaxDD vs k
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # F1 macro per model
    for r_i, asset in enumerate(ASSETS):
        # F1m
        ax = axes[r_i, 0]
        sub = overall[overall["asset"] == asset]
        for model in MODELS:
            row = sub[sub["model"] == model].sort_values("k")
            ax.plot(row["k"], row["f1_macro"], marker="o", lw=1.6, ms=6,
                    color=MODEL_COLORS[model], label=model)
        ax.axhline(0.333, color="grey", ls="--", lw=0.6, label="chance")
        ax.set_xlabel("k"); ax.set_ylabel("F1 macro")
        ax.set_title(f"{asset.upper()} F1 macro", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Best Sharpe per (k, model) — across all rules
        bt_no_bh = bt[bt["model"] != "benchmark"]
        for col_i, (metric, title) in enumerate([
            ("annualized_sharpe", "Sharpe (best rule)"),
            ("total_return",      "Return (best rule)"),
            ("max_drawdown",      "MaxDD (best rule)"),
        ]):
            ax = axes[r_i, col_i + 1]
            for model in MODELS:
                ys = []
                ks = []
                for k_val in [0.4, 0.5, 0.7, 1.0]:
                    sub2 = bt_no_bh[(bt_no_bh["asset"] == asset) &
                                    (bt_no_bh["model"] == model) &
                                    (bt_no_bh["k"] == k_val)]
                    if not len(sub2):
                        continue
                    if metric == "max_drawdown":
                        # Best MaxDD = closest to 0 (least negative)
                        v = sub2[metric].max()
                    else:
                        v = sub2[metric].max()
                    ys.append(v); ks.append(k_val)
                ax.plot(ks, ys, marker="o", lw=1.6, ms=6,
                        color=MODEL_COLORS[model], label=model)
            # B&H reference
            bh = bt[(bt["model"] == "benchmark") & (bt["asset"] == asset)]
            if len(bh):
                bh_v = bh[metric].mean()
                ax.axhline(bh_v, color="black", ls="--", lw=0.8,
                           label=f"B&H ({bh_v:+.2f})")
            ax.set_xlabel("k")
            ax.set_title(f"{asset.upper()} {title}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Phase 4.6 — k-threshold ablation (F1m + best-rule backtest metrics per model)\n"
                 "k=0.5 V5_PLAN spec. Lower k = more Buy/Sell, harder classification. Higher k = more Hold.",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out / "v5_p4_k_metrics.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    out = PROJECT_ROOT / "reports" / "Phase4.6_k_ablation"
    plot_label_dist(out)
    plot_metrics(out)


if __name__ == "__main__":
    main()
