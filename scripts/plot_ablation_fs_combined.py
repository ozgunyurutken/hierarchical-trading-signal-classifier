"""
Combined comparison plot: v1 (29 feat) + B1 (15 osc+vol) + B2 (24 drop 5)
+ B3 (MI top-15) — 4 ablation runs, 4 configs each, on the same XGBoost
backbone. Shows the redundancy → hierarchy-rescue narrative in one chart.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25})

LABELS = ["v1 (29 feat full)", "B1 (15 osc+vol)", "B2 (24 -5 long-trend)", "B3 (MI top-15)"]
PATHS = [
    PROJECT_ROOT / "data/labels/btc_ablation_v2_summary.csv",
    PROJECT_ROOT / "data/labels/btc_ablation_fs_b1_summary.csv",
    PROJECT_ROOT / "data/labels/btc_ablation_fs_b2_summary.csv",
    PROJECT_ROOT / "data/labels/btc_ablation_fs_b3_summary.csv",
]
COLORS = ["#777", "#a3c1da", "#7d3c98", "#d62728"]
CFGS = ["A1_flat", "A2_trend", "A3_macro", "A4_full"]


def load(p):
    df = pd.read_csv(p)
    return df[df["cfg"].isin(CFGS)].set_index("cfg").apply(pd.to_numeric, errors="coerce")


frames = [load(p) for p in PATHS]
bh = pd.read_csv(PATHS[0])
bh_row = bh[bh["cfg"] == "Buy&Hold"].iloc[0]
bh_sharpe = float(bh_row["sharpe"]); bh_ret = float(bh_row["return"])

xpos = np.arange(len(CFGS))
w = 0.21

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

for ax, col, title, fmt in [
    (axes[0, 0], "sharpe", "Sharpe Ratio", "{:.2f}"),
    (axes[0, 1], "return", "Toplam Getiri", "{:+.1%}"),
    (axes[1, 0], "max_dd", "Maximum Drawdown", "{:.1%}"),
    (axes[1, 1], "win_rate", "Win Rate", "{:.1%}"),
]:
    for i, (df, lbl, c) in enumerate(zip(frames, LABELS, COLORS)):
        offset = (i - 1.5) * w
        vals = df.loc[CFGS, col]
        ax.bar(xpos + offset, vals, w, label=lbl, color=c,
               edgecolor="#222", lw=0.4)
        for j, v in enumerate(vals):
            ax.text(xpos[j] + offset, v, fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6, rotation=0)

    if col == "sharpe":
        ax.axhline(bh_sharpe, color="black", ls="--", lw=0.8, alpha=0.7,
                   label=f"v1 B&H ({bh_sharpe:.2f})")
    if col == "return":
        ax.axhline(bh_ret, color="black", ls="--", lw=0.8, alpha=0.7,
                   label=f"v1 B&H ({bh_ret:+.1%})")
    ax.set_xticks(xpos); ax.set_xticklabels(CFGS)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.axhline(0, color="black", lw=0.4)

fig.suptitle(
    "Feature Selection Ablation — XGBoost × 4 mimari konfig × 4 tech subset"
    "  (BTC v1 verisi, test 462 gün)",
    fontsize=12, fontweight="bold", y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = PROJECT_ROOT / "reports" / "ablation_fs_combined.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out.relative_to(PROJECT_ROOT)}")

print("\nSummary across 4 subsets × 4 configs (Sharpe):")
combined = pd.DataFrame({lbl: df["sharpe"] for lbl, df in zip(LABELS, frames)})
print(combined.to_string())
print(f"\nB&H reference Sharpe: {bh_sharpe:.2f}")
