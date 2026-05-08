"""
Visualise v1 (462-day) vs v2-bigger (533-day) ablation side-by-side
to make the report's "flat baseline wins, gap widens with more data"
narrative immediate.
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

V1 = pd.read_csv(PROJECT_ROOT / "data/labels/btc_ablation_v2_summary.csv")
V2 = pd.read_csv(PROJECT_ROOT / "data/labels/btc_ablation_v2_bigger_summary.csv")
# drop B&H rows
v1 = V1[V1["cfg"] != "Buy&Hold"].set_index("cfg").apply(pd.to_numeric, errors="coerce")
v2 = V2[V2["cfg"] != "Buy&Hold"].set_index("cfg").apply(pd.to_numeric, errors="coerce")
bh1 = V1[V1["cfg"] == "Buy&Hold"].iloc[0]
bh2 = V2[V2["cfg"] == "Buy&Hold"].iloc[0]
configs = ["A1_flat", "A2_trend", "A3_macro", "A4_full"]

fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
metrics = [
    ("sharpe", "Sharpe Ratio", "{:.2f}", float(bh1["sharpe"]), float(bh2["sharpe"])),
    ("return", "Toplam Getiri", "{:+.1%}", float(bh1["return"]), float(bh2["return"])),
    ("max_dd", "Maximum Drawdown", "{:.1%}", float(bh1["max_dd"]), float(bh2["max_dd"])),
    ("win_rate", "Win Rate", "{:.1%}", None, None),
]
xpos = np.arange(len(configs))
w = 0.38
for (col, title, fmt, bh1v, bh2v), ax in zip(metrics, axes.flatten()):
    ax.bar(xpos - w/2, v1.loc[configs, col], w, color="#9b8bbf",
           edgecolor="#333", lw=0.6, label="v1 (462g)")
    ax.bar(xpos + w/2, v2.loc[configs, col], w, color="#a3c1da",
           edgecolor="#333", lw=0.6, label="v2-bigger (533g)")
    for i, c in enumerate(configs):
        v1_val = v1.loc[c, col]; v2_val = v2.loc[c, col]
        ax.text(i - w/2, v1_val, fmt.format(v1_val), ha="center",
                va="bottom" if v1_val >= 0 else "top", fontsize=7.5)
        ax.text(i + w/2, v2_val, fmt.format(v2_val), ha="center",
                va="bottom" if v2_val >= 0 else "top", fontsize=7.5)
    if bh1v is not None:
        ax.axhline(bh1v, color="#5b3a8c", ls="--", lw=0.9, alpha=0.8,
                   label=f"v1 B&H ({fmt.format(bh1v)})")
    if bh2v is not None:
        ax.axhline(bh2v, color="#1f5b87", ls=":", lw=0.9, alpha=0.8,
                   label=f"v2 B&H ({fmt.format(bh2v)})")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(xpos); ax.set_xticklabels(configs)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="best")

fig.suptitle(
    "Ablation: v1 (462g, BTC 2014-09 →) vs v2-bigger (533g, BTC 2013-02 →)\n"
    "Both runs: same 4 configs (Flat / 2-stage Trend / 2-stage Macro / 3-stage Full) × XGBoost",
    fontsize=11.5, fontweight="bold", y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = PROJECT_ROOT / "reports" / "ablation_v1_vs_bigger.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"saved: {out.relative_to(PROJECT_ROOT)}")
