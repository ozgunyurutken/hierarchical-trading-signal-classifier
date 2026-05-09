"""V5 Phase 3 — Stage 1 ground truth vs predicted timeline.

Output:
  reports/Phase3/v5_p3_stage1_truth_vs_pred_btc.png
  reports/Phase3/v5_p3_stage1_truth_vs_pred_eth.png

Per asset, 4-panel layout (best model: Random Forest):
  1. Price + ZigZag GROUND TRUTH label shading
  2. Price + PREDICTED label shading (argmax of OOF probabilities)
  3. AGREEMENT/ERROR mask (green agreement, red mismatch)
  4. Per-period accuracy rolling (60-day rolling agreement rate)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

LABEL_COLORS = {"uptrend": "#7ec27e", "downtrend": "#e07e7e", "range": "#f0c870"}
ERROR_COLOR = "#cc2222"
AGREE_COLOR = "#5a9a5a"


def shade(ax, label_series, alpha=0.30):
    s = label_series.dropna()
    if s.empty:
        return
    cur, start = s.iloc[0], s.index[0]
    for i in range(1, len(s)):
        v = s.iloc[i]
        if v != cur:
            ax.axvspan(start, s.index[i], color=LABEL_COLORS.get(cur, "white"), alpha=alpha, lw=0)
            cur, start = v, s.index[i]
    ax.axvspan(start, s.index[-1], color=LABEL_COLORS.get(cur, "white"), alpha=alpha, lw=0)


def shade_correctness(ax, agreement_series):
    s = agreement_series
    if s.empty:
        return
    cur, start = s.iloc[0], s.index[0]
    for i in range(1, len(s)):
        v = s.iloc[i]
        if v != cur:
            color = AGREE_COLOR if cur else ERROR_COLOR
            ax.axvspan(start, s.index[i], color=color, alpha=0.50, lw=0)
            cur, start = v, s.index[i]
    color = AGREE_COLOR if cur else ERROR_COLOR
    ax.axvspan(start, s.index[-1], color=color, alpha=0.50, lw=0)


def plot_truth_vs_pred(asset: str, model: str, color_close: str, out_path: Path):
    proc = PROJECT_ROOT / "data" / "processed"
    df = pd.read_csv(proc / f"{asset}_aligned_v5.csv", index_col=0, parse_dates=True)
    oof = pd.read_csv(proc / f"{asset}_stage1_oof_{model}_v5.csv", index_col=0, parse_dates=True)

    close = df["Close"].reindex(oof.index)
    truth = oof["true_label"]
    pred = oof["pred_label"]
    agree = (truth == pred)
    rolling_acc = agree.astype(float).rolling(60, min_periods=20).mean()

    n_total = len(oof)
    n_correct = int(agree.sum())
    overall_acc = n_correct / n_total

    # Per-true-class breakdown
    breakdown = []
    for cls in ["downtrend", "range", "uptrend"]:
        mask = truth == cls
        n_cls = int(mask.sum())
        n_correct_cls = int((agree & mask).sum())
        breakdown.append((cls, n_cls, n_correct_cls, n_correct_cls / max(n_cls, 1)))

    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1.4, 0.5, 0.8]})
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    # Panel 1 — TRUTH
    shade(axes[0], truth)
    axes[0].semilogy(close.index, close.values, color=color_close, lw=1.0)
    axes[0].set_ylabel(f"{asset.upper()} (log)")
    axes[0].set_title("Ground truth — ZigZag offline label", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — PRED
    shade(axes[1], pred)
    axes[1].semilogy(close.index, close.values, color=color_close, lw=1.0)
    axes[1].set_ylabel(f"{asset.upper()} (log)")
    axes[1].set_title(f"Predicted (argmax) — Stage 1 {model} OOF", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — agreement / disagreement bar
    shade_correctness(axes[2], agree)
    axes[2].set_yticks([])
    axes[2].set_ylim(0, 1)
    axes[2].set_title(f"Agreement (green) vs Mismatch (red) — overall {overall_acc:.1%}",
                      fontsize=11, fontweight="bold")

    # Panel 4 — rolling 60-day accuracy
    axes[3].plot(rolling_acc.index, rolling_acc.values, color="#3a6fb0", lw=1.2)
    axes[3].axhline(0.333, color="grey", ls="--", lw=0.6, label="chance (0.33)")
    axes[3].axhline(overall_acc, color="#cc4444", ls=":", lw=0.8, label=f"overall {overall_acc:.2f}")
    axes[3].set_ylim(0, 1)
    axes[3].set_ylabel("rolling accuracy")
    axes[3].set_title("60-day rolling agreement rate (truth vs predicted)",
                      fontsize=11, fontweight="bold")
    axes[3].legend(loc="lower right", fontsize=9)
    axes[3].grid(True, alpha=0.3)

    handles = [Patch(facecolor=c, alpha=0.5, label=l) for l, c in LABEL_COLORS.items()]
    handles += [Patch(facecolor=AGREE_COLOR, alpha=0.6, label="agreement"),
                Patch(facecolor=ERROR_COLOR, alpha=0.6, label="mismatch")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.005), fontsize=10)

    breakdown_text = "  |  ".join([f"{cls} acc {acc:.2%} ({nc}/{nt})"
                                    for cls, nt, nc, acc in breakdown])
    fig.suptitle(f"Phase 3 — Stage 1 truth vs predicted ({asset.upper()}, {model})\n"
                 f"OOF span: {oof.index.min().date()} → {oof.index.max().date()}, "
                 f"n={n_total}\nPer-class: {breakdown_text}",
                 fontsize=12, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}  (overall acc {overall_acc:.4f})")


def main():
    out_dir = PROJECT_ROOT / "reports" / "Phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    overall = pd.read_csv(out_dir / "v5_p3_stage1_overall.csv")
    btc_best = overall[overall["asset"] == "btc"].sort_values("f1_macro").iloc[-1]["model"]
    eth_best = overall[overall["asset"] == "eth"].sort_values("f1_macro").iloc[-1]["model"]

    plot_truth_vs_pred("btc", btc_best, "#F7931A",
                       out_dir / f"v5_p3_stage1_truth_vs_pred_btc.png")
    plot_truth_vs_pred("eth", eth_best, "#627EEA",
                       out_dir / f"v5_p3_stage1_truth_vs_pred_eth.png")


if __name__ == "__main__":
    main()
