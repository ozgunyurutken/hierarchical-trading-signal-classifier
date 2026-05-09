"""V5 Phase 3 — Segment-level evaluation visualization.

Reads:  reports/Phase3.5_after_tune/v5_p3_stage1_segment_metrics.csv
Output: reports/Phase3.5_after_tune/v5_p3_stage1_segment_metrics.png
        reports/Phase3.5_after_tune/v5_p3_stage1_smoothing_curve.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VARIANT_ORDER = [
    "baseline",
    "3fold tuned",
    "5fold tuned",
    "5fold + smooth5d",
    "5fold + smooth10d",
    "5fold + smooth20d",
]
ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]


def main() -> int:
    rep = PROJECT_ROOT / "reports" / "Phase3.5_after_tune"
    df = pd.read_csv(rep / "v5_p3_stage1_segment_metrics.csv")

    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    # ===== Plot 1: 4-panel summary =====
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    metrics_titles = [
        ("frame_f1_macro",            "Frame F1 macro — per-day classification",   "F1 macro"),
        ("mean_iou",                  "Mean IoU — temporal Jaccard overlap",       "IoU"),
        ("onset_f1",                  "Onset Detection F1 (±5d tolerance)",        "F1 onsets"),
        ("majority_vote_consistency", "Majority-Vote Consistency — segment-level", "consistency"),
    ]

    for ax, (metric, title, ylabel) in zip(axes.flat, metrics_titles):
        # Build heatmap-friendly matrix: rows = (asset/model), cols = variant
        pivot = (df.pivot_table(index=["asset", "model"],
                                columns="variant", values=metric)
                   .reindex(columns=VARIANT_ORDER))
        idx_labels = [f"{a.upper()}/{m}" for a, m in pivot.index]

        arr = pivot.values
        vmin, vmax = np.nanmin(arr), np.nanmax(arr)
        im = ax.imshow(arr, cmap="YlGn", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(VARIANT_ORDER)))
        ax.set_xticklabels(VARIANT_ORDER, fontsize=8, rotation=20, ha="right")
        ax.set_yticks(range(len(idx_labels)))
        ax.set_yticklabels(idx_labels, fontsize=8)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = arr[i, j]
                if np.isnan(v): continue
                # White text on dark cells
                tcol = "black" if v < (vmin + vmax) / 2 else "white"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=tcol)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label=ylabel)

    fig.suptitle(
        "Phase 3 — Stage 1 segment-level evaluation across tuning variants\n"
        "Frame F1 macro is per-day; IoU/Onset/MVC capture regime-level structure",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = rep / "v5_p3_stage1_segment_metrics.png"
    fig.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1.relative_to(PROJECT_ROOT)}")

    # ===== Plot 2: Smoothing window ablation curves =====
    smooth_df = df[df["variant"].str.startswith("5fold tuned") |
                   df["variant"].str.startswith("5fold + smooth")].copy()

    # window=0 for raw 5fold tuned
    def variant_to_window(v):
        if v == "5fold tuned":
            return 0
        if v.startswith("5fold + smooth"):
            return int(v.replace("5fold + smooth", "").replace("d", ""))
        return None

    smooth_df["window"] = smooth_df["variant"].apply(variant_to_window)
    smooth_df = smooth_df.dropna(subset=["window"]).sort_values("window")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharex=True)
    metrics_curves = [
        ("frame_f1_macro",            "Frame F1 macro"),
        ("mean_iou",                  "Mean IoU"),
        ("onset_f1",                  "Onset F1"),
        ("majority_vote_consistency", "Majority-Vote Consistency"),
    ]
    colors = {
        ("btc", "xgboost"):       "#cc4444",
        ("btc", "lightgbm"):      "#3a6fb0",
        ("btc", "random_forest"): "#2a7a2a",
        ("btc", "mlp"):           "#bf6e1d",
        ("eth", "xgboost"):       "#aa6677",
        ("eth", "lightgbm"):      "#7a99cc",
        ("eth", "random_forest"): "#88bb77",
        ("eth", "mlp"):           "#dd9966",
    }

    for ax, (metric, title) in zip(axes, metrics_curves):
        for asset in ASSETS:
            for model in MODELS:
                sub = smooth_df[(smooth_df["asset"] == asset) &
                                (smooth_df["model"] == model)]
                ls = "-" if asset == "btc" else "--"
                ax.plot(sub["window"], sub[metric], ls=ls, marker="o", lw=1.4, ms=5,
                        color=colors[(asset, model)],
                        label=f"{asset.upper()}/{model}" if metric == "frame_f1_macro" else None)
        ax.set_xlabel("smoothing window (days)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xticks([0, 5, 10, 20])

    axes[0].set_ylabel("metric value")
    axes[0].legend(fontsize=7, loc="lower right", ncol=2)
    fig.suptitle("Phase 3 — Smoothing window ablation (5-fold tuned predictions)\n"
                 "Onset F1 peaks at 5-10d, F1m/IoU/MVC monotonic up to 20d",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out2 = rep / "v5_p3_stage1_smoothing_curve.png"
    fig.savefig(out2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2.relative_to(PROJECT_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
