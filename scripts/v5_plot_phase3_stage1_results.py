"""V5 Phase 3 — Stage 1 training results plots.

Outputs (variant=base, default):
  reports/Phase3/v5_p3_stage1_confusion_grid.png
  reports/Phase3/v5_p3_stage1_f1_per_fold.png
  reports/Phase3/v5_p3_stage1_oof_timeline_btc.png
  reports/Phase3/v5_p3_stage1_oof_timeline_eth.png

Outputs (variant=tuned):
  reports/Phase3/v5_p3_stage1_confusion_grid_tuned.png
  reports/Phase3/v5_p3_stage1_f1_per_fold_tuned.png
  reports/Phase3/v5_p3_stage1_oof_timeline_btc_tuned.png
  reports/Phase3/v5_p3_stage1_oof_timeline_eth_tuned.png

Usage:
  python scripts/v5_plot_phase3_stage1_results.py            # baseline
  python scripts/v5_plot_phase3_stage1_results.py --variant tuned
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

from src.models.v5_stage1_trainer import LABEL_TO_IDX, CLASS_NAMES

ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
LABEL_COLORS = {"uptrend": "#7ec27e", "downtrend": "#e07e7e", "range": "#f0c870"}


def _suffix_for(variant: str) -> tuple[str, str]:
    """Return (oof_suffix, plot_suffix). variant=base -> ('', ''), tuned -> ('_tuned','_tuned')."""
    if variant == "base":
        return "", ""
    if variant == "tuned":
        return "_tuned", "_tuned"
    raise ValueError(f"Unknown variant: {variant}")


def plot_confusion_grid(out_dir: Path, variant: str):
    proc = PROJECT_ROOT / "data" / "processed"
    oof_suffix, plot_suffix = _suffix_for(variant)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    for r, asset in enumerate(ASSETS):
        for c, model in enumerate(MODELS):
            oof = pd.read_csv(proc / f"{asset}_stage1_oof_{model}_v5{oof_suffix}.csv",
                              index_col=0, parse_dates=True)
            y_true = oof["true_label"].map(LABEL_TO_IDX).to_numpy()
            y_pred = oof["pred_label"].map(LABEL_TO_IDX).to_numpy()
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            cm_norm = cm / cm.sum(axis=1, keepdims=True)

            ax = axes[r, c]
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                            ha="center", va="center", fontsize=9,
                            color="white" if cm_norm[i, j] > 0.5 else "black")
            ax.set_xticks(range(3)); ax.set_yticks(range(3))
            ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=20)
            ax.set_yticklabels(CLASS_NAMES, fontsize=9)
            if r == 1: ax.set_xlabel("predicted", fontsize=10)
            if c == 0: ax.set_ylabel(f"{asset.upper()}\ntrue", fontsize=10, fontweight="bold")
            ax.set_title(model, fontsize=11, fontweight="bold")

    title_tag = " — TUNED" if variant == "tuned" else ""
    fig.suptitle(f"Phase 3 — Stage 1 confusion matrices (row-normalized){title_tag}\n"
                 "4 classifiers x 2 assets, walk-forward OOF predictions",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / f"v5_p3_stage1_confusion_grid{plot_suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_f1_per_fold(out_dir: Path, variant: str):
    _, plot_suffix = _suffix_for(variant)
    metrics_name = "v5_p3_stage1_metrics_tuned.csv" if variant == "tuned" else "v5_p3_stage1_metrics.csv"
    metrics = pd.read_csv(out_dir / metrics_name, parse_dates=["val_start", "val_end"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    colors = {"xgboost": "#cc4444", "lightgbm": "#3a6fb0",
              "random_forest": "#2a7a2a", "mlp": "#bf6e1d"}

    for ax, asset in zip(axes, ASSETS):
        sub = metrics[metrics["asset"] == asset]
        for model in MODELS:
            r = sub[sub["model"] == model]
            ax.plot(r["val_start"], r["f1_macro"], marker="o", lw=1.5, ms=6,
                    color=colors[model], label=f"{model}  (mean {r['f1_macro'].mean():.3f})")
        ax.axhline(0.333, color="grey", ls="--", lw=0.8, label="chance (0.33)")
        ax.set_title(f"{asset.upper()} — F1 macro per walk-forward fold",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("validation start")
        if asset == ASSETS[0]: ax.set_ylabel("F1 macro")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, alpha=0.3)

    title_tag = " — TUNED" if variant == "tuned" else ""
    fig.suptitle(f"Phase 3 — Stage 1 walk-forward F1 macro stability{title_tag}",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / f"v5_p3_stage1_f1_per_fold{plot_suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_oof_timeline(out_dir: Path, asset: str, color_close: str,
                      best_model: str, variant: str):
    proc = PROJECT_ROOT / "data" / "processed"
    oof_suffix, plot_suffix = _suffix_for(variant)
    df = pd.read_csv(proc / f"{asset}_aligned_v5.csv", index_col=0, parse_dates=True)
    oof = pd.read_csv(proc / f"{asset}_stage1_oof_{best_model}_v5{oof_suffix}.csv",
                      index_col=0, parse_dates=True)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1.0]})
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    # Panel 1: price + true label shading
    cur, start = oof["true_label"].iloc[0], oof.index[0]
    for i in range(1, len(oof)):
        v = oof["true_label"].iloc[i]
        if v != cur:
            axes[0].axvspan(start, oof.index[i], color=LABEL_COLORS[cur], alpha=0.30, lw=0)
            cur, start = v, oof.index[i]
    axes[0].axvspan(start, oof.index[-1], color=LABEL_COLORS[cur], alpha=0.30, lw=0)
    close = df["Close"].reindex(oof.index)
    axes[0].semilogy(close.index, close.values, color=color_close, lw=1.0)
    axes[0].set_ylabel(f"{asset.upper()} (log)")
    axes[0].set_title(f"{asset.upper()} price + ZigZag true label (OOF span)",
                       fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: stacked OOF probabilities
    axes[1].stackplot(oof.index,
                      oof["P_downtrend"], oof["P_range"], oof["P_uptrend"],
                      colors=[LABEL_COLORS["downtrend"], LABEL_COLORS["range"], LABEL_COLORS["uptrend"]],
                      labels=["P(downtrend)", "P(range)", "P(uptrend)"], alpha=0.85)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("probability")
    axes[1].set_title(f"Stage 1 OOF probabilities ({best_model})",
                       fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper left", fontsize=9, ncol=3)
    axes[1].grid(True, alpha=0.3)

    title_tag = " — TUNED" if variant == "tuned" else ""
    fig.suptitle(f"Phase 3 — Stage 1 OOF timeline ({asset.upper()}, best model: {best_model}){title_tag}",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / f"v5_p3_stage1_oof_timeline_{asset}{plot_suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "tuned"], default="base",
                    help="Which OOF/metrics to plot. 'base' uses *_v5.csv, "
                         "'tuned' uses *_v5_tuned.csv. Default base.")
    args = ap.parse_args()

    # variant=base outputs to reports/Phase3, variant=tuned outputs to reports/Phase3.5_after_tune
    out_dir = PROJECT_ROOT / "reports" / (
        "Phase3.5_after_tune" if args.variant == "tuned" else "Phase3"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # metrics CSV co-located with the plots, but for tuned variant the f1_per_fold
    # plotter loads from out_dir; for base it loads from Phase3.
    plot_confusion_grid(out_dir, args.variant)
    plot_f1_per_fold(out_dir, args.variant)

    overall_name = ("v5_p3_stage1_overall_tuned.csv" if args.variant == "tuned"
                    else "v5_p3_stage1_overall.csv")
    overall = pd.read_csv(out_dir / overall_name)
    btc_best = overall[overall["asset"] == "btc"].sort_values("f1_macro").iloc[-1]["model"]
    eth_best = overall[overall["asset"] == "eth"].sort_values("f1_macro").iloc[-1]["model"]
    print(f"[{args.variant}] BTC best by F1 macro: {btc_best}")
    print(f"[{args.variant}] ETH best by F1 macro: {eth_best}")

    plot_oof_timeline(out_dir, "btc", "#F7931A", btc_best, args.variant)
    plot_oof_timeline(out_dir, "eth", "#627EEA", eth_best, args.variant)


if __name__ == "__main__":
    main()
