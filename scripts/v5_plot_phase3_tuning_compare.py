"""V5 Phase 3 — Pre vs Post HP-tuning comparison plot.

Reads:
  reports/Phase3/v5_p3_stage1_overall.csv                          (baseline)
  reports/Phase3.5_after_tune/v5_p3_stage1_overall_tuned.csv       (5-fold tuned)
  reports/Phase3.5_after_tune/v5_p3_stage1_overall_tuned_3fold.csv (3-fold tuned, archive)

Output:
  reports/Phase3.5_after_tune/v5_p3_stage1_tuning_compare.png
    4-panel:
      (1) F1 macro per model x asset, baseline vs 3-fold vs 5-fold
      (2) Range F1 per model x asset, same series
      (3) Tuning Δ vs baseline (heatmap, both 3-fold and 5-fold)
      (4) Decision-gate summary: # models passing F1m >= 0.50
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
DECISION_GATE = 0.50


def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)[["asset", "model", "accuracy", "f1_macro", "f1_range"]]
    df["variant"] = label
    return df


def main():
    rep_base = PROJECT_ROOT / "reports" / "Phase3"
    rep_tune = PROJECT_ROOT / "reports" / "Phase3.5_after_tune"
    rep_tune.mkdir(parents=True, exist_ok=True)

    base  = _load(rep_base / "v5_p3_stage1_overall.csv", "baseline")
    tuned = _load(rep_tune / "v5_p3_stage1_overall_tuned.csv", "5-fold tuned")
    archived_3fold = rep_tune / "v5_p3_stage1_overall_tuned_3fold.csv"
    if archived_3fold.exists():
        tuned3 = _load(archived_3fold, "3-fold tuned")
    else:
        tuned3 = None

    long_df = pd.concat([base, tuned3, tuned] if tuned3 is not None else [base, tuned],
                        ignore_index=True)
    variants = ["baseline", "3-fold tuned", "5-fold tuned"] if tuned3 is not None else \
               ["baseline", "5-fold tuned"]

    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.20)

    # --- (1) F1 macro grouped bar ---
    ax1 = fig.add_subplot(gs[0, 0])
    _grouped_bar(ax1, long_df, "f1_macro", variants,
                 title="F1 macro — baseline vs tuned variants",
                 ylabel="F1 macro", hline=DECISION_GATE,
                 hline_label=f"decision gate ({DECISION_GATE:.2f})")

    # --- (2) Range F1 grouped bar ---
    ax2 = fig.add_subplot(gs[0, 1])
    _grouped_bar(ax2, long_df, "f1_range", variants,
                 title="Range F1 (most-imbalanced class)",
                 ylabel="F1 (range)", hline=None)

    # --- (3) Δ vs baseline heatmap ---
    ax3 = fig.add_subplot(gs[1, 0])
    _delta_heatmap(ax3, long_df)

    # --- (4) Decision-gate summary ---
    ax4 = fig.add_subplot(gs[1, 1])
    _gate_summary(ax4, long_df, variants)

    fig.suptitle("Phase 3 — Stage 1 hyperparameter tuning ablation\n"
                 "Baseline (fixed HP)  vs  3-fold inner CV tuning (overfit)  vs  5-fold inner CV tuning",
                 fontsize=14, fontweight="bold", y=0.995)

    out = rep_tune / "v5_p3_stage1_tuning_compare.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def _grouped_bar(ax, long_df, metric, variants, title, ylabel, hline=None, hline_label=None):
    colors = {"baseline": "#888888", "3-fold tuned": "#cc8844", "5-fold tuned": "#3a8a3a"}
    n_var = len(variants)
    width = 0.85 / n_var
    asset_model = [(a, m) for a in ASSETS for m in MODELS]
    x = np.arange(len(asset_model))

    for vi, var in enumerate(variants):
        sub = long_df[long_df["variant"] == var]
        ys = []
        for a, m in asset_model:
            row = sub[(sub["asset"] == a) & (sub["model"] == m)]
            ys.append(row[metric].iloc[0] if len(row) else np.nan)
        ax.bar(x + vi * width - 0.425 + width / 2, ys, width=width,
               color=colors[var], label=var, edgecolor="black", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a.upper()}\n{m}" for a, m in asset_model], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    if hline is not None:
        ax.axhline(hline, color="red", ls="--", lw=0.8, label=hline_label)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, max(0.8, long_df[metric].max() * 1.1))


def _delta_heatmap(ax, long_df):
    base = long_df[long_df["variant"] == "baseline"].set_index(["asset", "model"])
    rows = []
    variants = [v for v in ["3-fold tuned", "5-fold tuned"] if v in long_df["variant"].values]
    for a in ASSETS:
        for m in MODELS:
            row = {"label": f"{a.upper()}/{m}"}
            for v in variants:
                vsub = long_df[(long_df["variant"] == v) &
                               (long_df["asset"] == a) & (long_df["model"] == m)]
                if len(vsub):
                    delta = vsub["f1_macro"].iloc[0] - base.loc[(a, m), "f1_macro"]
                    row[v] = delta
                else:
                    row[v] = np.nan
            rows.append(row)
    delta_df = pd.DataFrame(rows).set_index("label")[variants]
    arr = delta_df.values

    vmax = max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_yticks(range(len(delta_df)))
    ax.set_yticklabels(delta_df.index, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    fontsize=9, color="black",
                    fontweight="bold" if abs(v) > 0.03 else "normal")
    ax.set_title("Δ F1 macro vs baseline\n(green = improvement, red = regression)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)


def _gate_summary(ax, long_df, variants):
    rows = []
    for v in variants:
        sub = long_df[long_df["variant"] == v]
        n_pass = int((sub["f1_macro"] >= DECISION_GATE).sum())
        n_total = len(sub)
        rows.append({"variant": v, "pass": n_pass, "fail": n_total - n_pass})
    df = pd.DataFrame(rows)

    x = np.arange(len(df))
    ax.bar(x, df["pass"], color="#3a8a3a", edgecolor="black", lw=0.5, label="PASS")
    ax.bar(x, df["fail"], bottom=df["pass"], color="#cc4444", edgecolor="black", lw=0.5, label="FAIL")
    for i, (p, f) in enumerate(zip(df["pass"], df["fail"])):
        ax.text(i, p / 2, f"PASS {p}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=12)
        if f:
            ax.text(i, p + f / 2, f"FAIL {f}", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], fontsize=10)
    ax.set_ylabel("# models  (out of 8)")
    ax.set_title(f"Decision gate: F1 macro ≥ {DECISION_GATE:.2f}",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 8.5)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")


if __name__ == "__main__":
    main()
