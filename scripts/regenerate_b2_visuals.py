"""
Regenerate the v2/feature-selection final-version plots from the B2
7-model retrain artefacts. After this, only the truly current plots
remain in reports/.

Outputs:
  reports/b2_summary_table.png   — 7 model + B&H summary table image
  reports/b2_metric_bars.png     — Sharpe / Return / MCC / Win% bars
  reports/b2_equity_curves.png   — 7 equity curves + B&H, log scale
  reports/b2_cm_grid.png         — 7-panel confusion matrix grid
  reports/b2_pred_distribution.png  — 7-panel pred class distribution
  reports/b2_roc_grid.png        — 7-panel ROC OvR curves
  reports/v1_vs_b2_models.png    — head-to-head Sharpe + Return vs v1
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.evaluation.backtester import Backtester

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

CLASSES = ["Sell", "Hold", "Buy"]
MODEL_ORDER_AB = ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]
MODEL_ORDER_ZZ = ["xgboost", "mlp"]
MODEL_LABEL_AB = {"lda": "LDA", "mlp": "MLP", "xgboost": "XGB",
                  "lightgbm": "LGBM", "random_forest": "RF"}
PALETTE_AB = {"lda": "#1f77b4", "mlp": "#ff7f0e", "xgboost": "#2ca02c",
              "lightgbm": "#d62728", "random_forest": "#9467bd"}
PALETTE_ZZ = {"xgboost": "#8c564b", "mlp": "#e377c2"}


def load_data():
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    sig = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_test_signals_b2.csv",
                      index_col=0, parse_dates=True)
    sig_zz = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_test_signals_b2_zigzag.csv",
                         index_col=0, parse_dates=True)
    final = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "final_iter6_b2_summary.csv")
    bt_summary = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_backtest_b2_summary.csv")
    return btc, sig, sig_zz, final, bt_summary


def plot_summary_table(final: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    cols = list(final.columns)
    cell_text = final.astype(str).values.tolist()

    table = ax.table(cellText=cell_text, colLabels=cols, loc="center",
                     cellLoc="center", colLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10)
    table.scale(1, 1.6)
    for j in range(len(cols)):
        table[0, j].set_facecolor("#1f3a5f"); table[0, j].set_text_props(color="white", weight="bold")
    # highlight ZZ-MLP row (best Sharpe)
    zz_mlp_idx = final.index[final["Model"].astype(str).eq("ZZ-MLP")].tolist()
    if zz_mlp_idx:
        r = zz_mlp_idx[0] + 1
        for j in range(len(cols)):
            table[r, j].set_facecolor("#fff4b8")
    # highlight XGB row
    xgb_idx = final.index[final["Model"].astype(str).eq("XGB")].tolist()
    if xgb_idx:
        r = xgb_idx[0] + 1
        for j in range(len(cols)):
            table[r, j].set_facecolor("#cfe5ff")
    bh_idx = final.index[final["Model"].astype(str).eq("Buy & Hold")].tolist()
    if bh_idx:
        r = bh_idx[0] + 1
        for j in range(len(cols)):
            table[r, j].set_facecolor("#eee")

    plt.title("v2/feature-selection B2 — Final 7-Model Test Summary "
              "(BTC 462-gün test seti)\n"
              "★ ZZ-MLP Sharpe 1.68 / Return +89.5% — proje rekoru, B&H'ı +41.9pp geçti",
              fontsize=11, fontweight="bold", pad=16)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.name}")


def plot_metric_bars(final: pd.DataFrame, out: Path):
    df = final[final["Model"] != "Buy & Hold"].copy()
    bh = final[final["Model"] == "Buy & Hold"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    metrics = [
        ("Sharpe", "Sharpe Ratio", "{:.2f}", float(bh["Sharpe"])),
        ("Return", "Toplam Getiri", None, bh["Return"]),
        ("MCC", "Matthews Correlation", "{:.3f}", None),
        ("Win%", "Win Rate", None, None),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

    for ax, (col, title, fmt, bh_val) in zip(axes.flatten(), metrics):
        if col in ("Return", "Win%"):
            vals = df[col].astype(str).str.rstrip("%").replace({"-": np.nan}).astype(float)
        else:
            vals = df[col].astype(float)
        bars = ax.bar(df["Model"], vals, color=colors[:len(df)], edgecolor="#333", lw=0.7)
        for b, v in zip(bars, vals):
            label = (f"{v:.2f}" if col in ("Sharpe", "MCC")
                     else f"{v:+.1f}%" if col == "Return"
                     else f"{v:.1f}%")
            ax.text(b.get_x()+b.get_width()/2, v, label,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", lw=0.4)
        if bh_val is not None and bh_val != "-":
            try:
                bh_num = (float(bh_val) if col == "Sharpe"
                          else float(str(bh_val).rstrip("%")))
                ax.axhline(bh_num, color="darkred", ls="--", lw=1,
                           label=f"B&H ({bh_num:.2f}{'%' if col=='Return' else ''})")
                ax.legend(fontsize=8)
            except (ValueError, TypeError):
                pass
        ax.set_title(title, fontsize=11, fontweight="bold")

    fig.suptitle("v2/feature-selection B2 — 7 Modelin Test Performansı",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.name}")


def plot_equity(btc, sig, sig_zz, out: Path):
    bt = Backtester()
    test_idx = sig.index
    close = btc["Close"].loc[test_idx]
    bh = bt.run_buy_and_hold(close)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(bh.index, bh / bh.iloc[0], color="black", lw=1.5, ls="--",
            label=f"Buy & Hold ({(bh.iloc[-1]/bh.iloc[0]-1)*100:+.1f}%)")
    for clf in MODEL_ORDER_AB:
        col = f"{clf}_pred"
        if col not in sig.columns: continue
        eq = bt.run(sig[col], close)["equity_curve"]
        eq_n = eq / eq.iloc[0]
        ret = (eq.iloc[-1]/eq.iloc[0] - 1) * 100
        ax.plot(eq_n.index, eq_n, lw=1.4, color=PALETTE_AB[clf],
                label=f"B2 {MODEL_LABEL_AB[clf]} ({ret:+.1f}%)")
    test_idx_z = sig_zz.index; close_z = btc["Close"].loc[test_idx_z]
    for clf in MODEL_ORDER_ZZ:
        col = f"{clf}_pred"
        if col not in sig_zz.columns: continue
        eq = bt.run(sig_zz[col], close_z)["equity_curve"]
        eq_n = eq / eq.iloc[0]
        ret = (eq.iloc[-1]/eq.iloc[0] - 1) * 100
        ls = "-." if clf == "mlp" else ":"
        ax.plot(eq_n.index, eq_n, lw=1.4, color=PALETTE_ZZ[clf], ls=ls,
                label=f"B2 ZZ-{MODEL_LABEL_AB[clf]} ({ret:+.1f}%)")

    ax.set_yscale("log")
    ax.set_title("v2/feature-selection B2 — Equity Curves (test seti, log skala)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Equity (normalized)")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.name}")


def plot_cm_grid(sig, sig_zz, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()
    panels = [(sig, clf, MODEL_LABEL_AB[clf]) for clf in MODEL_ORDER_AB] + \
             [(sig_zz, clf, f"ZZ-{MODEL_LABEL_AB[clf]}") for clf in MODEL_ORDER_ZZ]

    for idx, (ax, (df, clf, lbl)) in enumerate(zip(axes, panels)):
        col = f"{clf}_pred"
        cm = confusion_matrix(df["y_true"], df[col], labels=CLASSES)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]*100:.0f}%)",
                        ha="center", va="center", fontsize=8.5,
                        color="white" if cmn[i,j] > 0.5 else "black")
        ax.set_xlabel("Predicted", fontsize=8.5)
        if (idx % 4) == 0:
            ax.set_ylabel("Actual", fontsize=8.5)
        ax.grid(False)
    # Hide last empty axis
    axes[-1].axis("off")
    fig.suptitle("v2/feature-selection B2 — Confusion Matrix Grid (test seti)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.name}")


def plot_pred_dist(sig, sig_zz, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    axes = axes.flatten()
    panels = [(sig, clf, MODEL_LABEL_AB[clf]) for clf in MODEL_ORDER_AB] + \
             [(sig_zz, clf, f"ZZ-{MODEL_LABEL_AB[clf]}") for clf in MODEL_ORDER_ZZ]

    for ax, (df, clf, lbl) in zip(axes, panels):
        col = f"{clf}_pred"
        true_dist = df["y_true"].value_counts().reindex(CLASSES, fill_value=0)
        pred_dist = df[col].value_counts().reindex(CLASSES, fill_value=0)
        x = np.arange(3); w = 0.35
        ax.bar(x - w/2, true_dist, w, label="Actual", color="#1f3a5f", alpha=0.7)
        ax.bar(x + w/2, pred_dist, w, label="Predicted", color="#d62728", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(CLASSES)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    fig.suptitle("v2/feature-selection B2 — Prediction vs Actual Class Distribution",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.name}")


def plot_roc_grid(sig, sig_zz, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5))
    axes = axes.flatten()
    panels = [(sig, clf, MODEL_LABEL_AB[clf]) for clf in MODEL_ORDER_AB] + \
             [(sig_zz, clf, f"ZZ-{MODEL_LABEL_AB[clf]}") for clf in MODEL_ORDER_ZZ]

    for ax, (df, clf, lbl) in zip(axes, panels):
        y_true_bin = label_binarize(df["y_true"], classes=CLASSES)
        proba_cols = [f"{clf}_proba_{c}" for c in CLASSES]
        if not all(c in df.columns for c in proba_cols):
            ax.text(0.5, 0.5, f"no proba", ha="center", va="center")
            ax.set_title(lbl); continue
        y_score = df[proba_cols].values
        for i, c in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=1.4, label=f"{c} (AUC={roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("FPR", fontsize=8.5); ax.set_ylabel("TPR", fontsize=8.5)
        ax.set_title(lbl, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
    axes[-1].axis("off")
    fig.suptitle("v2/feature-selection B2 — ROC Curves (one-vs-rest, test seti)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.name}")


def plot_v1_vs_b2(out: Path):
    """Bar comparison v1 vs B2 across 7 models on Sharpe and Return."""
    v1 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "final_iter2_summary_table.csv") \
        if (PROJECT_ROOT / "data" / "labels" / "final_iter2_summary_table.csv").exists() else None
    b2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "final_iter6_b2_summary.csv")
    if v1 is None:
        print("  skip v1_vs_b2: no v1 final_iter2_summary_table.csv on this branch")
        return

    common = ["LDA", "MLP", "XGB", "LGBM", "RF", "ZZ-XGB", "ZZ-MLP"]
    v1m = v1.set_index("Model").reindex(common)
    b2m = b2.set_index("Model").reindex(common)
    v1_sh = pd.to_numeric(v1m["Sharpe"], errors="coerce")
    b2_sh = pd.to_numeric(b2m["Sharpe"], errors="coerce")
    v1_re = v1m["Return"].astype(str).str.rstrip("%").replace("-", np.nan).astype(float)
    b2_re = b2m["Return"].astype(str).str.rstrip("%").replace("-", np.nan).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(common)); w = 0.4

    ax = axes[0]
    ax.bar(x - w/2, v1_sh, w, label="v1 (29 feat)", color="#9b8bbf", edgecolor="#333", lw=0.5)
    ax.bar(x + w/2, b2_sh, w, label="B2 (24 feat)", color="#1f5b87", edgecolor="#333", lw=0.5)
    for i, (a, b) in enumerate(zip(v1_sh, b2_sh)):
        ax.text(i - w/2, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.75, color="darkred", ls="--", lw=1, label="B&H (0.75)")
    ax.set_xticks(x); ax.set_xticklabels(common)
    ax.set_title("Sharpe Ratio: v1 vs B2", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(x - w/2, v1_re, w, label="v1 (29 feat)", color="#9b8bbf", edgecolor="#333", lw=0.5)
    ax.bar(x + w/2, b2_re, w, label="B2 (24 feat)", color="#1f5b87", edgecolor="#333", lw=0.5)
    for i, (a, b) in enumerate(zip(v1_re, b2_re)):
        ax.text(i - w/2, a, f"{a:+.1f}%", ha="center",
                va="bottom" if a >= 0 else "top", fontsize=7.5)
        ax.text(i + w/2, b, f"{b:+.1f}%", ha="center",
                va="bottom" if b >= 0 else "top", fontsize=7.5)
    ax.axhline(47.6, color="darkred", ls="--", lw=1, label="B&H +47.6%")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xticks(x); ax.set_xticklabels(common)
    ax.set_title("Total Return: v1 vs B2", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

    fig.suptitle("Head-to-head: v1 (29 feat) vs B2 (24 feat) — same 7 models, "
                 "same data + Stage 1/2 OOFs, only Stage 3 input differs",
                 fontsize=11, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.name}")


def main():
    print("[1] Load")
    btc, sig, sig_zz, final, bt_summary = load_data()
    rep = PROJECT_ROOT / "reports"

    print("\n[2] Plots")
    plot_summary_table(final, rep / "b2_summary_table.png")
    plot_metric_bars(final, rep / "b2_metric_bars.png")
    plot_equity(btc, sig, sig_zz, rep / "b2_equity_curves.png")
    plot_cm_grid(sig, sig_zz, rep / "b2_cm_grid.png")
    plot_pred_dist(sig, sig_zz, rep / "b2_pred_distribution.png")
    plot_roc_grid(sig, sig_zz, rep / "b2_roc_grid.png")
    plot_v1_vs_b2(rep / "v1_vs_b2_models.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
