"""
Final iter 2 visual refresh — Phase A + B + C combined.

Outputs (all under reports/):
  iter2_cm_grid_full.png            — 7-model confusion matrix grid
  iter2_roc_grid_full.png           — 7-model ROC (one-vs-rest)
  iter2_pred_dist_full.png          — predicted vs true class shares (7 models + true)
  iter2_equity_full_v2.png          — equity + drawdown panel, all v2 strategies vs B&H
  iter2_metric_comparison_full.png  — WF F1 / Test F1 / MCC / Sharpe / Return per model
  iter2_zigzag_vs_sma.png           — SMA vs ZigZag trend labels on BTC log-price
  iter2_summary_table.png           — final report-ready summary table (rendered)

Old MVP v1 plots have already been archived to old_results/.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, roc_curve,
)
from sklearn.preprocessing import LabelBinarizer

from src.evaluation.backtester import Backtester

REPORTS = PROJECT_ROOT / "reports"
LABELS = PROJECT_ROOT / "data" / "labels"
PROC = PROJECT_ROOT / "data" / "processed"

CLASSES = ["Buy", "Hold", "Sell"]
COLOR_BSH = {"Buy": "#2ECC71", "Sell": "#E74C3C", "Hold": "#F1C40F"}


def load_all_test_signals() -> pd.DataFrame:
    """Concatenate Phase A (LDA, MLP) + Phase B (XGB, LGBM, RF) + Phase C (ZZ-XGB, ZZ-MLP)."""
    a = pd.read_csv(LABELS / "btc_test_signals_v2.csv", index_col=0, parse_dates=True)
    b = pd.read_csv(LABELS / "btc_test_signals_v2_phase_b.csv", index_col=0, parse_dates=True)
    c = pd.read_csv(LABELS / "btc_test_signals_v2_zigzag.csv", index_col=0, parse_dates=True)

    out = a.copy()
    for df, suffix in [(b, ""), (c, "_zz")]:
        for col in df.columns:
            if col == "y_true":
                continue
            new = col if not suffix else col.replace("_pred", f"{suffix}_pred").replace("_proba_", f"{suffix}_proba_")
            # ZigZag: rename xgboost->xgboost_zz, mlp->mlp_zz
            if suffix:
                new = "zz_" + col if "_pred" in col or "_proba_" in col else col
            out[new] = df[col].reindex(out.index)
    return out


def model_columns(test_sig: pd.DataFrame) -> list[str]:
    """Return ordered list of model keys present in the dataframe."""
    candidates = ["lda", "mlp", "xgboost", "lightgbm", "random_forest", "zz_xgboost", "zz_mlp"]
    return [m for m in candidates if f"{m}_pred" in test_sig.columns]


def model_label(m: str) -> str:
    return {"lda": "LDA", "mlp": "MLP", "xgboost": "XGB", "lightgbm": "LGBM",
            "random_forest": "RF", "zz_xgboost": "ZZ-XGB", "zz_mlp": "ZZ-MLP"}[m]


def plot_cm_grid(test_sig: pd.DataFrame) -> None:
    models = model_columns(test_sig)
    cols = len(models)
    fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 4.5))
    if cols == 1:
        axes = [axes]
    y_true = test_sig["y_true"]
    for ax, m in zip(axes, models):
        cm = confusion_matrix(y_true, test_sig[f"{m}_pred"], labels=CLASSES)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
            cbar=ax is axes[-1], square=True, vmin=0, vmax=1,
        )
        acc = accuracy_score(y_true, test_sig[f"{m}_pred"])
        f1 = f1_score(y_true, test_sig[f"{m}_pred"], average="macro", zero_division=0)
        ax.set_title(f"{model_label(m)}\nacc={acc:.3f}, f1={f1:.3f}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True" if ax is axes[0] else "")
    plt.suptitle("Stage 3 v2 — Confusion matrices (BTC test, 505 days)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_cm_grid_full.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved: iter2_cm_grid_full.png")


def plot_roc_grid(test_sig: pd.DataFrame) -> None:
    models = model_columns(test_sig)
    cols = len(models)
    fig, axes = plt.subplots(1, cols, figsize=(3.5 * cols, 4.5))
    if cols == 1:
        axes = [axes]
    lb = LabelBinarizer().fit(CLASSES)
    y_bin = lb.transform(test_sig["y_true"].values)

    for ax, m in zip(axes, models):
        cols_proba = [f"{m}_proba_{c}" for c in CLASSES]
        if not all(c in test_sig.columns for c in cols_proba):
            ax.set_visible(False); continue
        proba = np.stack([test_sig[c].values for c in cols_proba], axis=1)
        for i, c in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            auc_v = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLOR_BSH[c], lw=2, label=f"{c} ({auc_v:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR" if ax is axes[0] else "")
        ax.set_title(model_label(m))
        ax.legend(loc="lower right", fontsize=7)
    plt.suptitle("Stage 3 v2 — ROC (one-vs-rest)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_roc_grid_full.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved: iter2_roc_grid_full.png")


def plot_pred_dist(test_sig: pd.DataFrame) -> None:
    models = model_columns(test_sig)
    rows = []
    for m in models:
        for cls in CLASSES:
            rows.append({"model": model_label(m), "class": cls,
                         "share": (test_sig[f"{m}_pred"] == cls).mean()})
    for cls in CLASSES:
        rows.append({"model": "TRUE", "class": cls, "share": (test_sig["y_true"] == cls).mean()})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = df.pivot(index="model", columns="class", values="share")[CLASSES]
    order = ["TRUE"] + [model_label(m) for m in models]
    pivot = pivot.reindex(order)
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[COLOR_BSH[c] for c in CLASSES], edgecolor="white")
    ax.set_xlim(0, 1); ax.set_xlabel("Test set share")
    ax.set_title("Stage 3 v2 — predicted vs true class distribution (test 505 days)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.13, 1.0))
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_pred_dist_full.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved: iter2_pred_dist_full.png")


def plot_equity_full(test_sig: pd.DataFrame) -> None:
    btc = pd.read_csv(PROC / "btc_aligned.csv", index_col=0, parse_dates=True)
    test_close = btc["Close"].loc[test_sig.index]
    bt = Backtester()
    bh = bt.run_buy_and_hold(test_close)

    strategies: dict[str, dict] = {}
    for m in model_columns(test_sig):
        strategies[model_label(m)] = bt.run(test_sig[f"{m}_pred"], test_close)

    palette = {
        "LDA": "#3498DB", "MLP": "#9B59B6",
        "XGB": "#E67E22", "LGBM": "#27AE60", "RF": "#16A085",
        "ZZ-XGB": "#D35400", "ZZ-MLP": "#8E44AD",
    }
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    for label, result in strategies.items():
        eq = result["equity_curve"]
        axes[0].plot(eq.index, eq.values,
                     label=f"{label}: ret={result['total_return']:+.1%} sh={result['sharpe_ratio']:.2f}",
                     color=palette.get(label, "#34495E"), linewidth=1.4, alpha=0.9)
        cm = eq.cummax()
        dd = (eq - cm) / cm * 100
        axes[1].plot(dd.index, dd.values, color=palette.get(label, "#34495E"),
                     linewidth=0.8, alpha=0.6)

    axes[0].plot(bh.index, bh.values,
                 label=f"Buy&Hold: ret={bh.iloc[-1] / bh.iloc[0] - 1:+.1%}",
                 color="#2C3E50", linewidth=2, linestyle="--")
    bh_dd = (bh - bh.cummax()) / bh.cummax() * 100
    axes[1].plot(bh_dd.index, bh_dd.values, color="#2C3E50", linewidth=1.5,
                 linestyle="--", alpha=0.7, label="Buy&Hold")

    axes[0].set_title("v2 strategies — full comparison (BTC test 2024-04 → 2025-12)",
                      fontsize=13)
    axes[0].set_ylabel("Portfolio value ($)")
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Drawdown (%)")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_equity_full_v2.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved: iter2_equity_full_v2.png")


def plot_metric_comparison(test_sig: pd.DataFrame) -> None:
    btc = pd.read_csv(PROC / "btc_aligned.csv", index_col=0, parse_dates=True)
    test_close = btc["Close"].loc[test_sig.index]
    bt = Backtester()
    bh = bt.run_buy_and_hold(test_close)

    rows = []
    for m in model_columns(test_sig):
        y_pred = test_sig[f"{m}_pred"]
        acc = accuracy_score(test_sig["y_true"], y_pred)
        f1 = f1_score(test_sig["y_true"], y_pred, average="macro", zero_division=0)
        from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
        bal = balanced_accuracy_score(test_sig["y_true"], y_pred)
        mcc = matthews_corrcoef(test_sig["y_true"], y_pred)
        result = bt.run(y_pred, test_close)
        rows.append({
            "Model": model_label(m),
            "Test F1 (macro)": f1,
            "Test MCC": mcc,
            "Backtest Sharpe": result["sharpe_ratio"],
            "Backtest Return": result["total_return"],
        })

    df = pd.DataFrame(rows).set_index("Model")
    bh_sharpe = float(test_close.pct_change().mean() / test_close.pct_change().std() * np.sqrt(252))
    bh_ret = float(bh.iloc[-1] / bh.iloc[0] - 1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, c in zip(axes, df.columns):
        bars = df[c].sort_values()
        bars.plot(kind="barh", ax=ax, color="#3498DB", edgecolor="white")
        ax.set_title(c, fontsize=11)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
        if c == "Backtest Sharpe":
            ax.axvline(bh_sharpe, color="#E74C3C", linestyle="--", linewidth=1.5,
                       label=f"B&H {bh_sharpe:.2f}")
            ax.legend(fontsize=8)
        elif c == "Backtest Return":
            ax.axvline(bh_ret, color="#E74C3C", linestyle="--", linewidth=1.5,
                       label=f"B&H {bh_ret:+.1%}")
            ax.legend(fontsize=8)

    plt.suptitle("Stage 3 v2 — classification metrics + economic metrics per model",
                 fontsize=12, y=1.03)
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_metric_comparison_full.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved: iter2_metric_comparison_full.png")


def plot_zigzag_vs_sma() -> None:
    btc = pd.read_csv(PROC / "btc_aligned.csv", index_col=0, parse_dates=True)
    sma = pd.read_csv(LABELS / "btc_trend_labels.csv", index_col=0, parse_dates=True).iloc[:, 0]
    zz = pd.read_csv(LABELS / "btc_trend_labels_zigzag.csv", index_col=0, parse_dates=True).iloc[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    color_map = {"Uptrend": "#2ECC71", "Downtrend": "#E74C3C", "Sideways": "#95A5A6"}
    for ax, (name, lbl) in zip(axes, [("SMA crossover (MVP)", sma), ("Causal ZigZag (iter 2)", zz)]):
        ax.plot(btc.index, btc["Close"], color="black", linewidth=0.5, alpha=0.4)
        for cls, color in color_map.items():
            mask = lbl == cls
            ax.scatter(lbl.index[mask], btc["Close"].loc[lbl.index[mask]],
                       c=color, s=2, alpha=0.7, label=cls)
        dist = (lbl.value_counts(normalize=True) * 100).round(1).to_dict()
        ax.set_title(f"{name}: {dist}", fontsize=11)
        ax.set_yscale("log")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_zigzag_vs_sma.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  saved: iter2_zigzag_vs_sma.png")


def plot_summary_table(test_sig: pd.DataFrame) -> None:
    btc = pd.read_csv(PROC / "btc_aligned.csv", index_col=0, parse_dates=True)
    test_close = btc["Close"].loc[test_sig.index]
    bt = Backtester()
    bh = bt.run_buy_and_hold(test_close)

    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef
    rows = []
    for m in model_columns(test_sig):
        y_pred = test_sig[f"{m}_pred"]
        result = bt.run(y_pred, test_close)
        rows.append({
            "Model": model_label(m),
            "Test Acc": f"{accuracy_score(test_sig['y_true'], y_pred):.3f}",
            "Test F1": f"{f1_score(test_sig['y_true'], y_pred, average='macro', zero_division=0):.3f}",
            "MCC": f"{matthews_corrcoef(test_sig['y_true'], y_pred):.3f}",
            "Return": f"{result['total_return']:+.1%}",
            "Sharpe": f"{result['sharpe_ratio']:.2f}",
            "MaxDD": f"{result['max_drawdown']:.1%}",
            "Trades": f"{result['n_trades']}",
            "Win%": f"{result['win_rate']:.1%}" if result['n_trades'] else "-",
        })
    rows.append({
        "Model": "Buy & Hold",
        "Test Acc": "-", "Test F1": "-", "MCC": "-",
        "Return": f"{bh.iloc[-1] / bh.iloc[0] - 1:+.1%}",
        "Sharpe": f"{test_close.pct_change().mean() / test_close.pct_change().std() * np.sqrt(252):.2f}",
        "MaxDD": f"{((bh - bh.cummax()) / bh.cummax()).min():.1%}",
        "Trades": "1", "Win%": "-",
    })
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 0.45 * len(df) + 1.5))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10)
    table.scale(1, 1.5)

    # Highlight best Sharpe row(s)
    sharpe_vals = [float(r["Sharpe"]) for r in rows]
    best_i = int(np.argmax(sharpe_vals))
    for j in range(len(df.columns)):
        cell = table[(best_i + 1, j)]  # +1 for header row
        cell.set_facecolor("#FFF3CD")
        cell.set_text_props(weight="bold")

    # Header bold
    for j in range(len(df.columns)):
        table[(0, j)].set_text_props(weight="bold")
        table[(0, j)].set_facecolor("#34495E")
        table[(0, j)].set_text_props(color="white", weight="bold")

    plt.title("Stage 3 v2 — final summary (BTC test 2024-04 → 2025-12, 505 days)",
              fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_summary_table.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  saved: iter2_summary_table.png")

    # Also save as CSV for the report
    df.to_csv(LABELS / "final_iter2_summary_table.csv", index=False)


def main() -> None:
    test_sig = load_all_test_signals()
    print(f"Loaded combined test signals: {test_sig.shape}, columns: {list(test_sig.columns)[:8]}...")

    print("\n[1] CM grid")
    plot_cm_grid(test_sig)
    print("\n[2] ROC grid")
    plot_roc_grid(test_sig)
    print("\n[3] Predicted distribution")
    plot_pred_dist(test_sig)
    print("\n[4] Equity + drawdown panel")
    plot_equity_full(test_sig)
    print("\n[5] Metric comparison (classification + economic)")
    plot_metric_comparison(test_sig)
    print("\n[6] ZigZag vs SMA labels")
    plot_zigzag_vs_sma()
    print("\n[7] Final summary table")
    plot_summary_table(test_sig)


if __name__ == "__main__":
    main()
