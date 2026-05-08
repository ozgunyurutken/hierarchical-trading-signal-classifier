"""
Regenerate iter 2 visuals from saved v2 artifacts:
  - 5-model confusion matrix grid (Stage 3 v2)
  - 5-model ROC curves (one-vs-rest, all classes)
  - Decision boundary on PCA 2D for the best 3 models
  - v1 vs v2 head-to-head equity + drawdown
  - Stage 3 prediction-vs-true distribution comparison
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

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from src.evaluation.backtester import Backtester

REPORTS = PROJECT_ROOT / "reports"
LABELS = PROJECT_ROOT / "data" / "labels"
PROC = PROJECT_ROOT / "data" / "processed"

CLASSES = ["Buy", "Hold", "Sell"]
COLOR_BSH = {"Buy": "#2ECC71", "Sell": "#E74C3C", "Hold": "#F1C40F"}


def load_test_signals_v2() -> pd.DataFrame:
    a = pd.read_csv(LABELS / "btc_test_signals_v2.csv", index_col=0, parse_dates=True)
    b = pd.read_csv(LABELS / "btc_test_signals_v2_phase_b.csv", index_col=0, parse_dates=True)
    out = a.join(b.drop(columns=["y_true"], errors="ignore"))
    return out


def plot_cm_grid(test_sig: pd.DataFrame) -> None:
    models = ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    y_true = test_sig["y_true"]
    for ax, m in zip(axes, models):
        if f"{m}_pred" not in test_sig.columns:
            ax.set_visible(False)
            continue
        cm = confusion_matrix(y_true, test_sig[f"{m}_pred"], labels=CLASSES)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
            cbar=ax is axes[-1], square=True, vmin=0, vmax=1,
        )
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, test_sig[f"{m}_pred"])
        f1 = f1_score(y_true, test_sig[f"{m}_pred"], average="macro", zero_division=0)
        ax.set_title(f"{m.upper()}\nacc={acc:.3f}, f1={f1:.3f}", fontsize=10)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True" if ax is axes[0] else "")
    plt.suptitle("Stage 3 v2 — Confusion matrices (test set, 505 days)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_cm_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: iter2_cm_grid.png")


def plot_roc_grid(test_sig: pd.DataFrame) -> None:
    models = ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    lb = LabelBinarizer().fit(CLASSES)
    y_bin = lb.transform(test_sig["y_true"].values)

    for ax, m in zip(axes, models):
        if f"{m}_proba_Buy" not in test_sig.columns:
            ax.set_visible(False)
            continue
        proba = np.stack([test_sig[f"{m}_proba_{c}"].values for c in CLASSES], axis=1)
        for i, c in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            auc_v = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=COLOR_BSH[c], lw=2, label=f"{c} (AUC={auc_v:.2f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR" if ax is axes[0] else "")
        ax.set_title(m.upper())
        ax.legend(loc="lower right", fontsize=8)
    plt.suptitle("Stage 3 v2 — ROC (one-vs-rest)", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_roc_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: iter2_roc_grid.png")


def plot_pred_dist_vs_true(test_sig: pd.DataFrame) -> None:
    models = ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]
    rows = []
    for m in models:
        if f"{m}_pred" not in test_sig.columns:
            continue
        for cls in CLASSES:
            rows.append({
                "model": m.upper(),
                "class": cls,
                "share": (test_sig[f"{m}_pred"] == cls).mean(),
            })
    rows.append({"model": "TRUE", "class": "Buy", "share": (test_sig["y_true"] == "Buy").mean()})
    rows.append({"model": "TRUE", "class": "Hold", "share": (test_sig["y_true"] == "Hold").mean()})
    rows.append({"model": "TRUE", "class": "Sell", "share": (test_sig["y_true"] == "Sell").mean()})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = df.pivot(index="model", columns="class", values="share")[CLASSES]
    pivot = pivot.reindex(["TRUE", "LDA", "MLP", "XGBOOST", "LIGHTGBM", "RANDOM_FOREST"])
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[COLOR_BSH[c] for c in CLASSES], edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test set share (n=505)")
    ax.set_title("Stage 3 v2 — Predicted vs true class distribution")
    ax.legend(loc="upper right", bbox_to_anchor=(1.13, 1.0))
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_pred_distribution.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: iter2_pred_distribution.png")


def plot_pca_decision(test_sig: pd.DataFrame) -> None:
    """PCA 2D scatter of test set, colored by best 3 models' predicted class."""
    models = ["mlp", "xgboost", "lightgbm"]

    # Reconstruct test combined feature matrix (osc + s1 + s2)
    osc = pd.read_csv(PROC / "btc_features_stage3_v2.csv", index_col=0, parse_dates=True)
    s1 = pd.read_csv(LABELS / "btc_stage1_oof_lda.csv", index_col=0, parse_dates=True)
    s2 = pd.read_csv(LABELS / "btc_oof_regime_posterior.csv", index_col=0, parse_dates=True)

    common = test_sig.index.intersection(osc.index).intersection(s1.index).intersection(s2.index)
    X_test = pd.concat([osc.loc[common], s1.loc[common], s2.loc[common]], axis=1)
    X_test = X_test.fillna(X_test.median())

    scaler = StandardScaler().fit(X_test)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(scaler.transform(X_test))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    # First panel: TRUE
    for cls in CLASSES:
        mask = (test_sig.loc[common, "y_true"] == cls).values
        axes[0].scatter(Z[mask, 0], Z[mask, 1], c=COLOR_BSH[cls], label=cls, alpha=0.6, s=14)
    axes[0].set_title("TRUE labels"); axes[0].legend(fontsize=9)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    for ax, m in zip(axes[1:], models):
        if f"{m}_pred" not in test_sig.columns:
            ax.set_visible(False); continue
        for cls in CLASSES:
            mask = (test_sig.loc[common, f"{m}_pred"] == cls).values
            ax.scatter(Z[mask, 0], Z[mask, 1], c=COLOR_BSH[cls], label=cls, alpha=0.6, s=14)
        ax.set_title(f"{m.upper()} predictions")
        ax.set_xlabel("PC1")
        ax.legend(fontsize=9)

    plt.suptitle(f"Stage 3 v2 — PCA 2D (var explained: {pca.explained_variance_ratio_.sum():.1%})",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_decision_boundary_pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: iter2_decision_boundary_pca.png")


def plot_full_equity_comparison(test_sig: pd.DataFrame) -> None:
    btc = pd.read_csv(PROC / "btc_aligned.csv", index_col=0, parse_dates=True)
    test_close = btc["Close"].loc[test_sig.index]

    bt = Backtester()
    bh = bt.run_buy_and_hold(test_close)

    strategies = {}
    for m in ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]:
        if f"{m}_pred" not in test_sig.columns:
            continue
        strategies[m.upper()] = bt.run(test_sig[f"{m}_pred"], test_close)

    # v1 baselines
    v1 = pd.read_csv(LABELS / "btc_test_signals.csv", index_col=0, parse_dates=True)
    strategies["v1 LDA"] = bt.run(v1["lda_pred"], btc["Close"].loc[v1.index])
    strategies["v1 MLP"] = bt.run(v1["mlp_pred"], btc["Close"].loc[v1.index])

    # 2-panel: equity + drawdown
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    palette = {
        "v1 LDA": "#7F8C8D", "v1 MLP": "#BDC3C7",
        "LDA": "#3498DB", "MLP": "#9B59B6",
        "XGBOOST": "#E67E22", "LIGHTGBM": "#27AE60", "RANDOM_FOREST": "#16A085",
    }
    for label, result in strategies.items():
        eq = result["equity_curve"]
        axes[0].plot(eq.index, eq.values,
                     label=f"{label} ret={result['total_return']:+.1%} sh={result['sharpe_ratio']:.2f}",
                     color=palette.get(label, "#34495E"), linewidth=1.3, alpha=0.85)
        cummax = eq.cummax()
        dd = (eq - cummax) / cummax * 100
        axes[1].plot(dd.index, dd.values, color=palette.get(label, "#34495E"), linewidth=0.9, alpha=0.7)

    axes[0].plot(bh.index, bh.values, label=f"Buy&Hold ret={bh.iloc[-1]/bh.iloc[0]-1:+.1%}",
                 color="#2C3E50", linewidth=1.5, linestyle="--")
    bh_dd = (bh - bh.cummax()) / bh.cummax() * 100
    axes[1].plot(bh_dd.index, bh_dd.values, color="#2C3E50", linewidth=1.2, linestyle="--", alpha=0.7)

    axes[0].set_title("v1 vs v2 strategies — equity curves on test period (BTC, 2024-04 → 2025-12)")
    axes[0].set_ylabel("Portfolio value ($)")
    axes[0].legend(loc="upper left", fontsize=9, ncol=2)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_equity_full_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: iter2_equity_full_comparison.png")


def plot_metric_radar() -> None:
    """Compact bar chart: WF F1 / Test acc / Test F1 / MCC / Sharpe per model."""
    summary = pd.read_csv(LABELS / "btc_stage3_v2_full_summary.csv", index_col=0)
    bt = pd.read_csv(LABELS / "btc_backtest_v2_phase_b_summary.csv")

    sharpe = {row["strategy"].replace("v2 ", ""): row["sharpe"] for _, row in bt.iterrows()
              if row["strategy"].startswith("v2 ")}
    rets = {row["strategy"].replace("v2 ", ""): row["total_return"] for _, row in bt.iterrows()
            if row["strategy"].startswith("v2 ")}

    # Add MLP from phase A backtest
    bt_a = pd.read_csv(LABELS / "btc_backtest_v2_summary.csv")
    for _, r in bt_a.iterrows():
        s = r["strategy"]
        if s in ("v2 LDA", "v2 MLP"):
            sharpe[s.replace("v2 ", "")] = r["sharpe"]
            rets[s.replace("v2 ", "")] = r["total_return"]

    rows = []
    for m in summary.index:
        rows.append({
            "Model": m,
            "Test F1 (macro)": summary.loc[m, "test_f1"],
            "Test MCC": summary.loc[m, "test_mcc"],
            "Backtest Sharpe": sharpe.get(m, 0),
            "Backtest Return": rets.get(m, 0),
        })
    df = pd.DataFrame(rows).set_index("Model")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    cols = df.columns
    for ax, c in zip(axes, cols):
        df[c].sort_values().plot(kind="barh", ax=ax, color="#3498DB", edgecolor="white")
        ax.set_title(c)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.suptitle("Stage 3 v2 — model comparison (classification + economic)", fontsize=12, y=1.03)
    plt.tight_layout()
    plt.savefig(REPORTS / "iter2_metric_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: iter2_metric_comparison.png")


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    test_sig = load_test_signals_v2()
    print(f"Loaded test signals: {test_sig.shape}, columns: {list(test_sig.columns)[:6]}...")

    print("\n[1] CM grid (5 models)")
    plot_cm_grid(test_sig)

    print("\n[2] ROC grid (5 models)")
    plot_roc_grid(test_sig)

    print("\n[3] Predicted vs true distribution")
    plot_pred_dist_vs_true(test_sig)

    print("\n[4] PCA 2D decision boundary (top 3 models)")
    plot_pca_decision(test_sig)

    print("\n[5] Full equity + drawdown comparison")
    plot_full_equity_comparison(test_sig)

    print("\n[6] Metric comparison (classification + economic)")
    plot_metric_radar()


if __name__ == "__main__":
    main()
