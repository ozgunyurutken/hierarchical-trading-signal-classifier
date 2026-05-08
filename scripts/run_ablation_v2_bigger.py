"""
v2 ablation study — 4 configurations × XGBoost (the v1 best model).

Tests whether the hierarchical 3-stage architecture provides measurable
value over simpler designs. All configs share:
  • same data (BTC v4 aligned, stage3_v2 features, v4 OOF labels)
  • same train/test split (chronological 85/15)
  • same walk-forward folds (12 mo min train, 6 mo step)
  • same Optuna 8-trial XGB hyper-parameter search per config
  • same backtest (zero-cost, long-only, hold-on-Hold)

Configurations:
  A1 — Flat (1-stage):     X = tech features
  A2 — 2-stage Trend:      X = tech + s1_oof
  A3 — 2-stage Macro:      X = tech + s2_oof
  A4 — 3-stage Full (v4):  X = tech + s1_oof + s2_oof    (v1 reference)

Outputs:
  data/labels/btc_ablation_v2_bigger_summary.csv
  data/labels/btc_ablation_v2_bigger_test_signals.csv
  reports/ablation_v2_bigger_comparison.png   (4-panel: Sharpe, Return, MaxDD, Win%)
  reports/ablation_v2_bigger_equity.png       (4 equity curves overlaid + B&H)
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.backtester import Backtester
from src.evaluation.metrics import compute_all_metrics
from src.models.classifiers import get_classifier
from src.models.optuna_helpers import tune_classifier_walk_forward
from src.models.stage1_trainer import expanding_window_walk_forward
from src.utils.config import cfg
from src.utils.helpers import chronological_train_test_split, save_csv

config = cfg()
TEST_SIZE = config["training"]["test_size"]
RANDOM_STATE = config["training"]["random_state"]
N_TRIALS = 8
STEP_MONTHS = 6
MIN_TRAIN_MONTHS = 12

CONFIGS = ["A1_flat", "A2_trend", "A3_macro", "A4_full"]


def load_inputs() -> dict:
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned_v2.csv",
                      index_col=0, parse_dates=True)
    tech = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_features_stage3_v2_bigger.csv",
                       index_col=0, parse_dates=True)
    y = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_adaptive_bigger.csv",
                    index_col=0, parse_dates=True).iloc[:, 0]
    s1 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_stage1_oof_lda_bigger.csv",
                     index_col=0, parse_dates=True)
    s2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior_bigger.csv",
                     index_col=0, parse_dates=True)
    common = tech.index.intersection(y.index).intersection(s1.index).intersection(s2.index)
    print(f"  common index: {len(common)} days, "
          f"{common.min().date()} → {common.max().date()}")
    return {
        "btc": btc, "tech": tech.loc[common], "y": y.loc[common],
        "s1": s1.loc[common], "s2": s2.loc[common], "common": common,
    }


def build_X(tech: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, cfg_name: str) -> pd.DataFrame:
    if cfg_name == "A1_flat":
        return tech.copy()
    if cfg_name == "A2_trend":
        return pd.concat([tech, s1.add_prefix("s1_")], axis=1)
    if cfg_name == "A3_macro":
        return pd.concat([tech, s2.add_prefix("s2_")], axis=1)
    if cfg_name == "A4_full":
        return pd.concat([tech, s1.add_prefix("s1_"), s2.add_prefix("s2_")], axis=1)
    raise ValueError(cfg_name)


def run_one_config(cfg_name: str, X: pd.DataFrame, y: pd.Series,
                   close_test: pd.Series, bt: Backtester) -> dict:
    print(f"\n  ---- {cfg_name}  X.shape={X.shape}  ----")
    X_tr, X_te = chronological_train_test_split(X, test_size=TEST_SIZE)
    y_tr, y_te = y.loc[X_tr.index], y.loc[X_te.index]
    print(f"    Train={X_tr.shape}  Test={X_te.shape}")

    folds = expanding_window_walk_forward(
        X_tr, y_tr, min_train_months=MIN_TRAIN_MONTHS, step_months=STEP_MONTHS
    )
    print(f"    Walk-forward folds: {len(folds)}")

    t0 = time.time()
    best_params, _ = tune_classifier_walk_forward(
        X_tr, y_tr, "xgboost", folds=folds, n_trials=N_TRIALS,
        study_name=f"ablation_{cfg_name}",
    )
    print(f"    Optuna best params: {best_params}")
    print(f"    Tune time: {time.time()-t0:.1f}s")

    clf = get_classifier("xgboost", **best_params)
    clf.fit(X_tr, y_tr)
    train_med = X_tr.median()
    Xc = X_te.fillna(train_med)
    pred = clf.predict(Xc)
    proba = clf.predict_proba(Xc)
    classes = list(clf.classes_)
    m = compute_all_metrics(y_te.values, pred, y_proba=proba, classes=classes)
    bt_res = bt.run(pd.Series(pred, index=y_te.index), close_test)

    print(f"    Test acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  MCC={m['mcc']:.4f}")
    print(f"    Backtest: ret={bt_res['total_return']*100:+.1f}%  "
          f"Sharpe={bt_res['sharpe_ratio']:.2f}  "
          f"MaxDD={bt_res['max_drawdown']*100:.1f}%  "
          f"Trades={bt_res['n_trades']}  Win={bt_res['win_rate']*100:.1f}%")

    return {
        "cfg": cfg_name, "n_features": X.shape[1],
        "test_acc": round(m["accuracy"], 4), "test_f1": round(m["f1_macro"], 4),
        "test_mcc": round(m["mcc"], 4), "test_balanced_acc": round(m["balanced_accuracy"], 4),
        "return": round(bt_res["total_return"], 4),
        "sharpe": round(bt_res["sharpe_ratio"], 4),
        "max_dd": round(bt_res["max_drawdown"], 4),
        "n_trades": int(bt_res["n_trades"]),
        "win_rate": round(bt_res["win_rate"], 4),
        "y_te_index": list(y_te.index.astype(str)),
        "pred": pred.tolist(),
        "best_params": best_params,
    }


def make_plots(results: list[dict], close_test: pd.Series, bh_summary: dict) -> None:
    df = pd.DataFrame([{k: r[k] for k in
        ["cfg","n_features","test_acc","test_f1","test_mcc","return","sharpe","max_dd","n_trades","win_rate"]}
        for r in results])
    df_disp = df.set_index("cfg")

    # Panel: 4-bar comparison
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col, title, fmt in [
        (axes[0, 0], "sharpe", "Sharpe Ratio (yüksek iyi)", "{:.2f}"),
        (axes[0, 1], "return", "Toplam Getiri (yüksek iyi)", "{:+.1%}"),
        (axes[1, 0], "max_dd", "Maximum Drawdown (sıfıra yakın iyi)", "{:.1%}"),
        (axes[1, 1], "win_rate", "Win Rate (yüksek iyi)", "{:.1%}"),
    ]:
        vals = df_disp[col]
        bars = ax.bar(df_disp.index, vals,
                      color=["#a3c1da", "#ffcc99", "#ff9999", "#9b8bbf"],
                      edgecolor="#333", lw=0.7)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(""); ax.set_ylabel(col)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v, fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    # mark Buy & Hold line on Sharpe + Return
    axes[0, 0].axhline(bh_summary["sharpe"], color="darkred", ls="--", lw=1, label=f"B&H ({bh_summary['sharpe']:.2f})")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(bh_summary["return"], color="darkred", ls="--", lw=1, label=f"B&H ({bh_summary['return']:+.1%})")
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("Ablation v2-bigger — 4 Mimari Konfigürasyonu (XGBoost, BTC test seti 533 gün)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = PROJECT_ROOT / "reports" / "ablation_v2_bigger_comparison.png"
    fig.savefig(out1, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out1.relative_to(PROJECT_ROOT)}")

    # Panel: equity curves
    bt = Backtester()
    fig2, ax = plt.subplots(figsize=(13, 6))
    bh_eq = bt.run_buy_and_hold(close_test)
    ax.plot(bh_eq.index, bh_eq / bh_eq.iloc[0], color="black", lw=1.4, ls="--", label="Buy & Hold")
    palette = ["#1f77b4", "#ff7f0e", "#d62728", "#7d3c98"]
    for r, col in zip(results, palette):
        idx = pd.to_datetime(r["y_te_index"])
        pred = pd.Series(r["pred"], index=idx)
        eq = bt.run(pred, close_test.loc[idx])["equity_curve"]
        ax.plot(eq.index, eq / eq.iloc[0], color=col, lw=1.5,
                label=f"{r['cfg']}  Sharpe={r['sharpe']:+.2f}  Ret={r['return']*100:+.1f}%")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.set_title("Ablation v2-bigger — Equity Curves (test seti)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Equity (normalized, log scale)")
    ax.legend(loc="upper left", fontsize=9)
    fig2.tight_layout()
    out2 = PROJECT_ROOT / "reports" / "ablation_v2_bigger_equity.png"
    fig2.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close(fig2)
    print(f"  saved: {out2.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("[1] Load inputs")
    data = load_inputs()
    btc, tech, y, s1, s2 = data["btc"], data["tech"], data["y"], data["s1"], data["s2"]
    bt = Backtester()

    print("\n[2] Build chronological train/test slice for the test close prices")
    _, _ = chronological_train_test_split(tech, test_size=TEST_SIZE)
    _, X_te_template = chronological_train_test_split(tech, test_size=TEST_SIZE)
    close_test = btc["Close"].loc[X_te_template.index]
    print(f"    test close: {len(close_test)} days, "
          f"{close_test.index.min().date()} → {close_test.index.max().date()}")

    print("\n[3] Run 4 ablation configs")
    results = []
    for cfg_name in CONFIGS:
        X = build_X(tech, s1, s2, cfg_name)
        r = run_one_config(cfg_name, X, y, close_test, bt)
        results.append(r)

    print("\n[4] Buy & Hold baseline on the test slice")
    bh_eq = bt.run_buy_and_hold(close_test)
    bh_ret = bh_eq.iloc[-1]/bh_eq.iloc[0] - 1
    bh_sharpe = close_test.pct_change().mean() / close_test.pct_change().std() * np.sqrt(252)
    bh_dd = ((bh_eq - bh_eq.cummax())/bh_eq.cummax()).min()
    bh_summary = {"return": float(bh_ret), "sharpe": float(bh_sharpe), "max_dd": float(bh_dd)}
    print(f"    B&H: ret={bh_ret*100:+.1f}%  Sharpe={bh_sharpe:.2f}  MaxDD={bh_dd*100:.1f}%")

    print("\n[5] Write summary CSV")
    summary = pd.DataFrame([{k: r[k] for k in
        ["cfg","n_features","test_acc","test_f1","test_mcc","test_balanced_acc",
         "return","sharpe","max_dd","n_trades","win_rate"]} for r in results])
    summary.loc[len(summary)] = {"cfg": "Buy&Hold", "n_features": "-",
        "test_acc": "-", "test_f1": "-", "test_mcc": "-", "test_balanced_acc": "-",
        "return": round(bh_ret, 4), "sharpe": round(bh_sharpe, 4),
        "max_dd": round(bh_dd, 4), "n_trades": 1, "win_rate": "-"}
    print("\n=== Ablation v2 Summary ===")
    print(summary.to_string(index=False))
    save_csv(summary, PROJECT_ROOT / "data" / "labels" / "btc_ablation_v2_bigger_summary.csv")

    test_signals = pd.DataFrame({"y_true": y.loc[pd.to_datetime(results[0]["y_te_index"])].values},
                                index=pd.to_datetime(results[0]["y_te_index"]))
    for r in results:
        test_signals[r["cfg"]] = r["pred"]
    save_csv(test_signals, PROJECT_ROOT / "data" / "labels" / "btc_ablation_v2_bigger_test_signals.csv")

    print("\n[6] Render plots")
    make_plots(results, close_test, bh_summary)

    print("\nAblation v2 complete.")


if __name__ == "__main__":
    main()
