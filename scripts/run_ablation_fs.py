"""
v2/feature-selection ablation runner. CLI: --subset {b1,b2,b3}

Tests whether redundancy in the 29 Stage 3 v2 technical features is the
reason A4 (full hierarchical) doesn't beat A1 (flat) in the v2/ablation
result. We ran 4-config ablation per subset:
  A1 Flat:        X = tech_subset
  A2 2-stage T:   X = tech_subset + s1
  A3 2-stage M:   X = tech_subset + s2
  A4 3-stage F:   X = tech_subset + s1 + s2

If a smaller, less-trend-redundant tech subset makes A4 > A1, the
hierarchical-architecture hypothesis is rescued.

Subsets:
  b1  Aggressive (no trend, no momentum, no long log_ret) — 15 feat
       oscillator (7: RSI, Stoch K/D, MACD line/signal/hist, Williams)
       + volatility (5: BB upper/lower/bw/pctb, ATR)
       + CCI, ROC, hist_volatility (3) — short-horizon noise indicators
  b2  Moderate — drop only 5 long-term trend cols — 24 feat
       drops: log_ret_50d, log_ret_100d, above_sma_200,
              adx_value, sharpe_proxy_20d
  b3  MI top-15 — data-driven (mutual_info_classif vs y_signal)

Outputs:
  data/labels/btc_ablation_fs_<subset>_summary.csv
  reports/ablation_fs_<subset>_comparison.png
  reports/ablation_fs_<subset>_equity.png
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
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
from sklearn.feature_selection import mutual_info_classif

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

# B1 — aggressive: keep only short-horizon oscillators + volatility
B1_KEEP = [
    "RSI_14", "Stoch_K_14", "Stoch_D_3",
    "MACD_line", "MACD_signal", "MACD_histogram",
    "Williams_R_14",
    "BB_upper", "BB_lower", "BB_bandwidth", "BB_pct_b",
    "ATR_14",
    "CCI_20", "ROC_10", "hist_volatility_20",
]

# B2 — moderate: drop only the 5 most-trend-related cols
B2_DROP = [
    "log_ret_50d", "log_ret_100d",
    "above_sma_200",
    "adx_value",
    "sharpe_proxy_20d",
]


def select_subset(tech: pd.DataFrame, y: pd.Series, subset: str) -> pd.DataFrame:
    if subset == "b1":
        keep = [c for c in B1_KEEP if c in tech.columns]
        missing = [c for c in B1_KEEP if c not in tech.columns]
        if missing:
            print(f"  WARN B1: missing {missing}")
        return tech[keep].copy()

    if subset == "b2":
        keep = [c for c in tech.columns if c not in B2_DROP]
        return tech[keep].copy()

    if subset == "b3":
        common = tech.dropna().index.intersection(y.dropna().index)
        Xc = tech.loc[common].fillna(tech.median())
        yc = y.loc[common]
        # Only use train slice for MI ranking (no test leakage)
        X_tr, _ = chronological_train_test_split(Xc, test_size=TEST_SIZE)
        y_tr = yc.loc[X_tr.index]
        mi = mutual_info_classif(X_tr, y_tr, random_state=RANDOM_STATE,
                                 discrete_features=False)
        mi_ser = pd.Series(mi, index=Xc.columns).sort_values(ascending=False)
        print(f"  B3 MI ranking (top 20):")
        for c, v in mi_ser.head(20).items():
            print(f"    {c:24s}  MI={v:.4f}")
        keep = mi_ser.head(15).index.tolist()
        return tech[keep].copy()

    raise ValueError(f"unknown subset {subset}")


def build_X(tech_sub: pd.DataFrame, s1: pd.DataFrame, s2: pd.DataFrame, cfg_name: str):
    if cfg_name == "A1_flat":
        return tech_sub.copy()
    if cfg_name == "A2_trend":
        return pd.concat([tech_sub, s1.add_prefix("s1_")], axis=1)
    if cfg_name == "A3_macro":
        return pd.concat([tech_sub, s2.add_prefix("s2_")], axis=1)
    if cfg_name == "A4_full":
        return pd.concat([tech_sub, s1.add_prefix("s1_"), s2.add_prefix("s2_")], axis=1)
    raise ValueError(cfg_name)


def run_one(cfg_name, X, y, close_test, bt):
    print(f"\n  ---- {cfg_name}  X.shape={X.shape}  ----")
    X_tr, X_te = chronological_train_test_split(X, test_size=TEST_SIZE)
    y_tr, y_te = y.loc[X_tr.index], y.loc[X_te.index]
    folds = expanding_window_walk_forward(
        X_tr, y_tr, min_train_months=MIN_TRAIN_MONTHS, step_months=STEP_MONTHS
    )

    t0 = time.time()
    best_params, _ = tune_classifier_walk_forward(
        X_tr, y_tr, "xgboost", folds=folds, n_trials=N_TRIALS,
        study_name=f"fs_{cfg_name}",
    )
    clf = get_classifier("xgboost", **best_params)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te.fillna(X_tr.median()))
    proba = clf.predict_proba(X_te.fillna(X_tr.median()))
    m = compute_all_metrics(y_te.values, pred, y_proba=proba, classes=list(clf.classes_))
    bt_res = bt.run(pd.Series(pred, index=y_te.index), close_test)

    print(f"    {time.time()-t0:.1f}s  acc={m['accuracy']:.4f}  F1={m['f1_macro']:.4f}  "
          f"MCC={m['mcc']:.4f}  ret={bt_res['total_return']*100:+.1f}%  "
          f"Sharpe={bt_res['sharpe_ratio']:.2f}  Win={bt_res['win_rate']*100:.1f}%")
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
    }


def make_plot(results, close_test, bh_summary, subset):
    df = pd.DataFrame([{k: r[k] for k in
        ["cfg","n_features","test_acc","test_f1","test_mcc","return","sharpe","max_dd","win_rate"]}
        for r in results]).set_index("cfg")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, col, title, fmt in [
        (axes[0, 0], "sharpe", "Sharpe Ratio", "{:.2f}"),
        (axes[0, 1], "return", "Toplam Getiri", "{:+.1%}"),
        (axes[1, 0], "max_dd", "Maximum Drawdown", "{:.1%}"),
        (axes[1, 1], "win_rate", "Win Rate", "{:.1%}"),
    ]:
        vals = df[col]
        bars = ax.bar(df.index, vals,
                      color=["#a3c1da", "#ffcc99", "#ff9999", "#9b8bbf"],
                      edgecolor="#333", lw=0.7)
        ax.axhline(0, color="black", lw=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v, fmt.format(v),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=0.3)
    axes[0, 0].axhline(bh_summary["sharpe"], color="darkred", ls="--", lw=1,
                       label=f"B&H ({bh_summary['sharpe']:.2f})")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(bh_summary["return"], color="darkred", ls="--", lw=1,
                       label=f"B&H ({bh_summary['return']:+.1%})")
    axes[0, 1].legend(fontsize=8)
    n_feat = results[0]["n_features"]  # A1
    fig.suptitle(
        f"Ablation FS-{subset.upper()} — XGBoost, BTC v1 test seti 462 gün  "
        f"(tech subset = {n_feat} feat)",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = PROJECT_ROOT / "reports" / f"ablation_fs_{subset}_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["b1", "b2", "b3"], required=True)
    args = ap.parse_args()

    print(f"=== Ablation FS-{args.subset.upper()} ===")
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    tech = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_features_stage3_v2.csv",
                       index_col=0, parse_dates=True)
    y = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_adaptive.csv",
                    index_col=0, parse_dates=True).iloc[:, 0]
    s1 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_stage1_oof_lda.csv",
                     index_col=0, parse_dates=True)
    s2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv",
                     index_col=0, parse_dates=True)
    common = tech.index.intersection(y.index).intersection(s1.index).intersection(s2.index)
    tech = tech.loc[common]; y = y.loc[common]
    s1 = s1.loc[common]; s2 = s2.loc[common]

    print(f"\n[1] tech full: {tech.shape}, common index: {len(common)} days")
    tech_sub = select_subset(tech, y, args.subset)
    print(f"\n[2] tech subset {args.subset}: {tech_sub.shape}")
    print(f"    cols: {list(tech_sub.columns)}")

    _, X_te_tmp = chronological_train_test_split(tech, test_size=TEST_SIZE)
    close_test = btc["Close"].loc[X_te_tmp.index]
    print(f"\n[3] test slice: {len(close_test)} days, "
          f"{close_test.index.min().date()} → {close_test.index.max().date()}")

    bt = Backtester()
    results = []
    print("\n[4] Run 4 ablation configs")
    for cfg_name in CONFIGS:
        X = build_X(tech_sub, s1, s2, cfg_name)
        results.append(run_one(cfg_name, X, y, close_test, bt))

    bh = bt.run_buy_and_hold(close_test)
    bh_ret = bh.iloc[-1]/bh.iloc[0] - 1
    bh_sharpe = close_test.pct_change().mean() / close_test.pct_change().std() * np.sqrt(252)
    bh_dd = ((bh-bh.cummax())/bh.cummax()).min()
    bh_summary = {"return": float(bh_ret), "sharpe": float(bh_sharpe), "max_dd": float(bh_dd)}
    print(f"\n  B&H: ret={bh_ret*100:+.1f}%  Sharpe={bh_sharpe:.2f}  MaxDD={bh_dd*100:.1f}%")

    summary = pd.DataFrame([{k: r[k] for k in
        ["cfg","n_features","test_acc","test_f1","test_mcc","test_balanced_acc",
         "return","sharpe","max_dd","n_trades","win_rate"]} for r in results])
    summary.loc[len(summary)] = {
        "cfg": "Buy&Hold", "n_features": "-",
        "test_acc": "-", "test_f1": "-", "test_mcc": "-", "test_balanced_acc": "-",
        "return": round(bh_ret, 4), "sharpe": round(bh_sharpe, 4),
        "max_dd": round(bh_dd, 4), "n_trades": 1, "win_rate": "-",
    }
    print(f"\n=== Ablation FS-{args.subset.upper()} Summary ===")
    print(summary.to_string(index=False))
    save_csv(summary, PROJECT_ROOT / "data" / "labels" /
             f"btc_ablation_fs_{args.subset}_summary.csv")
    make_plot(results, close_test, bh_summary, args.subset)


if __name__ == "__main__":
    main()
