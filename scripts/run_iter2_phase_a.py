"""
Iter 2 Phase A — Stage 3 fix:
  1) trend-following features + adaptive threshold label
  2) 5-model retrain (LDA, MLP, XGB, LGBM, RF)
  3) v1 vs v2 backtest comparison

Runs as a standalone script (avoids the jupyter kernel-death issue with long Optuna runs).
Writes the same artifacts a notebook would.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.backtester import Backtester
from src.evaluation.metrics import compute_all_metrics
from src.features.technical_indicators import (
    compute_oscillator_indicators,
    compute_trend_following_features,
    compute_volatility_indicators,
    compute_volume_indicators,
)
from src.models.stage3_trainer import tune_stage3
from src.utils.config import cfg
from src.utils.helpers import chronological_train_test_split, save_csv

config = cfg()
TEST_SIZE = config["training"]["test_size"]
STEP_MONTHS = 6
MIN_TRAIN_MONTHS = 12
N_TRIALS = {"lda": 8, "mlp": 6}
# XGB/LGBM/RF dropped from Phase A: macOS Python 3.11 + xgboost/lightgbm + multithreading
# triggers a segfault inside the C extension during walk-forward. Will revisit in Phase B
# with OMP_NUM_THREADS=1 + n_jobs=1 wrappers.


def main() -> None:
    btc = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
        index_col=0, parse_dates=True,
    )
    ohlcv = btc[["Open", "High", "Low", "Close", "Volume"]]

    print("\n[1] Build Stage 3 v2 features (osc + vol + volume + trend_following)")
    osc = compute_oscillator_indicators(ohlcv)
    volat = compute_volatility_indicators(ohlcv)
    volu = compute_volume_indicators(ohlcv)
    trend_follow = compute_trend_following_features(ohlcv)
    stage3_v2 = pd.concat([osc, volat, volu, trend_follow], axis=1).dropna()
    print(f"  shape={stage3_v2.shape}, trend_follow cols={trend_follow.shape[1]}")
    save_csv(stage3_v2, PROJECT_ROOT / "data" / "processed" / "btc_features_stage3_v2.csv")

    print("\n[2] Adaptive (volatility-adjusted) signal labels primary")
    y_v1 = pd.read_csv(
        PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_fixed.csv",
        index_col=0, parse_dates=True,
    ).iloc[:, 0]
    y_v2 = pd.read_csv(
        PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_adaptive.csv",
        index_col=0, parse_dates=True,
    ).iloc[:, 0]
    print(f"  v1 fixed +/-1%        : {(y_v1.value_counts(normalize=True) * 100).round(1).to_dict()}")
    print(f"  v2 adaptive 0.5xstd   : {(y_v2.value_counts(normalize=True) * 100).round(1).to_dict()}")

    print("\n[3] Load Stage 1 OOF (LDA) + Stage 2 GMM posterior")
    s1_oof = pd.read_csv(
        PROJECT_ROOT / "data" / "labels" / "btc_stage1_oof_lda.csv",
        index_col=0, parse_dates=True,
    )
    s2_oof = pd.read_csv(
        PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv",
        index_col=0, parse_dates=True,
    )

    common = (
        stage3_v2.index
        .intersection(y_v2.index)
        .intersection(s1_oof.index)
        .intersection(s2_oof.index)
    )
    X_osc = stage3_v2.loc[common]
    y = y_v2.loc[common]
    s1 = s1_oof.loc[common]
    s2 = s2_oof.loc[common]

    X_train, X_test = chronological_train_test_split(X_osc, test_size=TEST_SIZE)
    y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]
    s1_train, s1_test = s1.loc[X_train.index], s1.loc[X_test.index]
    s2_train, s2_test = s2.loc[X_train.index], s2.loc[X_test.index]
    print(f"  Train={X_train.shape}, Test={X_test.shape}")
    print(f"  Train class dist: {(y_train.value_counts(normalize=True) * 100).round(1).to_dict()}")
    print(f"  Test  class dist: {(y_test.value_counts(normalize=True) * 100).round(1).to_dict()}")

    print("\n[4] Tune 2 classifiers via Optuna walk-forward (Phase A scope)")
    results = {}
    for clf_name in ["lda", "mlp"]:
        print(f"\n  >>> {clf_name.upper()}")
        t0 = time.time()
        res = tune_stage3(
            X_train, y_train, s1_train, s2_train,
            classifier_name=clf_name,
            n_trials=N_TRIALS[clf_name],
            save_model=False,
            step_months=STEP_MONTHS,
            min_train_months=MIN_TRAIN_MONTHS,
        )
        elapsed = time.time() - t0
        print(f"      best   : {res['best_params']}")
        print(f"      WF f1  : {res['avg_f1_macro']:.4f}")
        print(f"      time   : {elapsed:.1f}s")
        results[clf_name] = res

    print("\n[5] Test set evaluation")
    X_test_combined = pd.concat([X_test, s1_test, s2_test], axis=1)
    train_med = pd.concat([X_train, s1_train, s2_train], axis=1).median()
    X_test_clean = X_test_combined.fillna(train_med)

    test_signals = pd.DataFrame({"y_true": y_test.values}, index=y_test.index)
    rows = []
    for clf_name, res in results.items():
        model = res["model"]
        pred = model.predict(X_test_clean)
        proba = model.predict_proba(X_test_clean)
        m = compute_all_metrics(y_test.values, pred, y_proba=proba, classes=list(model.classes_))
        test_signals[f"{clf_name}_pred"] = pred
        for i, c in enumerate(model.classes_):
            test_signals[f"{clf_name}_proba_{c}"] = proba[:, i]
        rows.append({
            "model": clf_name.upper(),
            "WF_f1": res["avg_f1_macro"],
            "test_acc": m["accuracy"],
            "test_f1": m["f1_macro"],
            "test_balanced_acc": m["balanced_accuracy"],
            "test_mcc": m["mcc"],
        })
    summary = pd.DataFrame(rows).round(4).set_index("model")
    print("\n=== Stage 3 v2 — Test summary ===")
    print(summary.to_string())
    save_csv(summary, PROJECT_ROOT / "data" / "labels" / "btc_stage3_v2_summary.csv")

    print("\n=== Test predictions distribution ===")
    for clf_name in results:
        dist = (test_signals[f"{clf_name}_pred"].value_counts(normalize=True) * 100).round(1).to_dict()
        print(f"  {clf_name:14s}: {dist}")
    print(f"  {'TRUE':14s}: {(test_signals['y_true'].value_counts(normalize=True) * 100).round(1).to_dict()}")

    print("\n[6] Backtest comparison")
    test_close = btc["Close"].loc[y_test.index]
    bt = Backtester()
    bh = bt.run_buy_and_hold(test_close)

    all_strategies = {}
    for clf_name in results:
        sig = test_signals[f"{clf_name}_pred"]
        all_strategies[f"v2 {clf_name.upper()}"] = bt.run(sig, test_close)

    v1_signals = pd.read_csv(
        PROJECT_ROOT / "data" / "labels" / "btc_test_signals.csv",
        index_col=0, parse_dates=True,
    )
    all_strategies["v1 LDA"] = bt.run(v1_signals["lda_pred"], btc["Close"].loc[v1_signals.index])
    all_strategies["v1 MLP"] = bt.run(v1_signals["mlp_pred"], btc["Close"].loc[v1_signals.index])

    bt_rows = []
    for k, v in all_strategies.items():
        bt_rows.append({
            "strategy": k,
            "total_return": v["total_return"],
            "sharpe": v["sharpe_ratio"],
            "max_dd": v["max_drawdown"],
            "n_trades": v["n_trades"],
            "win_rate": v["win_rate"],
        })
    bt_rows.append({
        "strategy": "Buy & Hold",
        "total_return": float(bh.iloc[-1] / bh.iloc[0] - 1),
        "sharpe": float(test_close.pct_change().mean() / test_close.pct_change().std() * np.sqrt(252)),
        "max_dd": float(((bh - bh.cummax()) / bh.cummax()).min()),
        "n_trades": 1, "win_rate": np.nan,
    })
    bt_summary = pd.DataFrame(bt_rows).round(4)
    print("\n=== Backtest comparison ===")
    print(bt_summary.to_string(index=False))

    print("\n[7] Save artifacts + equity-curve plot")
    fig, ax = plt.subplots(figsize=(14, 6))
    color_map = {
        "v1 LDA": "#7F8C8D", "v1 MLP": "#BDC3C7",
        "v2 LDA": "#3498DB", "v2 MLP": "#9B59B6",
    }
    for label, result in all_strategies.items():
        ax.plot(
            result["equity_curve"].index, result["equity_curve"].values,
            label=f"{label} ret={result['total_return']:+.1%} sh={result['sharpe_ratio']:.2f}",
            color=color_map.get(label, "#34495E"),
            linewidth=1.3, alpha=0.85,
        )
    ax.plot(
        bh.index, bh.values,
        label=f"Buy&Hold ret={bh.iloc[-1] / bh.iloc[0] - 1:+.1%}",
        color="#2C3E50", linewidth=1.5, linestyle="--",
    )
    ax.set_title("v1 vs v2 strategies — equity curves on test period")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PROJECT_ROOT / "reports" / "iter2_equity_v1_vs_v2.png",
        dpi=120, bbox_inches="tight",
    )
    plt.close(fig)

    models_dir = PROJECT_ROOT / "app" / "models"
    for clf_name, res in results.items():
        joblib.dump(res["model"], models_dir / f"stage3_{clf_name}_v2.joblib")
    save_csv(test_signals, PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2.csv")
    save_csv(bt_summary, PROJECT_ROOT / "data" / "labels" / "btc_backtest_v2_summary.csv")

    best_clf = summary["test_f1"].idxmax()
    print(f"\nBest v2 model by test F1: {best_clf} (f1={summary.loc[best_clf]['test_f1']:.4f})")
    print("v2 artifacts saved.")


if __name__ == "__main__":
    main()
