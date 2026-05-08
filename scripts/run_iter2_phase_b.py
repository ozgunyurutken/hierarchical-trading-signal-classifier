"""
Iter 2 Phase B — XGBoost + LightGBM + RandomForest on v2 features + adaptive labels.

Run with:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/run_iter2_phase_b.py

The OMP_NUM_THREADS=1 environment + n_jobs=1 inside the wrappers prevents the libomp
segfault we hit on macOS Python 3.11 / Apple Silicon.
"""
from __future__ import annotations

import os
# Belt-and-suspenders: also set the env in-process before any C-extension import
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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
from src.models.stage3_trainer import tune_stage3
from src.utils.config import cfg
from src.utils.helpers import chronological_train_test_split, save_csv

config = cfg()
TEST_SIZE = config["training"]["test_size"]
STEP_MONTHS = 6
MIN_TRAIN_MONTHS = 12
N_TRIALS = {"xgboost": 12, "lightgbm": 12, "random_forest": 10}


def main() -> None:
    btc = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
        index_col=0, parse_dates=True,
    )

    # Reuse v2 features computed in Phase A
    stage3_v2 = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "btc_features_stage3_v2.csv",
        index_col=0, parse_dates=True,
    )
    y_v2 = pd.read_csv(
        PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_adaptive.csv",
        index_col=0, parse_dates=True,
    ).iloc[:, 0]
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

    print(f"Features={X_osc.shape}, Train={X_train.shape}, Test={X_test.shape}")
    print(f"Threading guard: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")

    print("\n[1] Tune 3 tree-based classifiers via Optuna walk-forward")
    results = {}
    for clf_name in ["xgboost", "lightgbm", "random_forest"]:
        print(f"\n  >>> {clf_name.upper()}")
        t0 = time.time()
        try:
            res = tune_stage3(
                X_train, y_train, s1_train, s2_train,
                classifier_name=clf_name,
                n_trials=N_TRIALS[clf_name],
                save_model=False,
                step_months=STEP_MONTHS,
                min_train_months=MIN_TRAIN_MONTHS,
            )
            print(f"      best   : {res['best_params']}")
            print(f"      WF f1  : {res['avg_f1_macro']:.4f}")
            print(f"      time   : {time.time() - t0:.1f}s")
            results[clf_name] = res
        except Exception as e:
            print(f"      FAILED : {type(e).__name__}: {e}")

    if not results:
        print("\nAll tree-based classifiers failed; aborting.")
        return

    print("\n[2] Test set evaluation")
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
    print("\n=== Stage 3 v2 (tree models) test summary ===")
    print(summary.to_string())

    print("\n=== Test predictions distribution ===")
    for clf_name in results:
        dist = (test_signals[f"{clf_name}_pred"].value_counts(normalize=True) * 100).round(1).to_dict()
        print(f"  {clf_name:14s}: {dist}")
    print(f"  {'TRUE':14s}: {(test_signals['y_true'].value_counts(normalize=True) * 100).round(1).to_dict()}")

    # Combined v2 summary (Phase A + Phase B)
    phase_a = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_stage3_v2_summary.csv", index_col=0)
    combined = pd.concat([phase_a, summary])
    save_csv(combined, PROJECT_ROOT / "data" / "labels" / "btc_stage3_v2_full_summary.csv")
    print("\n=== Combined v2 (Phase A + B) summary ===")
    print(combined.to_string())

    print("\n[3] Backtest")
    test_close = btc["Close"].loc[y_test.index]
    bt = Backtester()
    bh = bt.run_buy_and_hold(test_close)

    all_strategies = {}
    for clf_name in results:
        sig = test_signals[f"{clf_name}_pred"]
        all_strategies[f"v2 {clf_name.upper()}"] = bt.run(sig, test_close)

    # Reuse v1 + v2 LDA/MLP signals for full comparison
    v1 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_test_signals.csv", index_col=0, parse_dates=True)
    v2_a = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2.csv", index_col=0, parse_dates=True)
    all_strategies["v1 MLP"] = bt.run(v1["mlp_pred"], btc["Close"].loc[v1.index])
    all_strategies["v2 MLP"] = bt.run(v2_a["mlp_pred"], btc["Close"].loc[v2_a.index])

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

    # Equity curve plot
    fig, ax = plt.subplots(figsize=(14, 6))
    color_map = {
        "v1 MLP": "#BDC3C7", "v2 MLP": "#9B59B6",
        "v2 XGBOOST": "#E67E22", "v2 LIGHTGBM": "#27AE60", "v2 RANDOM_FOREST": "#16A085",
    }
    for label, result in all_strategies.items():
        ax.plot(
            result["equity_curve"].index, result["equity_curve"].values,
            label=f"{label} ret={result['total_return']:+.1%} sh={result['sharpe_ratio']:.2f}",
            color=color_map.get(label, "#34495E"), linewidth=1.3, alpha=0.85,
        )
    ax.plot(
        bh.index, bh.values,
        label=f"Buy&Hold ret={bh.iloc[-1] / bh.iloc[0] - 1:+.1%}",
        color="#2C3E50", linewidth=1.5, linestyle="--",
    )
    ax.set_title("Phase B — tree-based models vs MLP baselines")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        PROJECT_ROOT / "reports" / "iter2_phase_b_equity.png",
        dpi=120, bbox_inches="tight",
    )
    plt.close(fig)

    models_dir = PROJECT_ROOT / "app" / "models"
    for clf_name, res in results.items():
        joblib.dump(res["model"], models_dir / f"stage3_{clf_name}_v2.joblib")
    save_csv(test_signals, PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2_phase_b.csv")
    save_csv(bt_summary, PROJECT_ROOT / "data" / "labels" / "btc_backtest_v2_phase_b_summary.csv")
    print("\nPhase B artifacts saved.")


if __name__ == "__main__":
    main()
