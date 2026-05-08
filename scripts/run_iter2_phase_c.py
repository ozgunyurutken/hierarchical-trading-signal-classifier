"""
Iter 2 Phase C — break the SMA tautology in Stage 1.

The MVP labelled Stage 1 trends from SMA crossover and then trained on features
that include those same SMAs, producing 82% MLP test accuracy through trivial
rule recovery rather than real prediction. Phase C uses a causal ZigZag swing
labeller that does not depend on the SMA features, then retrains Stage 1 + Stage 3.

Outputs:
  - data/labels/btc_trend_labels_zigzag.csv
  - data/labels/btc_stage1_oof_lda_zigzag.csv
  - data/labels/btc_stage3_v2_zigzag_summary.csv
  - data/labels/btc_test_signals_v2_zigzag.csv (XGBoost path, best Phase B model)
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

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.backtester import Backtester
from src.evaluation.metrics import compute_all_metrics
from src.labels.trend_labels import generate_trend_labels, generate_trend_labels_zigzag
from src.models.stage1_trainer import tune_stage1
from src.models.stage3_trainer import tune_stage3
from src.utils.config import cfg
from src.utils.helpers import chronological_train_test_split, save_csv

config = cfg()
TEST_SIZE = config["training"]["test_size"]
STEP_MONTHS = 6
MIN_TRAIN_MONTHS = 12


def assemble_full_period_proba(result_dict, X_train, X_test):
    oof_train = result_dict["oof_predictions"].dropna()
    model = result_dict["model"]
    X_test_clean = X_test.fillna(X_train.median())
    test_proba = model.predict_proba(X_test_clean)
    test_df = pd.DataFrame(test_proba, index=X_test.index, columns=oof_train.columns)
    return pd.concat([oof_train, test_df]).sort_index()


def main() -> None:
    print("[1] Generate causal ZigZag trend labels (threshold=10%)")
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    close = btc["Close"]

    sma_labels = generate_trend_labels(btc)
    zz_labels = generate_trend_labels_zigzag(close, threshold=0.10, sideways_band=0.03)

    sma_dist = (sma_labels.value_counts(normalize=True) * 100).round(1)
    zz_dist = (zz_labels.value_counts(normalize=True) * 100).round(1)
    print(f"  SMA dist : {sma_dist.to_dict()}")
    print(f"  ZigZag   : {zz_dist.to_dict()}")

    save_csv(zz_labels.to_frame(), PROJECT_ROOT / "data" / "labels" / "btc_trend_labels_zigzag.csv")

    # Quick plot of label disagreement on price chart
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(close.index, close.values, color="black", linewidth=0.6, alpha=0.5, label="BTC Close")
    color_map = {"Uptrend": "#2ECC71", "Downtrend": "#E74C3C", "Sideways": "#95A5A6"}
    for cls, color in color_map.items():
        mask = zz_labels == cls
        ax.scatter(zz_labels.index[mask], close.loc[zz_labels.index[mask]],
                   c=color, s=2, alpha=0.6, label=f"ZZ {cls}")
    ax.set_yscale("log"); ax.set_title("ZigZag trend labels (threshold=10%)")
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "reports" / "iter2_zigzag_labels.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print("\n[2] Retrain Stage 1 LDA + MLP with ZigZag labels")
    X_stage1 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_features_stage1.csv",
                           index_col=0, parse_dates=True).dropna()
    common1 = X_stage1.index.intersection(zz_labels.index)
    X_s1, y_s1 = X_stage1.loc[common1], zz_labels.loc[common1]
    X_s1_train, X_s1_test = chronological_train_test_split(X_s1, test_size=TEST_SIZE)
    y_s1_train, y_s1_test = y_s1.loc[X_s1_train.index], y_s1.loc[X_s1_test.index]

    stage1_results = {}
    for clf_name in ["lda", "mlp"]:
        print(f"\n  >>> Stage 1 {clf_name.upper()} on ZigZag")
        t0 = time.time()
        res = tune_stage1(
            X_s1_train, y_s1_train,
            classifier_name=clf_name,
            n_trials=8 if clf_name == "lda" else 6,
            save_model=False,
            step_months=STEP_MONTHS,
            min_train_months=MIN_TRAIN_MONTHS,
        )
        elapsed = time.time() - t0
        # Test eval
        pred = res["model"].predict(X_s1_test.fillna(X_s1_train.median()))
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_s1_test, pred)
        f1m = f1_score(y_s1_test, pred, average="macro", zero_division=0)
        print(f"      WF f1: {res['avg_f1_macro']:.4f}, test acc={acc:.4f}, test f1={f1m:.4f}, {elapsed:.1f}s")
        stage1_results[clf_name] = res
        stage1_results[clf_name]["test_acc"] = acc
        stage1_results[clf_name]["test_f1"] = f1m

    # Use LDA Stage 1 OOF for Stage 3 (consistent with iter 2 line)
    s1_full_proba = assemble_full_period_proba(stage1_results["lda"], X_s1_train, X_s1_test)
    save_csv(s1_full_proba, PROJECT_ROOT / "data" / "labels" / "btc_stage1_oof_lda_zigzag.csv")

    print("\n[3] Retrain Stage 3 (XGBoost — best Phase B model) with ZigZag-derived Stage 1 OOF")
    X_osc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_features_stage3_v2.csv",
                        index_col=0, parse_dates=True)
    y_v2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_adaptive.csv",
                       index_col=0, parse_dates=True).iloc[:, 0]
    s2_oof = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv",
                         index_col=0, parse_dates=True)

    common = (X_osc.index
              .intersection(y_v2.index)
              .intersection(s1_full_proba.index)
              .intersection(s2_oof.index))
    X_osc, y, s1, s2 = X_osc.loc[common], y_v2.loc[common], s1_full_proba.loc[common], s2_oof.loc[common]
    X_train, X_test = chronological_train_test_split(X_osc, test_size=TEST_SIZE)
    y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]
    s1_train, s1_test = s1.loc[X_train.index], s1.loc[X_test.index]
    s2_train, s2_test = s2.loc[X_train.index], s2.loc[X_test.index]

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    stage3_results = {}
    for clf_name in ["xgboost", "mlp"]:
        print(f"\n  >>> Stage 3 {clf_name.upper()} (Stage 1 = ZigZag-LDA)")
        t0 = time.time()
        res = tune_stage3(
            X_train, y_train, s1_train, s2_train,
            classifier_name=clf_name,
            n_trials=12 if clf_name == "xgboost" else 6,
            save_model=False,
            step_months=STEP_MONTHS,
            min_train_months=MIN_TRAIN_MONTHS,
        )
        elapsed = time.time() - t0
        print(f"      best: {res['best_params']}, WF f1={res['avg_f1_macro']:.4f}, {elapsed:.1f}s")
        stage3_results[clf_name] = res

    print("\n[4] Test evaluation + backtest")
    X_test_combined = pd.concat([X_test, s1_test, s2_test], axis=1)
    train_med = pd.concat([X_train, s1_train, s2_train], axis=1).median()
    X_test_clean = X_test_combined.fillna(train_med)

    test_signals = pd.DataFrame({"y_true": y_test.values}, index=y_test.index)
    rows = []
    for clf_name, res in stage3_results.items():
        model = res["model"]
        pred = model.predict(X_test_clean)
        proba = model.predict_proba(X_test_clean)
        m = compute_all_metrics(y_test.values, pred, y_proba=proba, classes=list(model.classes_))
        test_signals[f"{clf_name}_pred"] = pred
        for i, c in enumerate(model.classes_):
            test_signals[f"{clf_name}_proba_{c}"] = proba[:, i]
        rows.append({
            "model": f"{clf_name.upper()}_zigzag",
            "WF_f1": res["avg_f1_macro"],
            "test_acc": m["accuracy"],
            "test_f1": m["f1_macro"],
            "test_balanced_acc": m["balanced_accuracy"],
            "test_mcc": m["mcc"],
        })
    summary = pd.DataFrame(rows).round(4).set_index("model")
    print("\n=== Stage 3 v2 ZigZag summary ===")
    print(summary.to_string())

    # Backtest
    test_close = btc["Close"].loc[y_test.index]
    bt = Backtester()
    bh = bt.run_buy_and_hold(test_close)

    bt_rows = []
    for clf_name in stage3_results:
        sig = test_signals[f"{clf_name}_pred"]
        result = bt.run(sig, test_close)
        bt_rows.append({
            "strategy": f"v2 ZZ {clf_name.upper()}",
            **{k: result[k] for k in ["total_return", "sharpe_ratio", "max_drawdown", "n_trades", "win_rate"]},
        })

    # Reference: SMA-based versions
    v2_phase_b = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2_phase_b.csv",
                             index_col=0, parse_dates=True)
    v2_phase_a = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2.csv",
                             index_col=0, parse_dates=True)
    bt_rows.append({
        "strategy": "v2 SMA XGBOOST",
        **{k: bt.run(v2_phase_b["xgboost_pred"], btc["Close"].loc[v2_phase_b.index])[k]
           for k in ["total_return", "sharpe_ratio", "max_drawdown", "n_trades", "win_rate"]},
    })
    bt_rows.append({
        "strategy": "v2 SMA MLP",
        **{k: bt.run(v2_phase_a["mlp_pred"], btc["Close"].loc[v2_phase_a.index])[k]
           for k in ["total_return", "sharpe_ratio", "max_drawdown", "n_trades", "win_rate"]},
    })
    bt_rows.append({
        "strategy": "Buy & Hold",
        "total_return": float(bh.iloc[-1] / bh.iloc[0] - 1),
        "sharpe_ratio": float(test_close.pct_change().mean() / test_close.pct_change().std() * np.sqrt(252)),
        "max_drawdown": float(((bh - bh.cummax()) / bh.cummax()).min()),
        "n_trades": 1, "win_rate": np.nan,
    })
    bt_summary = pd.DataFrame(bt_rows).round(4)
    print("\n=== Backtest comparison (Phase C) ===")
    print(bt_summary.to_string(index=False))

    save_csv(summary, PROJECT_ROOT / "data" / "labels" / "btc_stage3_v2_zigzag_summary.csv")
    save_csv(bt_summary, PROJECT_ROOT / "data" / "labels" / "btc_backtest_v2_zigzag_summary.csv")
    save_csv(test_signals, PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2_zigzag.csv")

    print("\nPhase C artifacts saved.")


if __name__ == "__main__":
    main()
