"""
v2/feature-selection — final retrain with the winning B2 tech subset.

B2 tech subset: 29 → 24 (drop 5 long-trend cols log_ret_50d, log_ret_100d,
above_sma_200, adx_value, sharpe_proxy_20d).

Stage 3 input (per config A4): 24 tech + 3 (s1) + 3 (s2) = 30 features.

Reuses v1 data + v1 Stage 1/2 OOF posteriors (no Stage 1/2 retrain
needed — only the Stage 3 input feature set changes).

Outputs:
  data/labels/btc_stage3_b2_full_summary.csv
  data/labels/btc_stage3_b2_zigzag_summary.csv
  data/labels/btc_test_signals_b2.csv
  data/labels/btc_test_signals_b2_zigzag.csv
  data/labels/btc_backtest_b2_summary.csv
  data/labels/final_iter6_b2_summary.csv
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

from src.evaluation.backtester import Backtester
from src.evaluation.metrics import compute_all_metrics
from src.models.stage3_trainer import tune_stage3
from src.utils.config import cfg
from src.utils.helpers import chronological_train_test_split, save_csv

config = cfg()
TEST_SIZE = config["training"]["test_size"]
N_TRIALS = {"lda": 8, "mlp": 6, "xgboost": 8, "lightgbm": 8, "random_forest": 8}
STEP_MONTHS = 6
MIN_TRAIN_MONTHS = 12

B2_DROP = ["log_ret_50d", "log_ret_100d", "above_sma_200", "adx_value", "sharpe_proxy_20d"]


def load() -> dict:
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    tech_full = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_features_stage3_v2.csv",
                            index_col=0, parse_dates=True)
    tech = tech_full[[c for c in tech_full.columns if c not in B2_DROP]].copy()
    print(f"  tech full {tech_full.shape} → B2 subset {tech.shape}")
    print(f"  dropped: {B2_DROP}")

    y = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_adaptive.csv",
                    index_col=0, parse_dates=True).iloc[:, 0]
    s1_sma = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_stage1_oof_lda.csv",
                         index_col=0, parse_dates=True)
    s1_zz = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_stage1_oof_lda_zigzag.csv",
                        index_col=0, parse_dates=True)
    s2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv",
                     index_col=0, parse_dates=True)
    return {"btc": btc, "tech": tech, "y": y, "s1_sma": s1_sma, "s1_zz": s1_zz, "s2": s2}


def run() -> None:
    print("=== B2 7-model retrain ===")
    d = load()
    btc, tech, y, s1_sma, s1_zz, s2 = d["btc"], d["tech"], d["y"], d["s1_sma"], d["s1_zz"], d["s2"]
    bt = Backtester()
    out = {"summary_ab": [], "summary_zz": [], "backtest": []}

    def split(s1):
        common = tech.index.intersection(y.index).intersection(s1.index).intersection(s2.index)
        X_tr, X_te = chronological_train_test_split(tech.loc[common], test_size=TEST_SIZE)
        return (X_tr, X_te,
                y.loc[X_tr.index], y.loc[X_te.index],
                s1.loc[X_tr.index], s1.loc[X_te.index],
                s2.loc[X_tr.index], s2.loc[X_te.index])

    print("\n---- Phase A+B (SMA Stage 1 OOF) ----")
    X_tr, X_te, y_tr, y_te, s1_tr, s1_te, s2_tr, s2_te = split(s1_sma)
    print(f"  Train={X_tr.shape}  Test={X_te.shape}  ({y_te.index.min().date()} → {y_te.index.max().date()})")
    train_med = pd.concat([X_tr, s1_tr, s2_tr], axis=1).median()
    test_close = btc["Close"].loc[y_te.index]
    test_signals = pd.DataFrame({"y_true": y_te.values}, index=y_te.index)

    for clf in ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]:
        print(f"\n  >>> Stage 3 {clf.upper()}")
        t0 = time.time()
        res = tune_stage3(
            X_tr, y_tr, s1_tr, s2_tr, classifier_name=clf,
            n_trials=N_TRIALS.get(clf, 6), save_model=False,
            step_months=STEP_MONTHS, min_train_months=MIN_TRAIN_MONTHS,
        )
        print(f"      WF f1={res['avg_f1_macro']:.4f}, {time.time()-t0:.1f}s")
        Xc = pd.concat([X_te, s1_te, s2_te], axis=1).fillna(train_med)
        pred = res["model"].predict(Xc); proba = res["model"].predict_proba(Xc)
        m = compute_all_metrics(y_te.values, pred, y_proba=proba, classes=list(res["model"].classes_))
        test_signals[f"{clf}_pred"] = pred
        for i, c in enumerate(res["model"].classes_):
            test_signals[f"{clf}_proba_{c}"] = proba[:, i]
        out["summary_ab"].append({
            "model": clf.upper(), "WF_f1": res["avg_f1_macro"],
            "test_acc": m["accuracy"], "test_f1": m["f1_macro"],
            "test_balanced_acc": m["balanced_accuracy"], "test_mcc": m["mcc"],
        })
        bt_res = bt.run(pd.Series(pred, index=y_te.index), test_close)
        out["backtest"].append({
            "strategy": f"B2 {clf.upper()}",
            "total_return": bt_res["total_return"], "sharpe": bt_res["sharpe_ratio"],
            "max_dd": bt_res["max_drawdown"], "n_trades": bt_res["n_trades"],
            "win_rate": bt_res["win_rate"],
        })

    print("\n---- Phase C (ZigZag Stage 1 OOF) ----")
    X_tr, X_te, y_tr, y_te, s1_tr, s1_te, s2_tr, s2_te = split(s1_zz)
    train_med_z = pd.concat([X_tr, s1_tr, s2_tr], axis=1).median()
    test_close_z = btc["Close"].loc[y_te.index]
    test_signals_zz = pd.DataFrame({"y_true": y_te.values}, index=y_te.index)

    for clf in ["xgboost", "mlp"]:
        print(f"\n  >>> Stage 3 ZZ {clf.upper()}")
        t0 = time.time()
        res = tune_stage3(
            X_tr, y_tr, s1_tr, s2_tr, classifier_name=clf,
            n_trials=N_TRIALS.get(clf, 6), save_model=False,
            step_months=STEP_MONTHS, min_train_months=MIN_TRAIN_MONTHS,
        )
        print(f"      WF f1={res['avg_f1_macro']:.4f}, {time.time()-t0:.1f}s")
        Xc = pd.concat([X_te, s1_te, s2_te], axis=1).fillna(train_med_z)
        pred = res["model"].predict(Xc); proba = res["model"].predict_proba(Xc)
        m = compute_all_metrics(y_te.values, pred, y_proba=proba, classes=list(res["model"].classes_))
        test_signals_zz[f"{clf}_pred"] = pred
        for i, c in enumerate(res["model"].classes_):
            test_signals_zz[f"{clf}_proba_{c}"] = proba[:, i]
        out["summary_zz"].append({
            "model": f"{clf.upper()}_zigzag", "WF_f1": res["avg_f1_macro"],
            "test_acc": m["accuracy"], "test_f1": m["f1_macro"],
            "test_balanced_acc": m["balanced_accuracy"], "test_mcc": m["mcc"],
        })
        bt_res = bt.run(pd.Series(pred, index=y_te.index), test_close_z)
        out["backtest"].append({
            "strategy": f"B2 ZZ {clf.upper()}",
            "total_return": bt_res["total_return"], "sharpe": bt_res["sharpe_ratio"],
            "max_dd": bt_res["max_drawdown"], "n_trades": bt_res["n_trades"],
            "win_rate": bt_res["win_rate"],
        })

    summary_ab = pd.DataFrame(out["summary_ab"]).round(4).set_index("model")
    summary_zz = pd.DataFrame(out["summary_zz"]).round(4).set_index("model")
    print("\n=== Stage 3 B2 Phase A+B test summary ===")
    print(summary_ab.to_string())
    print("\n=== Stage 3 B2 Phase C (ZigZag) test summary ===")
    print(summary_zz.to_string())
    save_csv(summary_ab, PROJECT_ROOT / "data" / "labels" / "btc_stage3_b2_full_summary.csv")
    save_csv(summary_zz, PROJECT_ROOT / "data" / "labels" / "btc_stage3_b2_zigzag_summary.csv")
    save_csv(test_signals, PROJECT_ROOT / "data" / "labels" / "btc_test_signals_b2.csv")
    save_csv(test_signals_zz, PROJECT_ROOT / "data" / "labels" / "btc_test_signals_b2_zigzag.csv")

    bt_df = pd.DataFrame(out["backtest"]).round(4)
    bh = bt.run_buy_and_hold(test_close)
    bh_row = {
        "strategy": "Buy & Hold",
        "total_return": round(bh.iloc[-1]/bh.iloc[0] - 1, 4),
        "sharpe": round(test_close.pct_change().mean()/test_close.pct_change().std()*np.sqrt(252), 4),
        "max_dd": round(((bh-bh.cummax())/bh.cummax()).min(), 4),
        "n_trades": 1, "win_rate": np.nan,
    }
    full_bt = pd.concat([bt_df, pd.DataFrame([bh_row])], ignore_index=True)
    save_csv(full_bt, PROJECT_ROOT / "data" / "labels" / "btc_backtest_b2_summary.csv")
    print("\n=== Backtest comparison B2 ===")
    print(full_bt.to_string(index=False))

    rows = []
    for clf, lbl in [("LDA","LDA"),("MLP","MLP"),("XGBOOST","XGB"),
                     ("LIGHTGBM","LGBM"),("RANDOM_FOREST","RF")]:
        m = summary_ab.loc[clf]
        b = bt_df[bt_df["strategy"].eq(f"B2 {clf}")].iloc[0]
        rows.append({"Model": lbl, "Test Acc": m["test_acc"], "Test F1": m["test_f1"],
                     "MCC": m["test_mcc"], "Return": f"{b['total_return']*100:+.1f}%",
                     "Sharpe": b["sharpe"], "MaxDD": f"{b['max_dd']*100:.1f}%",
                     "Trades": int(b["n_trades"]),
                     "Win%": (f"{b['win_rate']*100:.1f}%" if b["win_rate"] else "-")})
    for clf, lbl in [("XGBOOST","ZZ-XGB"), ("MLP","ZZ-MLP")]:
        m = summary_zz.loc[f"{clf}_zigzag"]
        b = bt_df[bt_df["strategy"].eq(f"B2 ZZ {clf}")].iloc[0]
        rows.append({"Model": lbl, "Test Acc": m["test_acc"], "Test F1": m["test_f1"],
                     "MCC": m["test_mcc"], "Return": f"{b['total_return']*100:+.1f}%",
                     "Sharpe": b["sharpe"], "MaxDD": f"{b['max_dd']*100:.1f}%",
                     "Trades": int(b["n_trades"]),
                     "Win%": (f"{b['win_rate']*100:.1f}%" if b["win_rate"] else "-")})
    rows.append({"Model": "Buy & Hold", "Test Acc": "-", "Test F1": "-", "MCC": "-",
                 "Return": f"{bh_row['total_return']*100:+.1f}%",
                 "Sharpe": bh_row["sharpe"], "MaxDD": f"{bh_row['max_dd']*100:.1f}%",
                 "Trades": 1, "Win%": "-"})
    final_df = pd.DataFrame(rows)
    print("\n=== final_iter6_b2_summary.csv ===")
    print(final_df.to_string(index=False))
    save_csv(final_df, PROJECT_ROOT / "data" / "labels" / "final_iter6_b2_summary.csv")


if __name__ == "__main__":
    run()
