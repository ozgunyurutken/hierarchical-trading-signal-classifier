"""
Step B of the v3 rerun: retrain all Stage 3 v2 models on the new s2_oof
(after the US2Y / yield-curve bug fix).

Reuses pre-computed Stage 3 features (btc_features_stage3_v2.csv) — they
are OHLCV-derived only and do not depend on US2Y, so they are still valid.
Reuses pre-computed Stage 1 OOF (SMA + ZigZag) for the same reason.

Models retrained:
  Phase A (SMA Stage 1 OOF + adaptive label):    LDA, MLP
  Phase B (SMA Stage 1 OOF + adaptive label):    XGBoost, LightGBM, RandomForest
  Phase C (ZigZag Stage 1 OOF + adaptive label): XGBoost, MLP

Outputs (overwrite):
  data/labels/btc_stage3_v2_summary.csv
  data/labels/btc_stage3_v2_full_summary.csv
  data/labels/btc_stage3_v2_zigzag_summary.csv
  data/labels/btc_test_signals_v2{,_phase_b,_zigzag}.csv
  data/labels/btc_backtest_v2{,_phase_b,_zigzag}_summary.csv
  data/labels/final_iter2_summary_table.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
import shutil
import datetime as _dt
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
STEP_MONTHS = 6
MIN_TRAIN_MONTHS = 12
N_TRIALS = {"lda": 8, "mlp": 6, "xgboost": 8, "lightgbm": 8, "random_forest": 8}


def load_inputs():
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    stage3_v2 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_features_stage3_v2.csv",
                            index_col=0, parse_dates=True)
    y_v2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_adaptive.csv",
                       index_col=0, parse_dates=True).iloc[:, 0]
    s1_sma = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_stage1_oof_lda.csv",
                         index_col=0, parse_dates=True)
    s1_zz = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_stage1_oof_lda_zigzag.csv",
                        index_col=0, parse_dates=True)
    s2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv",
                     index_col=0, parse_dates=True)
    return btc, stage3_v2, y_v2, s1_sma, s1_zz, s2


def split_chronological(X_osc, y, s1, s2):
    common = X_osc.index.intersection(y.index).intersection(s1.index).intersection(s2.index)
    X_osc, y, s1, s2 = X_osc.loc[common], y.loc[common], s1.loc[common], s2.loc[common]
    X_train, X_test = chronological_train_test_split(X_osc, test_size=TEST_SIZE)
    y_train, y_test = y.loc[X_train.index], y.loc[X_test.index]
    s1_train, s1_test = s1.loc[X_train.index], s1.loc[X_test.index]
    s2_train, s2_test = s2.loc[X_train.index], s2.loc[X_test.index]
    return X_train, X_test, y_train, y_test, s1_train, s1_test, s2_train, s2_test


def train_one_model(clf_name, X_train, y_train, s1_train, s2_train):
    t0 = time.time()
    res = tune_stage3(
        X_train, y_train, s1_train, s2_train,
        classifier_name=clf_name,
        n_trials=N_TRIALS.get(clf_name, 6),
        save_model=False,
        step_months=STEP_MONTHS,
        min_train_months=MIN_TRAIN_MONTHS,
    )
    print(f"      WF f1={res['avg_f1_macro']:.4f}, {time.time()-t0:.1f}s")
    return res


def evaluate_test(model, y_test, X_test, s1_test, s2_test, train_med_components):
    X_test_combined = pd.concat([X_test, s1_test, s2_test], axis=1)
    X_test_clean = X_test_combined.fillna(train_med_components)
    pred = model.predict(X_test_clean)
    proba = model.predict_proba(X_test_clean)
    m = compute_all_metrics(y_test.values, pred, y_proba=proba, classes=list(model.classes_))
    return pred, proba, m


def backup_old_summaries() -> None:
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    labels_dir = PROJECT_ROOT / "data" / "labels"
    targets = [
        "btc_stage3_v2_summary.csv",
        "btc_stage3_v2_full_summary.csv",
        "btc_stage3_v2_zigzag_summary.csv",
        "btc_test_signals_v2.csv",
        "btc_test_signals_v2_phase_b.csv",
        "btc_test_signals_v2_zigzag.csv",
        "btc_backtest_v2_summary.csv",
        "btc_backtest_v2_phase_b_summary.csv",
        "btc_backtest_v2_zigzag_summary.csv",
        "final_iter2_summary_table.csv",
    ]
    for name in targets:
        src = labels_dir / name
        if src.exists():
            dst = src.with_name(f"{src.stem}.backup_{ts}{src.suffix}")
            shutil.copy(src, dst)
            print(f"  Backup: {dst.name}")


def main() -> None:
    print("[0] Backup old summary CSVs (v2 → v3 transition)")
    backup_old_summaries()

    print("\n[1] Load inputs")
    btc, stage3_v2, y_v2, s1_sma, s1_zz, s2 = load_inputs()
    print(f"  Stage3 v2 features: {stage3_v2.shape}")
    print(f"  Stage 1 OOF SMA   : {s1_sma.shape}")
    print(f"  Stage 1 OOF ZigZag: {s1_zz.shape}")
    print(f"  Stage 2 OOF (NEW) : {s2.shape}")
    print(f"  y_signal adaptive : {y_v2.shape}")

    # =========================================================================
    # Phase A + B (SMA Stage 1 OOF)
    # =========================================================================
    print("\n=== PHASE A+B: SMA Stage 1 OOF + adaptive label ===")
    X_tr, X_te, y_tr, y_te, s1_tr, s1_te, s2_tr, s2_te = split_chronological(
        stage3_v2, y_v2, s1_sma, s2,
    )
    print(f"  Train={X_tr.shape}, Test={X_te.shape}")

    train_med = pd.concat([X_tr, s1_tr, s2_tr], axis=1).median()
    test_close = btc["Close"].loc[y_te.index]
    bt = Backtester()

    test_signals = pd.DataFrame({"y_true": y_te.values}, index=y_te.index)
    rows_summary, rows_backtest = [], []

    for clf_name in ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]:
        print(f"\n  >>> Stage 3 {clf_name.upper()}")
        res = train_one_model(clf_name, X_tr, y_tr, s1_tr, s2_tr)
        pred, proba, m = evaluate_test(res["model"], y_te, X_te, s1_te, s2_te, train_med)
        test_signals[f"{clf_name}_pred"] = pred
        for i, c in enumerate(res["model"].classes_):
            test_signals[f"{clf_name}_proba_{c}"] = proba[:, i]
        rows_summary.append({
            "model": clf_name.upper(),
            "WF_f1": res["avg_f1_macro"],
            "test_acc": m["accuracy"], "test_f1": m["f1_macro"],
            "test_balanced_acc": m["balanced_accuracy"], "test_mcc": m["mcc"],
        })
        bt_res = bt.run(pd.Series(pred, index=y_te.index), test_close)
        rows_backtest.append({
            "strategy": f"v3 {clf_name.upper()}",
            "total_return": bt_res["total_return"],
            "sharpe": bt_res["sharpe_ratio"],
            "max_dd": bt_res["max_drawdown"],
            "n_trades": bt_res["n_trades"],
            "win_rate": bt_res["win_rate"],
        })

    summary_ab = pd.DataFrame(rows_summary).round(4).set_index("model")
    print("\n=== Stage 3 v3 (Phase A+B) — Test summary ===")
    print(summary_ab.to_string())

    # Buy & Hold for benchmark
    bh = bt.run_buy_and_hold(test_close)
    rows_backtest.append({
        "strategy": "Buy & Hold",
        "total_return": float(bh.iloc[-1] / bh.iloc[0] - 1),
        "sharpe": float(test_close.pct_change().mean() / test_close.pct_change().std() * np.sqrt(252)),
        "max_dd": float(((bh - bh.cummax()) / bh.cummax()).min()),
        "n_trades": 1, "win_rate": np.nan,
    })
    bt_summary_ab = pd.DataFrame(rows_backtest).round(4)
    print("\n=== Backtest comparison (v3 Phase A+B + B&H) ===")
    print(bt_summary_ab.to_string(index=False))

    # Save Phase A+B artifacts
    save_csv(summary_ab.iloc[:2], PROJECT_ROOT / "data" / "labels" / "btc_stage3_v2_summary.csv")
    save_csv(summary_ab, PROJECT_ROOT / "data" / "labels" / "btc_stage3_v2_full_summary.csv")
    save_csv(test_signals, PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2.csv")
    # Phase B specific (XGB/LGBM/RF) — separate CSV for app + reports
    pb_signals = test_signals[
        ["y_true"]
        + [c for c in test_signals.columns if any(c.startswith(k) for k in ["xgboost", "lightgbm", "random_forest"])]
    ]
    save_csv(pb_signals, PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2_phase_b.csv")
    bt_ab_no_bh = bt_summary_ab[~bt_summary_ab["strategy"].eq("Buy & Hold")]
    bt_pa = bt_ab_no_bh[bt_ab_no_bh["strategy"].str.contains("LDA|MLP", regex=True)]
    bt_pb = bt_ab_no_bh[bt_ab_no_bh["strategy"].str.contains("XGBOOST|LIGHTGBM|RANDOM_FOREST", regex=True)]
    save_csv(pd.concat([bt_pa, bt_summary_ab[bt_summary_ab["strategy"].eq("Buy & Hold")]]),
             PROJECT_ROOT / "data" / "labels" / "btc_backtest_v2_summary.csv")
    save_csv(pd.concat([bt_pb, bt_summary_ab[bt_summary_ab["strategy"].eq("Buy & Hold")]]),
             PROJECT_ROOT / "data" / "labels" / "btc_backtest_v2_phase_b_summary.csv")

    # =========================================================================
    # Phase C (ZigZag Stage 1 OOF)
    # =========================================================================
    print("\n=== PHASE C: ZigZag Stage 1 OOF + adaptive label ===")
    X_trz, X_tez, y_trz, y_tez, s1_trz, s1_tez, s2_trz, s2_tez = split_chronological(
        stage3_v2, y_v2, s1_zz, s2,
    )
    print(f"  Train={X_trz.shape}, Test={X_tez.shape}")
    train_med_z = pd.concat([X_trz, s1_trz, s2_trz], axis=1).median()
    test_close_z = btc["Close"].loc[y_tez.index]

    test_signals_zz = pd.DataFrame({"y_true": y_tez.values}, index=y_tez.index)
    zz_summary, zz_backtest = [], []

    for clf_name in ["xgboost", "mlp"]:
        print(f"\n  >>> Stage 3 ZZ {clf_name.upper()}")
        res = train_one_model(clf_name, X_trz, y_trz, s1_trz, s2_trz)
        pred, proba, m = evaluate_test(res["model"], y_tez, X_tez, s1_tez, s2_tez, train_med_z)
        test_signals_zz[f"{clf_name}_pred"] = pred
        for i, c in enumerate(res["model"].classes_):
            test_signals_zz[f"{clf_name}_proba_{c}"] = proba[:, i]
        zz_summary.append({
            "model": f"{clf_name.upper()}_zigzag",
            "WF_f1": res["avg_f1_macro"],
            "test_acc": m["accuracy"], "test_f1": m["f1_macro"],
            "test_balanced_acc": m["balanced_accuracy"], "test_mcc": m["mcc"],
        })
        bt_res = bt.run(pd.Series(pred, index=y_tez.index), test_close_z)
        zz_backtest.append({
            "strategy": f"v3 ZZ {clf_name.upper()}",
            "total_return": bt_res["total_return"],
            "sharpe_ratio": bt_res["sharpe_ratio"],
            "max_drawdown": bt_res["max_drawdown"],
            "n_trades": bt_res["n_trades"],
            "win_rate": bt_res["win_rate"],
        })

    zz_summary_df = pd.DataFrame(zz_summary).round(4).set_index("model")
    print("\n=== Stage 3 v3 ZigZag — Test summary ===")
    print(zz_summary_df.to_string())

    bh_z = bt.run_buy_and_hold(test_close_z)
    zz_backtest.append({
        "strategy": "Buy & Hold",
        "total_return": float(bh_z.iloc[-1] / bh_z.iloc[0] - 1),
        "sharpe_ratio": float(test_close_z.pct_change().mean() / test_close_z.pct_change().std() * np.sqrt(252)),
        "max_drawdown": float(((bh_z - bh_z.cummax()) / bh_z.cummax()).min()),
        "n_trades": 1, "win_rate": np.nan,
    })
    zz_bt_df = pd.DataFrame(zz_backtest).round(4)
    print("\n=== Backtest comparison (v3 Phase C + B&H) ===")
    print(zz_bt_df.to_string(index=False))

    save_csv(zz_summary_df, PROJECT_ROOT / "data" / "labels" / "btc_stage3_v2_zigzag_summary.csv")
    save_csv(test_signals_zz, PROJECT_ROOT / "data" / "labels" / "btc_test_signals_v2_zigzag.csv")
    save_csv(zz_bt_df, PROJECT_ROOT / "data" / "labels" / "btc_backtest_v2_zigzag_summary.csv")

    # =========================================================================
    # Final aggregated summary
    # =========================================================================
    print("\n[FINAL] Build final_iter2_summary_table.csv (v3)")
    final_rows = []
    # SMA-based Phase A+B
    for clf_name, label_in_table in [
        ("lda", "LDA"), ("mlp", "MLP"), ("xgboost", "XGB"),
        ("lightgbm", "LGBM"), ("random_forest", "RF"),
    ]:
        m = summary_ab.loc[clf_name.upper()]
        bt_row = bt_summary_ab[bt_summary_ab["strategy"].eq(f"v3 {clf_name.upper()}")].iloc[0]
        final_rows.append({
            "Model": label_in_table,
            "Test Acc": m["test_acc"], "Test F1": m["test_f1"], "MCC": m["test_mcc"],
            "Return": f"{bt_row['total_return']*100:+.1f}%",
            "Sharpe": bt_row["sharpe"], "MaxDD": f"{bt_row['max_dd']*100:.1f}%",
            "Trades": int(bt_row["n_trades"]),
            "Win%": (f"{bt_row['win_rate']*100:.1f}%" if bt_row["win_rate"] else "-"),
        })
    # ZigZag
    for clf_name, label_in_table in [("xgboost", "ZZ-XGB"), ("mlp", "ZZ-MLP")]:
        m = zz_summary_df.loc[f"{clf_name.upper()}_zigzag"]
        bt_row = zz_bt_df[zz_bt_df["strategy"].eq(f"v3 ZZ {clf_name.upper()}")].iloc[0]
        final_rows.append({
            "Model": label_in_table,
            "Test Acc": m["test_acc"], "Test F1": m["test_f1"], "MCC": m["test_mcc"],
            "Return": f"{bt_row['total_return']*100:+.1f}%",
            "Sharpe": bt_row["sharpe_ratio"], "MaxDD": f"{bt_row['max_drawdown']*100:.1f}%",
            "Trades": int(bt_row["n_trades"]),
            "Win%": (f"{bt_row['win_rate']*100:.1f}%" if bt_row["win_rate"] else "-"),
        })
    # Buy & Hold
    bh_total = float(bh.iloc[-1] / bh.iloc[0] - 1)
    final_rows.append({
        "Model": "Buy & Hold",
        "Test Acc": "-", "Test F1": "-", "MCC": "-",
        "Return": f"{bh_total*100:+.1f}%",
        "Sharpe": float(test_close.pct_change().mean() / test_close.pct_change().std() * np.sqrt(252)),
        "MaxDD": f"{((bh - bh.cummax()) / bh.cummax()).min()*100:.1f}%",
        "Trades": 1, "Win%": "-",
    })
    final_df = pd.DataFrame(final_rows)
    print(final_df.to_string(index=False))
    save_csv(final_df, PROJECT_ROOT / "data" / "labels" / "final_iter2_summary_table.csv")

    print("\nv3 retrain complete.")


if __name__ == "__main__":
    main()
