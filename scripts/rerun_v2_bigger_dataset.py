"""
v2/bigger-dataset full retrain orchestrator.

Inputs already prepared:
  data/processed/btc_aligned_v2.csv  (4,538 rows, 2013-02-15 →)

Steps:
  1. Technical features  → btc_features_stage3_v2_bigger.csv  (~65 feat)
  2. Macro features      → btc_features_macro_bigger.csv      (~187 feat)
  3. Trend labels (SMA + ZigZag) on the new BTC series
  4. Signal labels (ATR-adaptive forward return)
  5. Stage 1 LDA OOF (5-fold) — SMA + ZigZag flavors
  6. Stage 2 GMM OOF (11-feature subset) — same feature set as v1
  7. Stage 3 retrain × 7 models (LDA, MLP, XGB, LGBM, RF, ZZ-XGB, ZZ-MLP)
     Optuna 8 trials, walk-forward CV (12-mo min train, 6-mo step)
  8. Backtest test predictions
  9. Write summary CSVs (suffix `_bigger`) + plot regen

This script overrides v2/bigger-dataset's working files but does NOT
touch v1's tagged commit (ab408d5). All artefacts get a `_bigger`
suffix to avoid clobbering v1 reference data when this branch merges.
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
from src.features.macro_features import compute_macro_features
from src.features.technical_indicators import (
    compute_oscillator_indicators,
    compute_trend_following_features,
    compute_volatility_indicators,
    compute_volume_indicators,
)
from src.labels.trend_labels import generate_trend_labels, generate_trend_labels_zigzag
from src.labels.signal_labels import generate_signal_labels
from src.labels.regime_labels import compute_oof_regime_posterior, predict_regime_posterior
from src.models.stage1_trainer import tune_stage1
from src.models.stage3_trainer import tune_stage3
from src.utils.config import cfg
from src.utils.helpers import chronological_train_test_split, save_csv

config = cfg()
TEST_SIZE = config["training"]["test_size"]
RANDOM_STATE = config["training"]["random_state"]
N_TRIALS = {"lda": 8, "mlp": 6, "xgboost": 8, "lightgbm": 8, "random_forest": 8}
STEP_MONTHS = 6
MIN_TRAIN_MONTHS = 12

STAGE2_FEATURE_NAMES = [
    "macro_VIX", "macro_VIX_zscore_50",
    "macro_Yield_Curve_10Y_2Y", "macro_Credit_Spread_log",
    "macro_Gold_Silver_Ratio", "macro_SP500_VIX_ratio",
    "macro_DXY_zscore_50", "macro_SP500_roc_20",
    "macro_FEDFUNDS", "macro_real_interest_rate", "macro_UNRATE",
]


def step_features(aligned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ohlcv = aligned[["Open", "High", "Low", "Close", "Volume"]]
    print("\n[F1] Technical features (osc + volat + vol + trend_following)")
    osc = compute_oscillator_indicators(ohlcv)
    volat = compute_volatility_indicators(ohlcv)
    volu = compute_volume_indicators(ohlcv)
    trend_follow = compute_trend_following_features(ohlcv)
    tech = pd.concat([osc, volat, volu, trend_follow], axis=1).dropna()
    print(f"  Stage 3 v2 features (concatenated): {tech.shape}")
    save_csv(tech, PROJECT_ROOT / "data" / "processed" / "btc_features_stage3_v2_bigger.csv")

    print("\n[F2] Macro features")
    macro = compute_macro_features(aligned)
    print(f"  macro: {macro.shape}")
    save_csv(macro, PROJECT_ROOT / "data" / "processed" / "btc_features_macro_bigger.csv")
    return tech, macro


def step_labels(aligned: pd.DataFrame) -> dict:
    print("\n[L1] Trend SMA labels")
    trend_sma = generate_trend_labels(aligned)
    print(f"  trend_sma: {trend_sma.shape},   class counts:\n{trend_sma.value_counts()}")

    print("\n[L2] Trend ZigZag labels")
    trend_zz = generate_trend_labels_zigzag(aligned["Close"], threshold=0.10, sideways_band=0.03)
    print(f"  trend_zz: {trend_zz.shape},   class counts:\n{trend_zz.value_counts()}")

    print("\n[L3] Signal labels (volatility-adaptive)")
    sig = generate_signal_labels(aligned["Close"], method="adaptive")
    print(f"  signals: {sig.shape},  class counts:\n{sig.value_counts()}")
    save_csv(sig, PROJECT_ROOT / "data" / "labels" / "btc_signal_labels_adaptive_bigger.csv")
    return {"trend_sma": trend_sma, "trend_zz": trend_zz, "y": sig}


def step_stage1_oof(tech_full: pd.DataFrame, trend_label: pd.Series, suffix: str) -> pd.DataFrame:
    print(f"\n[S1-{suffix}] Stage 1 LDA OOF (full {len(trend_label)} labels)")
    common = tech_full.index.intersection(trend_label.index)
    X = tech_full.loc[common]; y = trend_label.loc[common].dropna()
    common = X.index.intersection(y.index)
    X = X.loc[common]; y = y.loc[common]
    print(f"  X={X.shape}  y={y.shape}")

    X_tr, _ = chronological_train_test_split(X, test_size=TEST_SIZE)
    y_tr = y.loc[X_tr.index]
    res = tune_stage1(
        X_tr, y_tr, classifier_name="lda", n_trials=N_TRIALS["lda"],
        save_model=False, step_months=STEP_MONTHS, min_train_months=MIN_TRAIN_MONTHS,
    )
    full_model = res["model"]
    train_med = X_tr.median()
    proba = full_model.predict_proba(X.fillna(train_med))
    classes = list(full_model.classes_)
    df = pd.DataFrame(proba, index=X.index, columns=[f"P_{c}" for c in classes])
    save_csv(df, PROJECT_ROOT / "data" / "labels" / f"btc_stage1_oof_lda{suffix}_bigger.csv")
    return df


def step_stage2_oof(macro: pd.DataFrame) -> pd.DataFrame:
    print("\n[S2] Stage 2 GMM OOF (11-feature subset)")
    available = [c for c in STAGE2_FEATURE_NAMES if c in macro.columns]
    missing = [c for c in STAGE2_FEATURE_NAMES if c not in macro.columns]
    if missing:
        raise RuntimeError(f"missing Stage 2 features: {missing}")
    X = macro[available].dropna()
    print(f"  X.shape={X.shape}  {X.index.min().date()} → {X.index.max().date()}")
    X_tr, X_te = chronological_train_test_split(X, test_size=TEST_SIZE)
    oof_train, full_gmm, full_scaler = compute_oof_regime_posterior(
        X_tr, method="gmm", n_clusters=3, n_folds=5, random_state=RANDOM_STATE,
    )
    test_post = predict_regime_posterior(X_te, full_gmm, full_scaler, method="gmm", n_clusters=3)
    posterior = pd.concat([oof_train, test_post]).sort_index()
    save_csv(posterior, PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior_bigger.csv")
    print(f"  posterior shape: {posterior.shape}")
    return posterior


def step_stage3(
    tech: pd.DataFrame, y: pd.Series,
    s1_sma: pd.DataFrame, s1_zz: pd.DataFrame, s2: pd.DataFrame,
    aligned: pd.DataFrame,
) -> dict:
    print("\n[S3] Stage 3 retrain × 7 models")
    bt = Backtester()
    out = {"summary_ab": [], "summary_zz": [], "backtest": [], "test_signals": None,
           "test_signals_zz": None, "test_close": None, "test_close_zz": None}

    def split(s1):
        common = tech.index.intersection(y.index).intersection(s1.index).intersection(s2.index)
        X_tr, X_te = chronological_train_test_split(tech.loc[common], test_size=TEST_SIZE)
        return (X_tr, X_te,
                y.loc[X_tr.index], y.loc[X_te.index],
                s1.loc[X_tr.index], s1.loc[X_te.index],
                s2.loc[X_tr.index], s2.loc[X_te.index])

    print("\n  ---- Phase A+B (SMA Stage 1 OOF) ----")
    X_tr, X_te, y_tr, y_te, s1_tr, s1_te, s2_tr, s2_te = split(s1_sma)
    print(f"    Train={X_tr.shape}  Test={X_te.shape}  ({y_te.index.min().date()} → {y_te.index.max().date()})")
    train_med = pd.concat([X_tr, s1_tr, s2_tr], axis=1).median()
    test_close = aligned["Close"].loc[y_te.index]
    out["test_close"] = test_close
    test_signals = pd.DataFrame({"y_true": y_te.values}, index=y_te.index)

    for clf in ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]:
        print(f"\n    >>> Stage 3 {clf.upper()}")
        t0 = time.time()
        res = tune_stage3(
            X_tr, y_tr, s1_tr, s2_tr, classifier_name=clf,
            n_trials=N_TRIALS.get(clf, 6), save_model=False,
            step_months=STEP_MONTHS, min_train_months=MIN_TRAIN_MONTHS,
        )
        print(f"        WF f1={res['avg_f1_macro']:.4f}, {time.time()-t0:.1f}s")
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
            "strategy": f"v2-bigger {clf.upper()}",
            "total_return": bt_res["total_return"], "sharpe": bt_res["sharpe_ratio"],
            "max_dd": bt_res["max_drawdown"], "n_trades": bt_res["n_trades"],
            "win_rate": bt_res["win_rate"],
        })
    out["test_signals"] = test_signals

    print("\n  ---- Phase C (ZigZag Stage 1 OOF) ----")
    X_tr, X_te, y_tr, y_te, s1_tr, s1_te, s2_tr, s2_te = split(s1_zz)
    print(f"    Train={X_tr.shape}  Test={X_te.shape}")
    train_med_z = pd.concat([X_tr, s1_tr, s2_tr], axis=1).median()
    test_close_z = aligned["Close"].loc[y_te.index]
    out["test_close_zz"] = test_close_z
    test_signals_zz = pd.DataFrame({"y_true": y_te.values}, index=y_te.index)

    for clf in ["xgboost", "mlp"]:
        print(f"\n    >>> Stage 3 ZZ {clf.upper()}")
        t0 = time.time()
        res = tune_stage3(
            X_tr, y_tr, s1_tr, s2_tr, classifier_name=clf,
            n_trials=N_TRIALS.get(clf, 6), save_model=False,
            step_months=STEP_MONTHS, min_train_months=MIN_TRAIN_MONTHS,
        )
        print(f"        WF f1={res['avg_f1_macro']:.4f}, {time.time()-t0:.1f}s")
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
            "strategy": f"v2-bigger ZZ {clf.upper()}",
            "total_return": bt_res["total_return"], "sharpe": bt_res["sharpe_ratio"],
            "max_dd": bt_res["max_drawdown"], "n_trades": bt_res["n_trades"],
            "win_rate": bt_res["win_rate"],
        })
    out["test_signals_zz"] = test_signals_zz
    return out


def write_outputs(res: dict) -> None:
    print("\n[W] Write summaries")
    labels = PROJECT_ROOT / "data" / "labels"

    summary_ab = pd.DataFrame(res["summary_ab"]).round(4).set_index("model")
    print("\n=== Stage 3 v2-bigger Phase A+B test summary ===")
    print(summary_ab.to_string())
    save_csv(summary_ab, labels / "btc_stage3_bigger_full_summary.csv")
    save_csv(res["test_signals"], labels / "btc_test_signals_bigger.csv")

    summary_zz = pd.DataFrame(res["summary_zz"]).round(4).set_index("model")
    print("\n=== Stage 3 v2-bigger Phase C (ZigZag) test summary ===")
    print(summary_zz.to_string())
    save_csv(summary_zz, labels / "btc_stage3_bigger_zigzag_summary.csv")
    save_csv(res["test_signals_zz"], labels / "btc_test_signals_bigger_zigzag.csv")

    bt_df = pd.DataFrame(res["backtest"]).round(4)
    bt = Backtester()
    bh = bt.run_buy_and_hold(res["test_close"])
    bh_ab = {
        "strategy": "Buy & Hold (A+B)",
        "total_return": round(bh.iloc[-1]/bh.iloc[0] - 1, 4),
        "sharpe": round(res["test_close"].pct_change().mean()/res["test_close"].pct_change().std()*np.sqrt(252), 4),
        "max_dd": round(((bh-bh.cummax())/bh.cummax()).min(), 4),
        "n_trades": 1, "win_rate": np.nan,
    }
    bh_z = bt.run_buy_and_hold(res["test_close_zz"])
    bh_zz = {
        "strategy": "Buy & Hold (ZZ)",
        "total_return": round(bh_z.iloc[-1]/bh_z.iloc[0] - 1, 4),
        "sharpe": round(res["test_close_zz"].pct_change().mean()/res["test_close_zz"].pct_change().std()*np.sqrt(252), 4),
        "max_dd": round(((bh_z-bh_z.cummax())/bh_z.cummax()).min(), 4),
        "n_trades": 1, "win_rate": np.nan,
    }
    full_bt = pd.concat([bt_df, pd.DataFrame([bh_ab, bh_zz])], ignore_index=True)
    save_csv(full_bt, labels / "btc_backtest_bigger_summary.csv")
    print("\n=== Backtest comparison v2-bigger ===")
    print(full_bt.to_string(index=False))

    # Final aggregated summary
    rows = []
    for clf, lbl in [("LDA","LDA"),("MLP","MLP"),("XGBOOST","XGB"),
                     ("LIGHTGBM","LGBM"),("RANDOM_FOREST","RF")]:
        m = summary_ab.loc[clf]
        b = bt_df[bt_df["strategy"].eq(f"v2-bigger {clf}")].iloc[0]
        rows.append({"Model": lbl, "Test Acc": m["test_acc"], "Test F1": m["test_f1"],
                     "MCC": m["test_mcc"], "Return": f"{b['total_return']*100:+.1f}%",
                     "Sharpe": b["sharpe"], "MaxDD": f"{b['max_dd']*100:.1f}%",
                     "Trades": int(b["n_trades"]),
                     "Win%": (f"{b['win_rate']*100:.1f}%" if b["win_rate"] else "-")})
    for clf, lbl in [("XGBOOST","ZZ-XGB"), ("MLP","ZZ-MLP")]:
        m = summary_zz.loc[f"{clf}_zigzag"]
        b = bt_df[bt_df["strategy"].eq(f"v2-bigger ZZ {clf}")].iloc[0]
        rows.append({"Model": lbl, "Test Acc": m["test_acc"], "Test F1": m["test_f1"],
                     "MCC": m["test_mcc"], "Return": f"{b['total_return']*100:+.1f}%",
                     "Sharpe": b["sharpe"], "MaxDD": f"{b['max_dd']*100:.1f}%",
                     "Trades": int(b["n_trades"]),
                     "Win%": (f"{b['win_rate']*100:.1f}%" if b["win_rate"] else "-")})
    rows.append({"Model": "Buy & Hold", "Test Acc": "-", "Test F1": "-", "MCC": "-",
                 "Return": f"{bh_ab['total_return']*100:+.1f}%",
                 "Sharpe": bh_ab["sharpe"], "MaxDD": f"{bh_ab['max_dd']*100:.1f}%",
                 "Trades": 1, "Win%": "-"})
    final_df = pd.DataFrame(rows)
    print("\n=== final_iter5_summary_bigger.csv ===")
    print(final_df.to_string(index=False))
    save_csv(final_df, labels / "final_iter5_summary_bigger.csv")


def main() -> None:
    aligned = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned_v2.csv",
                          index_col=0, parse_dates=True)
    print(f"Loaded aligned_v2: {aligned.shape}, {aligned.index.min().date()} → {aligned.index.max().date()}")
    tech, macro = step_features(aligned)
    labels = step_labels(aligned)
    s1_sma_full = step_stage1_oof(tech, labels["trend_sma"], suffix="")
    s1_zz_full  = step_stage1_oof(tech, labels["trend_zz"],  suffix="_zigzag")
    s2 = step_stage2_oof(macro)
    res = step_stage3(tech, labels["y"], s1_sma_full, s1_zz_full, s2, aligned)
    write_outputs(res)
    print("\nv2-bigger retrain complete.")


if __name__ == "__main__":
    main()
