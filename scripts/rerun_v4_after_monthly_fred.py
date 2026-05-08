"""
v4 rerun: Stage 2 GMM is now 11-feature (was 8) — adding FEDFUNDS,
real_interest_rate (Fisher), UNRATE — and Stage 3 retrains on the new
posterior. Joblib model artefacts ARE saved this time (closes the demo
training-serving skew left over from v3).

Pipeline:
  1. Reuse already-recomputed btc_features_macro.csv (187 cols, post-FRED)
  2. Stage 2 OOF posterior with 11-feature subset → btc_oof_regime_posterior.csv
  3. Stage 3 retrain × 7 models, save_model=True → app/models/stage3_*_v2.joblib
  4. Backtest test predictions + write summary CSVs (override v3)
  5. Compare v3 → v4 cluster shift (ARI), backtest deltas
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
from sklearn.metrics import adjusted_rand_score

from src.evaluation.backtester import Backtester
from src.evaluation.metrics import compute_all_metrics
from src.labels.regime_labels import compute_oof_regime_posterior, predict_regime_posterior
from src.models.stage3_trainer import tune_stage3
from src.utils.config import cfg
from src.utils.helpers import chronological_train_test_split, save_csv

config = cfg()
TEST_SIZE = config["training"]["test_size"]
RANDOM_STATE = config["training"]["random_state"]
STEP_MONTHS = 6
MIN_TRAIN_MONTHS = 12
N_TRIALS = {"lda": 8, "mlp": 6, "xgboost": 8, "lightgbm": 8, "random_forest": 8}

# 8 → 11 feature subset for Stage 2 GMM
STAGE2_FEATURE_NAMES_V4 = [
    # Original 8
    "macro_VIX",
    "macro_VIX_zscore_50",
    "macro_Yield_Curve_10Y_2Y",
    "macro_Credit_Spread_log",
    "macro_Gold_Silver_Ratio",
    "macro_SP500_VIX_ratio",
    "macro_DXY_zscore_50",
    "macro_SP500_roc_20",
    # New 3 (monthly FRED)
    "macro_FEDFUNDS",
    "macro_real_interest_rate",
    "macro_UNRATE",
]


def backup(path: Path) -> None:
    if not path.exists():
        return
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    shutil.copy(path, dst)
    print(f"  Backup: {dst.name}")


def stage2_regen() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (new_posterior, old_posterior_aligned_for_comparison)."""
    print("\n[Stage 2] Build 11-feature subset, OOF GMM regen")
    macro_feats = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_features_macro.csv",
                              index_col=0, parse_dates=True)
    available = [c for c in STAGE2_FEATURE_NAMES_V4 if c in macro_feats.columns]
    missing = [c for c in STAGE2_FEATURE_NAMES_V4 if c not in macro_feats.columns]
    if missing:
        raise RuntimeError(f"Missing Stage 2 features: {missing}")
    print(f"  features ({len(available)}): {available}")

    X = macro_feats[available].dropna()
    print(f"  X shape after dropna: {X.shape}, "
          f"{X.index.min().date()} → {X.index.max().date()}")

    X_tr, X_te = chronological_train_test_split(X, test_size=TEST_SIZE)
    print(f"  Train={X_tr.shape}  Test={X_te.shape}")

    oof_train, full_gmm, full_scaler = compute_oof_regime_posterior(
        X_tr, method="gmm", n_clusters=3, n_folds=5, random_state=RANDOM_STATE,
    )
    test_post = predict_regime_posterior(X_te, full_gmm, full_scaler, method="gmm", n_clusters=3)
    new_posterior = pd.concat([oof_train, test_post]).sort_index()
    print(f"  New posterior: {new_posterior.shape}")

    old_path = PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv"
    old = pd.read_csv(old_path, index_col=0, parse_dates=True)
    common = new_posterior.index.intersection(old.index)
    old_a = old.loc[common]; new_a = new_posterior.loc[common]
    ari = adjusted_rand_score(old_a.values.argmax(1), new_a.values.argmax(1))
    pct = (old_a.values.argmax(1) == new_a.values.argmax(1)).mean()
    l2 = np.linalg.norm(old_a.values - new_a.values, axis=1)
    print(f"  v3 → v4 ARI={ari:.3f}, hard-label agreement={pct:.1%}, "
          f"soft L2 mean={l2.mean():.3f}")

    backup(old_path)
    save_csv(new_posterior, old_path)
    return new_posterior, old_a


def stage3_retrain(s2_new: pd.DataFrame) -> dict:
    print("\n[Stage 3] Retrain 7 models on new s2_oof, save_model=True")
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
    s2 = s2_new

    bt = Backtester()
    out = {"summary_ab": [], "summary_zz": [], "backtest": []}

    def split_train_test(s1):
        common = stage3_v2.index.intersection(y_v2.index).intersection(s1.index).intersection(s2.index)
        X_tr, X_te = chronological_train_test_split(stage3_v2.loc[common], test_size=TEST_SIZE)
        return (
            X_tr, X_te,
            y_v2.loc[X_tr.index], y_v2.loc[X_te.index],
            s1.loc[X_tr.index], s1.loc[X_te.index],
            s2.loc[X_tr.index], s2.loc[X_te.index],
        )

    # --- Phase A + B (SMA Stage 1 OOF) ---
    print("\n  ---- Phase A+B (SMA Stage 1 OOF) ----")
    X_tr, X_te, y_tr, y_te, s1_tr, s1_te, s2_tr, s2_te = split_train_test(s1_sma)
    print(f"  Train={X_tr.shape}  Test={X_te.shape}")

    train_med = pd.concat([X_tr, s1_tr, s2_tr], axis=1).median()
    test_close = btc["Close"].loc[y_te.index]
    test_signals = pd.DataFrame({"y_true": y_te.values}, index=y_te.index)

    models_dir = PROJECT_ROOT / "app" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for clf_name in ["lda", "mlp", "xgboost", "lightgbm", "random_forest"]:
        print(f"\n    >>> Stage 3 {clf_name.upper()}")
        t0 = time.time()
        res = tune_stage3(
            X_tr, y_tr, s1_tr, s2_tr,
            classifier_name=clf_name,
            n_trials=N_TRIALS.get(clf_name, 6),
            save_model=False,
            step_months=STEP_MONTHS, min_train_months=MIN_TRAIN_MONTHS,
        )
        # Save under explicit v2-suffixed path (stage3_trainer's built-in
        # save would overwrite the unversioned MVP joblib).
        import joblib as _joblib
        v2_path = models_dir / f"stage3_{clf_name}_v2.joblib"
        _joblib.dump(res["model"], v2_path)
        print(f"        WF f1={res['avg_f1_macro']:.4f}, {time.time()-t0:.1f}s, saved={v2_path.name}")
        Xc = pd.concat([X_te, s1_te, s2_te], axis=1).fillna(train_med)
        pred = res["model"].predict(Xc); proba = res["model"].predict_proba(Xc)
        m = compute_all_metrics(y_te.values, pred, y_proba=proba, classes=list(res["model"].classes_))
        test_signals[f"{clf_name}_pred"] = pred
        for i, c in enumerate(res["model"].classes_):
            test_signals[f"{clf_name}_proba_{c}"] = proba[:, i]
        out["summary_ab"].append({
            "model": clf_name.upper(), "WF_f1": res["avg_f1_macro"],
            "test_acc": m["accuracy"], "test_f1": m["f1_macro"],
            "test_balanced_acc": m["balanced_accuracy"], "test_mcc": m["mcc"],
        })
        bt_res = bt.run(pd.Series(pred, index=y_te.index), test_close)
        out["backtest"].append({
            "strategy": f"v4 {clf_name.upper()}",
            "total_return": bt_res["total_return"], "sharpe": bt_res["sharpe_ratio"],
            "max_dd": bt_res["max_drawdown"], "n_trades": bt_res["n_trades"],
            "win_rate": bt_res["win_rate"],
        })

    bh = bt.run_buy_and_hold(test_close)
    out["bh_ab"] = {
        "total_return": float(bh.iloc[-1]/bh.iloc[0] - 1),
        "sharpe": float(test_close.pct_change().mean() / test_close.pct_change().std() * np.sqrt(252)),
        "max_dd": float(((bh - bh.cummax())/bh.cummax()).min()),
    }
    out["test_signals_ab"] = test_signals
    out["test_close_ab"] = test_close

    # --- Phase C (ZigZag Stage 1 OOF) ---
    print("\n  ---- Phase C (ZigZag Stage 1 OOF) ----")
    X_tr, X_te, y_tr, y_te, s1_tr, s1_te, s2_tr, s2_te = split_train_test(s1_zz)
    print(f"  Train={X_tr.shape}  Test={X_te.shape}")
    train_med_z = pd.concat([X_tr, s1_tr, s2_tr], axis=1).median()
    test_close_z = btc["Close"].loc[y_te.index]
    test_signals_zz = pd.DataFrame({"y_true": y_te.values}, index=y_te.index)

    for clf_name in ["xgboost", "mlp"]:
        print(f"\n    >>> Stage 3 ZZ {clf_name.upper()}")
        t0 = time.time()
        res = tune_stage3(
            X_tr, y_tr, s1_tr, s2_tr,
            classifier_name=clf_name,
            n_trials=N_TRIALS.get(clf_name, 6),
            save_model=False,
            step_months=STEP_MONTHS, min_train_months=MIN_TRAIN_MONTHS,
        )
        import joblib as _joblib
        v2_path = models_dir / f"stage3_{clf_name}_v2_zigzag.joblib"
        _joblib.dump(res["model"], v2_path)
        print(f"        WF f1={res['avg_f1_macro']:.4f}, {time.time()-t0:.1f}s, saved={v2_path.name}")
        Xc = pd.concat([X_te, s1_te, s2_te], axis=1).fillna(train_med_z)
        pred = res["model"].predict(Xc); proba = res["model"].predict_proba(Xc)
        m = compute_all_metrics(y_te.values, pred, y_proba=proba, classes=list(res["model"].classes_))
        test_signals_zz[f"{clf_name}_pred"] = pred
        for i, c in enumerate(res["model"].classes_):
            test_signals_zz[f"{clf_name}_proba_{c}"] = proba[:, i]
        out["summary_zz"].append({
            "model": f"{clf_name.upper()}_zigzag", "WF_f1": res["avg_f1_macro"],
            "test_acc": m["accuracy"], "test_f1": m["f1_macro"],
            "test_balanced_acc": m["balanced_accuracy"], "test_mcc": m["mcc"],
        })
        bt_res = bt.run(pd.Series(pred, index=y_te.index), test_close_z)
        out["backtest"].append({
            "strategy": f"v4 ZZ {clf_name.upper()}",
            "total_return": bt_res["total_return"], "sharpe": bt_res["sharpe_ratio"],
            "max_dd": bt_res["max_drawdown"], "n_trades": bt_res["n_trades"],
            "win_rate": bt_res["win_rate"],
        })

    bh_z = bt.run_buy_and_hold(test_close_z)
    out["bh_zz"] = {
        "total_return": float(bh_z.iloc[-1]/bh_z.iloc[0] - 1),
        "sharpe": float(test_close_z.pct_change().mean() / test_close_z.pct_change().std() * np.sqrt(252)),
        "max_dd": float(((bh_z - bh_z.cummax())/bh_z.cummax()).min()),
    }
    out["test_signals_zz"] = test_signals_zz
    out["test_close_zz"] = test_close_z
    return out


def write_outputs(res: dict) -> None:
    print("\n[Write] CSV summaries (v4 → override v3)")
    labels = PROJECT_ROOT / "data" / "labels"
    for name in [
        "btc_stage3_v2_summary.csv", "btc_stage3_v2_full_summary.csv",
        "btc_stage3_v2_zigzag_summary.csv",
        "btc_test_signals_v2.csv", "btc_test_signals_v2_phase_b.csv",
        "btc_test_signals_v2_zigzag.csv",
        "btc_backtest_v2_summary.csv", "btc_backtest_v2_phase_b_summary.csv",
        "btc_backtest_v2_zigzag_summary.csv",
        "final_iter2_summary_table.csv",
    ]:
        backup(labels / name)

    summary_ab = pd.DataFrame(res["summary_ab"]).round(4).set_index("model")
    print("\n=== Stage 3 v4 Phase A+B test summary ===")
    print(summary_ab.to_string())
    save_csv(summary_ab.iloc[:2], labels / "btc_stage3_v2_summary.csv")
    save_csv(summary_ab, labels / "btc_stage3_v2_full_summary.csv")

    test_sig = res["test_signals_ab"]
    save_csv(test_sig, labels / "btc_test_signals_v2.csv")
    pb = test_sig[["y_true"] + [c for c in test_sig.columns
                                if any(c.startswith(k) for k in ["xgboost", "lightgbm", "random_forest"])]]
    save_csv(pb, labels / "btc_test_signals_v2_phase_b.csv")

    summary_zz = pd.DataFrame(res["summary_zz"]).round(4).set_index("model")
    print("\n=== Stage 3 v4 Phase C (ZigZag) test summary ===")
    print(summary_zz.to_string())
    save_csv(summary_zz, labels / "btc_stage3_v2_zigzag_summary.csv")
    save_csv(res["test_signals_zz"], labels / "btc_test_signals_v2_zigzag.csv")

    # Backtest CSVs
    bt_df = pd.DataFrame(res["backtest"]).round(4)
    bh_ab = res["bh_ab"]; bh_zz = res["bh_zz"]
    bh_row_ab = {"strategy": "Buy & Hold", "total_return": round(bh_ab["total_return"], 4),
                 "sharpe": round(bh_ab["sharpe"], 4), "max_dd": round(bh_ab["max_dd"], 4),
                 "n_trades": 1, "win_rate": np.nan}
    bh_row_zz = {"strategy": "Buy & Hold", "total_return": round(bh_zz["total_return"], 4),
                 "sharpe": round(bh_zz["sharpe"], 4), "max_dd": round(bh_zz["max_dd"], 4),
                 "n_trades": 1, "win_rate": np.nan}

    pa = bt_df[bt_df["strategy"].str.contains("LDA|MLP", regex=True)]
    pa = pa[~pa["strategy"].str.contains("ZZ")]
    pb = bt_df[bt_df["strategy"].str.contains("XGBOOST|LIGHTGBM|RANDOM_FOREST", regex=True)]
    pb = pb[~pb["strategy"].str.contains("ZZ")]
    pc = bt_df[bt_df["strategy"].str.contains("ZZ")]
    pc = pc.rename(columns={"sharpe": "sharpe_ratio", "max_dd": "max_drawdown"})

    save_csv(pd.concat([pa, pd.DataFrame([bh_row_ab])]), labels / "btc_backtest_v2_summary.csv")
    save_csv(pd.concat([pb, pd.DataFrame([bh_row_ab])]), labels / "btc_backtest_v2_phase_b_summary.csv")
    bh_zz_renamed = {**bh_row_zz, "sharpe_ratio": bh_row_zz.pop("sharpe"),
                     "max_drawdown": bh_row_zz.pop("max_dd")}
    save_csv(pd.concat([pc, pd.DataFrame([bh_zz_renamed])]), labels / "btc_backtest_v2_zigzag_summary.csv")

    print("\n=== Backtest comparison ===")
    print(bt_df.to_string(index=False))
    print(f"  Buy & Hold (Phase A/B test): return={bh_ab['total_return']*100:+.1f}%  Sharpe={bh_ab['sharpe']:.2f}")
    print(f"  Buy & Hold (Phase C   test): return={bh_zz['total_return']*100:+.1f}%  Sharpe={bh_zz['sharpe']:.2f}")

    # Final aggregated summary
    rows = []
    for clf, lbl in [("lda","LDA"),("mlp","MLP"),("xgboost","XGB"),
                     ("lightgbm","LGBM"),("random_forest","RF")]:
        m = summary_ab.loc[clf.upper()]
        b = bt_df[bt_df["strategy"].eq(f"v4 {clf.upper()}")].iloc[0]
        rows.append({"Model": lbl, "Test Acc": m["test_acc"], "Test F1": m["test_f1"],
                     "MCC": m["test_mcc"], "Return": f"{b['total_return']*100:+.1f}%",
                     "Sharpe": b["sharpe"], "MaxDD": f"{b['max_dd']*100:.1f}%",
                     "Trades": int(b["n_trades"]),
                     "Win%": (f"{b['win_rate']*100:.1f}%" if b["win_rate"] else "-")})
    for clf, lbl in [("xgboost","ZZ-XGB"), ("mlp","ZZ-MLP")]:
        m = summary_zz.loc[f"{clf.upper()}_zigzag"]
        b = bt_df[bt_df["strategy"].eq(f"v4 ZZ {clf.upper()}")].iloc[0]
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
    print("\n=== final_iter2_summary_table.csv (v4) ===")
    print(final_df.to_string(index=False))
    save_csv(final_df, labels / "final_iter2_summary_table.csv")


def main() -> None:
    s2_new, _ = stage2_regen()
    res = stage3_retrain(s2_new)
    write_outputs(res)
    print("\nv4 retrain complete.")


if __name__ == "__main__":
    main()
