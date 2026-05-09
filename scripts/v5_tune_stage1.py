"""V5 Phase 3 — Optuna HP tuning for Stage 1 trend classifier.

Runs an Optuna study per (asset, model) over the inner WF-CV objective,
saves all trials + best params. Does NOT retrain the outer expanding-window
CV here — that lives in v5_train_stage1_tuned.py so this script is rerunnable
without overwriting outer OOF.

Outputs:
  reports/Phase3/v5_p3_stage1_optuna_studies.csv  (every trial, per (asset, model))
  reports/Phase3/v5_p3_stage1_optuna_best.csv     (8 rows: best params per (asset, model))

Usage:
  python scripts/v5_tune_stage1.py                      # full run (default 30 trials)
  python scripts/v5_tune_stage1.py --smoke              # smoke test (BTC XGB only, 5 trials)
  python scripts/v5_tune_stage1.py --n-trials 20        # override trial budget
  python scripts/v5_tune_stage1.py --models xgboost rf  # subset
  python scripts/v5_tune_stage1.py --assets btc         # subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.v5_trend_features import STAGE1_FEATURE_COLS
from src.models.v5_stage1_optuna import (
    InnerCVConfig, inner_walk_forward_splits, run_study, study_to_trials_df,
)


ALL_ASSETS = ["btc", "eth"]
ALL_MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]


def _load_dataset(asset: str) -> tuple[pd.DataFrame, pd.Series]:
    proc = PROJECT_ROOT / "data" / "processed"
    df = pd.read_csv(proc / f"{asset}_features_stage1_v5_zz.csv",
                     index_col=0, parse_dates=True)
    return df[STAGE1_FEATURE_COLS], df["trend_label"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=ALL_ASSETS,
                    choices=ALL_ASSETS, help="Subset of assets to tune.")
    ap.add_argument("--models", nargs="+", default=ALL_MODELS,
                    choices=ALL_MODELS, help="Subset of models to tune.")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Optuna trial budget per (asset, model).")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke mode: BTC XGB only, 5 trials, output to *_smoke.csv.")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix added to output filenames (default empty).")
    ap.add_argument("--storage", default="sqlite:///optuna_v5.db",
                    help="Optuna storage URL for live dashboard tracking. "
                         "Use 'none' for in-memory. Default sqlite:///optuna_v5.db.")
    args = ap.parse_args()

    if args.smoke:
        args.assets = ["btc"]
        args.models = ["xgboost"]
        args.n_trials = 5
        args.out_suffix = "_smoke"
        args.storage = "none"  # smoke runs are throwaway

    storage = None if args.storage.lower() == "none" else args.storage

    out_reports = PROJECT_ROOT / "reports" / "Phase3.5_after_tune"
    out_reports.mkdir(parents=True, exist_ok=True)

    inner_cfg = InnerCVConfig()  # train_min 750, val 300, step 300, gap 10

    print(f"[CONFIG] inner CV {inner_cfg}")
    print(f"[CONFIG] assets {args.assets} | models {args.models}")
    print(f"[CONFIG] n_trials {args.n_trials} per (asset, model) | balanced=True")
    print(f"[CONFIG] storage: {storage or 'in-memory'}")
    if storage:
        print(f"[CONFIG] live dashboard:  optuna-dashboard {storage}\n")
    else:
        print()

    all_trials: list[pd.DataFrame] = []
    best_rows: list[dict] = []

    for asset in args.assets:
        X, y = _load_dataset(asset)
        inner_folds = list(inner_walk_forward_splits(len(X), inner_cfg))
        fold_dates = [
            f"{X.index[v[0]].date()}->{X.index[v[-1]].date()}"
            for _, v in inner_folds
        ]
        print(f"[{asset.upper()}] dataset {X.shape}, "
              f"span {X.index.min().date()} -> {X.index.max().date()}, "
              f"{len(inner_folds)} inner WF folds")
        for i, fd in enumerate(fold_dates, 1):
            print(f"        inner fold {i}: val {fd}")

        for model_name in args.models:
            t0 = time.time()
            print(f"  >>> tuning {model_name:14s} ...", flush=True)
            study, best = run_study(
                asset=asset, model_name=model_name,
                X=X, y=y, n_trials=args.n_trials, balanced=True,
                inner_cfg=inner_cfg, storage=storage,
            )
            dt = time.time() - t0

            trials_df = study_to_trials_df(study, asset, model_name)
            all_trials.append(trials_df)

            best_rows.append({
                "asset":         asset,
                "model":         model_name,
                "best_value":    study.best_value,
                "best_trial":    study.best_trial.number,
                "n_trials_done": len([t for t in study.trials if t.value is not None]),
                "n_pruned":      len([t for t in study.trials if t.state.name == "PRUNED"]),
                "elapsed_s":     round(dt, 1),
                "best_params":   json.dumps(best, default=_json_default),
            })

            print(f"      best F1m {study.best_value:.4f}  "
                  f"({study.best_trial.number}/{args.n_trials} trial)  "
                  f"{dt:.1f}s")
            print(f"      params: {best}")

    trials_path = out_reports / f"v5_p3_stage1_optuna_studies{args.out_suffix}.csv"
    best_path = out_reports / f"v5_p3_stage1_optuna_best{args.out_suffix}.csv"

    pd.concat(all_trials, ignore_index=True).to_csv(trials_path, index=False)
    pd.DataFrame(best_rows).to_csv(best_path, index=False)

    print(f"\n=== Optuna tuning complete ===")
    print(f"trials  -> {trials_path.relative_to(PROJECT_ROOT)}")
    print(f"best    -> {best_path.relative_to(PROJECT_ROOT)}")
    print()
    print(pd.DataFrame(best_rows)[["asset", "model", "best_value",
                                    "best_trial", "n_pruned", "elapsed_s"]].to_string(index=False))

    return 0


def _json_default(o):
    """JSON serializer for tuple HPs (hidden_layer_sizes)."""
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Not serializable: {type(o)}")


if __name__ == "__main__":
    raise SystemExit(main())
