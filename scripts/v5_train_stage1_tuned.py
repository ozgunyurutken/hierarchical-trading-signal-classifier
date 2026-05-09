"""V5 Phase 3 — Stage 1 retrain with tuned hyperparameters.

Reads best params from `reports/Phase3/v5_p3_stage1_optuna_best.csv`,
runs full outer expanding-window walk-forward CV using tuned models,
writes tuned OOF + per-fold metrics. Existing `_v5.csv` files (untuned
baseline) are left untouched for ablation comparison.

Outputs (suffix `_tuned`):
  data/processed/{btc,eth}_stage1_oof_{model}_v5_tuned.csv
  reports/Phase3.5_after_tune/v5_p3_stage1_metrics_tuned.csv
  reports/Phase3.5_after_tune/v5_p3_stage1_overall_tuned.csv
  reports/Phase3.5_after_tune/v5_p3_stage1_tuning_delta.csv  (pre vs post F1m, range F1)

Usage:
  python scripts/v5_train_stage1_tuned.py
  python scripts/v5_train_stage1_tuned.py --best-csv <path>  # custom best-params file
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
from src.models.v5_stage1_trainer import (
    MODEL_FACTORIES, train_walk_forward, fold_results_to_df, overall_metrics,
)


ASSETS = ["btc", "eth"]
MODELS = list(MODEL_FACTORIES.keys())


def _hp_from_row(row: pd.Series) -> dict:
    raw = json.loads(row["best_params"])
    if "hidden_layer_sizes" in raw and isinstance(raw["hidden_layer_sizes"], list):
        raw["hidden_layer_sizes"] = tuple(raw["hidden_layer_sizes"])
    return raw


def _load_baseline_overall(_reports_dir: Path) -> pd.DataFrame | None:
    """Baseline always lives in reports/Phase3 (untouched by tuning runs)."""
    p = PROJECT_ROOT / "reports" / "Phase3" / "v5_p3_stage1_overall.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--best-csv",
        default=None,
        help="Path to optuna best-params CSV (defaults to reports/Phase3/v5_p3_stage1_optuna_best.csv).",
    )
    args = ap.parse_args()

    proc = PROJECT_ROOT / "data" / "processed"
    out_reports = PROJECT_ROOT / "reports" / "Phase3.5_after_tune"
    out_reports.mkdir(parents=True, exist_ok=True)
    best_csv = (Path(args.best_csv) if args.best_csv
                else out_reports / "v5_p3_stage1_optuna_best.csv")

    if not best_csv.exists():
        sys.exit(f"[ERROR] best-params CSV not found: {best_csv}\n"
                 f"        run scripts/v5_tune_stage1.py first.")

    best = pd.read_csv(best_csv)
    print(f"[CONFIG] best params loaded from {best_csv.relative_to(PROJECT_ROOT)}")
    print(best[["asset", "model", "best_value", "best_trial"]].to_string(index=False))
    print()

    metric_rows = []
    overall_rows = []

    for asset in ASSETS:
        df = pd.read_csv(proc / f"{asset}_features_stage1_v5_zz.csv",
                         index_col=0, parse_dates=True)
        X = df[STAGE1_FEATURE_COLS]
        y = df["trend_label"]
        print(f"[{asset.upper()}] dataset {X.shape}, "
              f"span {X.index.min().date()} -> {X.index.max().date()}")

        for model_name in MODELS:
            mask = (best["asset"] == asset) & (best["model"] == model_name)
            if not mask.any():
                print(f"  ! skip {model_name}: no tuned params for ({asset}, {model_name})")
                continue
            hp = _hp_from_row(best[mask].iloc[0])

            t0 = time.time()
            oof, folds = train_walk_forward(X, y, model_name, balanced=True, hp=hp)
            dt = time.time() - t0

            oof_path = proc / f"{asset}_stage1_oof_{model_name}_v5_tuned.csv"
            oof.to_csv(oof_path)

            fold_df = fold_results_to_df(folds)
            fold_df["asset"] = asset
            fold_df["model"] = model_name
            metric_rows.append(fold_df)

            ov = overall_metrics(oof)
            overall_rows.append({
                "asset":        asset,
                "model":        model_name,
                "n_folds":      len(folds),
                "n_oof":        ov["n"],
                "accuracy":     ov["accuracy"],
                "f1_macro":     ov["f1_macro"],
                "f1_downtrend": ov["f1_per_class"][0],
                "f1_range":     ov["f1_per_class"][1],
                "f1_uptrend":   ov["f1_per_class"][2],
                "elapsed_s":    round(dt, 1),
            })
            print(f"  {model_name:14s}  acc {ov['accuracy']:.4f}  F1m {ov['f1_macro']:.4f}  "
                  f"range_F1 {ov['f1_per_class'][1]:.3f}  ({dt:.1f}s) -> "
                  f"{oof_path.relative_to(PROJECT_ROOT)}")

    metrics_df = pd.concat(metric_rows, ignore_index=True)
    overall_df = pd.DataFrame(overall_rows)

    metrics_df.to_csv(out_reports / "v5_p3_stage1_metrics_tuned.csv", index=False)
    overall_df.to_csv(out_reports / "v5_p3_stage1_overall_tuned.csv", index=False)

    # Compute delta vs baseline (if baseline file exists)
    baseline = _load_baseline_overall(out_reports)
    if baseline is not None:
        merged = overall_df.merge(
            baseline,
            on=["asset", "model"], suffixes=("_tuned", "_base"),
        )
        merged["d_f1m"]      = merged["f1_macro_tuned"]   - merged["f1_macro_base"]
        merged["d_f1_range"] = merged["f1_range_tuned"]   - merged["f1_range_base"]
        merged["d_acc"]      = merged["accuracy_tuned"]   - merged["accuracy_base"]
        delta = merged[["asset", "model",
                        "f1_macro_base", "f1_macro_tuned", "d_f1m",
                        "f1_range_base", "f1_range_tuned", "d_f1_range",
                        "accuracy_base", "accuracy_tuned", "d_acc"]]
        delta.to_csv(out_reports / "v5_p3_stage1_tuning_delta.csv", index=False)
        print(f"\n=== Tuning delta (post − pre) ===")
        print(delta.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print(f"\n=== Tuned overall ===")
    print(overall_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
