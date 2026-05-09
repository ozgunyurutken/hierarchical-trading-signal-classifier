"""V5 Phase 3 — Stage 1 walk-forward training: 4 classifiers x 2 assets.

Outputs:
  data/processed/{btc,eth}_stage1_oof_{model}_v5.csv  — OOF predictions per (asset, model)
  reports/Phase3/v5_p3_stage1_metrics.csv             — per-fold metrics summary
  reports/Phase3/v5_p3_stage1_overall.csv             — overall metrics summary

Models: xgboost, lightgbm, random_forest, mlp (fixed hyperparameters).
"""
from __future__ import annotations

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


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    out_reports = PROJECT_ROOT / "reports" / "Phase3"
    out_reports.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    overall_rows = []

    for asset in ASSETS:
        df = pd.read_csv(proc / f"{asset}_features_stage1_v5_zz.csv",
                         index_col=0, parse_dates=True)
        X = df[STAGE1_FEATURE_COLS]
        y = df["trend_label"]
        print(f"\n[{asset.upper()}] dataset: {X.shape}, span {X.index.min().date()} -> {X.index.max().date()}")

        for model_name in MODELS:
            t0 = time.time()
            oof, folds = train_walk_forward(X, y, model_name)
            dt = time.time() - t0

            oof_path = proc / f"{asset}_stage1_oof_{model_name}_v5.csv"
            oof.to_csv(oof_path)

            fold_df = fold_results_to_df(folds)
            fold_df["asset"] = asset
            fold_df["model"] = model_name
            metric_rows.append(fold_df)

            ov = overall_metrics(oof)
            overall_rows.append({
                "asset":    asset,
                "model":    model_name,
                "n_folds":  len(folds),
                "n_oof":    ov["n"],
                "accuracy": ov["accuracy"],
                "f1_macro": ov["f1_macro"],
                "f1_downtrend": ov["f1_per_class"][0],
                "f1_range":     ov["f1_per_class"][1],
                "f1_uptrend":   ov["f1_per_class"][2],
                "elapsed_s": round(dt, 1),
            })
            print(f"  {model_name:14s}  acc {ov['accuracy']:.4f}  F1m {ov['f1_macro']:.4f}  "
                  f"({dt:.1f}s)  -> {oof_path.relative_to(PROJECT_ROOT)}")

    metrics_df = pd.concat(metric_rows, ignore_index=True)
    overall_df = pd.DataFrame(overall_rows)

    metrics_df.to_csv(out_reports / "v5_p3_stage1_metrics.csv", index=False)
    overall_df.to_csv(out_reports / "v5_p3_stage1_overall.csv", index=False)
    print(f"\n=== Overall summary ===")
    print(overall_df.to_string(index=False))


if __name__ == "__main__":
    main()
