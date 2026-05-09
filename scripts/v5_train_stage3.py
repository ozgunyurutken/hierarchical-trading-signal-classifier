"""V5 Phase 4 — Stage 3 baseline walk-forward training (fixed HP).

Outputs:
  data/processed/{btc,eth}_stage3_oof_{model}_v5.csv
  reports/Phase4/v5_p4_stage3_metrics.csv
  reports/Phase4/v5_p4_stage3_overall.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.v5_stage3_features import STAGE3_FEATURE_COLS
from src.models.v5_stage3_trainer import (
    MODEL_FACTORIES, train_walk_forward, fold_results_to_df, overall_metrics,
)

ASSETS = ["btc", "eth"]
MODELS = list(MODEL_FACTORIES.keys())


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out_reports = PROJECT_ROOT / "reports" / "Phase4"
    out_reports.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    overall_rows = []

    for asset in ASSETS:
        df = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                         index_col=0, parse_dates=True)
        X = df[STAGE3_FEATURE_COLS]
        y = df["signal_label"]
        print(f"\n[{asset.upper()}] dataset: {X.shape}, "
              f"span {X.index.min().date()} -> {X.index.max().date()}")

        for model_name in MODELS:
            t0 = time.time()
            oof, folds = train_walk_forward(X, y, model_name, balanced=True)
            dt = time.time() - t0

            oof_path = proc / f"{asset}_stage3_oof_{model_name}_v5.csv"
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
                "f1_sell":  ov["f1_per_class"][0],
                "f1_hold":  ov["f1_per_class"][1],
                "f1_buy":   ov["f1_per_class"][2],
                "elapsed_s": round(dt, 1),
            })
            print(f"  {model_name:14s}  acc {ov['accuracy']:.4f}  F1m {ov['f1_macro']:.4f}  "
                  f"f1_sell {ov['f1_per_class'][0]:.3f}  f1_hold {ov['f1_per_class'][1]:.3f}  "
                  f"f1_buy {ov['f1_per_class'][2]:.3f}  ({dt:.1f}s)")

    pd.concat(metric_rows, ignore_index=True).to_csv(
        out_reports / "v5_p4_stage3_metrics.csv", index=False)
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(out_reports / "v5_p4_stage3_overall.csv", index=False)

    print(f"\n=== Stage 3 baseline overall ===")
    print(overall_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
