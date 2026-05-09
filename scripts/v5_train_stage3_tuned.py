"""V5 Phase 4 — Stage 3 retrain with tuned hyperparameters.

Reads:  reports/Phase4.5_after_tune/v5_p4_stage3_optuna_best.csv
Writes: data/processed/{btc,eth}_stage3_oof_{model}_v5_tuned.csv
        reports/Phase4.5_after_tune/v5_p4_stage3_metrics_tuned.csv
        reports/Phase4.5_after_tune/v5_p4_stage3_overall_tuned.csv
        reports/Phase4.5_after_tune/v5_p4_stage3_tuning_delta.csv
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

from src.features.v5_stage3_features import STAGE3_FEATURE_COLS
from src.models.v5_stage3_trainer import (
    MODEL_FACTORIES, train_walk_forward, fold_results_to_df, overall_metrics,
)


ASSETS = ["btc", "eth"]
MODELS = list(MODEL_FACTORIES.keys())


def _hp_from_row(row: pd.Series) -> dict:
    raw = json.loads(row["best_params"])
    if "hidden_layer_sizes" in raw and isinstance(raw["hidden_layer_sizes"], list):
        raw["hidden_layer_sizes"] = tuple(raw["hidden_layer_sizes"])
    return raw


def _load_baseline_overall() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "reports" / "Phase4" / "v5_p4_stage3_overall.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-csv", default=None)
    args = ap.parse_args()

    proc = PROJECT_ROOT / "data" / "processed"
    out_reports = PROJECT_ROOT / "reports" / "Phase4.5_after_tune"
    out_reports.mkdir(parents=True, exist_ok=True)
    best_csv = (Path(args.best_csv) if args.best_csv
                else out_reports / "v5_p4_stage3_optuna_best.csv")

    if not best_csv.exists():
        sys.exit(f"[ERROR] best-params CSV not found: {best_csv}\n"
                 f"        run scripts/v5_tune_stage3.py first.")

    best = pd.read_csv(best_csv)
    print(f"[CONFIG] best params loaded from {best_csv.relative_to(PROJECT_ROOT)}")
    print(best[["asset", "model", "best_value", "best_trial"]].to_string(index=False))
    print()

    metric_rows = []
    overall_rows = []

    for asset in ASSETS:
        df = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                         index_col=0, parse_dates=True)
        X = df[STAGE3_FEATURE_COLS]
        y = df["signal_label"]
        print(f"[{asset.upper()}] dataset {X.shape}")

        for model_name in MODELS:
            mask = (best["asset"] == asset) & (best["model"] == model_name)
            if not mask.any():
                continue
            hp = _hp_from_row(best[mask].iloc[0])

            t0 = time.time()
            oof, folds = train_walk_forward(X, y, model_name, balanced=True, hp=hp)
            dt = time.time() - t0

            oof_path = proc / f"{asset}_stage3_oof_{model_name}_v5_tuned.csv"
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
                  f"hold_F1 {ov['f1_per_class'][1]:.3f}  ({dt:.1f}s)")

    metrics_df = pd.concat(metric_rows, ignore_index=True)
    overall_df = pd.DataFrame(overall_rows)
    metrics_df.to_csv(out_reports / "v5_p4_stage3_metrics_tuned.csv", index=False)
    overall_df.to_csv(out_reports / "v5_p4_stage3_overall_tuned.csv", index=False)

    baseline = _load_baseline_overall()
    if baseline is not None:
        merged = overall_df.merge(baseline, on=["asset", "model"], suffixes=("_tuned", "_base"))
        merged["d_f1m"]    = merged["f1_macro_tuned"] - merged["f1_macro_base"]
        merged["d_f1_buy"] = merged["f1_buy_tuned"]   - merged["f1_buy_base"]
        merged["d_f1_hold"]= merged["f1_hold_tuned"]  - merged["f1_hold_base"]
        merged["d_f1_sell"]= merged["f1_sell_tuned"]  - merged["f1_sell_base"]
        merged["d_acc"]    = merged["accuracy_tuned"] - merged["accuracy_base"]
        delta = merged[["asset", "model",
                        "f1_macro_base", "f1_macro_tuned", "d_f1m",
                        "f1_buy_base",   "f1_buy_tuned",   "d_f1_buy",
                        "f1_hold_base",  "f1_hold_tuned",  "d_f1_hold",
                        "f1_sell_base",  "f1_sell_tuned",  "d_f1_sell",
                        "accuracy_base", "accuracy_tuned", "d_acc"]]
        delta.to_csv(out_reports / "v5_p4_stage3_tuning_delta.csv", index=False)
        print(f"\n=== Stage 3 tuning delta (post − pre) ===")
        print(delta.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print(f"\n=== Stage 3 tuned overall ===")
    print(overall_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
