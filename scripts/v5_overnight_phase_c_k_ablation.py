"""V5 Overnight Phase C — Signal label k-threshold ablation.

Tests different volatility multipliers k for signal label generation.
k controls Buy/Sell vs Hold class balance:
  k=0.4  -> Hold ~30%   (more conservative, less noisy classes)
  k=0.5  -> Hold ~22%   (current V5_PLAN)
  k=0.7  -> Hold ~40%
  k=1.0  -> Hold ~60%   (very conservative)

For each k:
  - Rebuild signal labels
  - Use existing 16-feature dataset (Stage 1/2 + oscillator unchanged)
  - Train tuned models (no Optuna re-tune for speed; reuse 5-fold tuned HP)
  - Backtest 3 rules
  - Compare with k=0.5 baseline

Outputs:
  reports/Phase4.6_k_ablation/v5_p4_k_ablation_overall.csv
  reports/Phase4.6_k_ablation/v5_p4_k_ablation_backtest.csv
  reports/Phase4.6_k_ablation/v5_p4_k_ablation_label_dist.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.v5_backtester import (
    backtest_stateful, backtest_defensive, backtest_prob_weighted,
    backtest_buy_and_hold,
)
from src.features.v5_stage3_features import STAGE3_FEATURE_COLS
from src.labels.v5_signal_labels import generate_v5_signal_labels, label_distribution
from src.models.v5_stage3_trainer import train_walk_forward, overall_metrics


ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
K_VALUES = [0.4, 0.5, 0.7, 1.0]


def _load_best_params() -> dict:
    """Load tuned best params from Phase 4.5 for reuse."""
    p = PROJECT_ROOT / "reports" / "Phase4.5_after_tune" / "v5_p4_stage3_optuna_best.csv"
    df = pd.read_csv(p)
    out = {}
    for _, row in df.iterrows():
        hp = json.loads(row["best_params"])
        if isinstance(hp.get("hidden_layer_sizes"), list):
            hp["hidden_layer_sizes"] = tuple(hp["hidden_layer_sizes"])
        out[(row["asset"], row["model"])] = hp
    return out


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out = PROJECT_ROOT / "reports" / "Phase4.6_k_ablation"
    out.mkdir(parents=True, exist_ok=True)

    best_params = _load_best_params()

    label_dist_rows = []
    overall_rows = []
    backtest_rows = []

    for asset in ASSETS:
        full = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                           index_col=0, parse_dates=True)
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        prices = ohlcv["Close"]

        X_features = full[STAGE3_FEATURE_COLS]

        for k in K_VALUES:
            t_k = time.time()
            new_labels = generate_v5_signal_labels(prices, h=5, k=k, window=20)
            joined = X_features.join(new_labels[["signal_label"]], how="inner").dropna()

            X = joined[STAGE3_FEATURE_COLS]
            y = joined["signal_label"]

            dist = label_distribution(y)
            for cls, row in dist.iterrows():
                label_dist_rows.append({
                    "asset": asset, "k": k,
                    "class": cls, "count": int(row["count"]),
                    "share": float(row["share"]),
                })

            print(f"\n[{asset.upper()}/k={k}] n={len(X)}  "
                  f"Buy={dist.loc['Buy','share']:.1%}  "
                  f"Hold={dist.loc['Hold','share']:.1%}  "
                  f"Sell={dist.loc['Sell','share']:.1%}", flush=True)

            for model_name in MODELS:
                hp = best_params[(asset, model_name)]

                oof, _ = train_walk_forward(X, y, model_name, balanced=True, hp=hp)

                ov = overall_metrics(oof)
                overall_rows.append({
                    "asset": asset, "k": k, "model": model_name,
                    "n_oof": ov["n"],
                    "accuracy": ov["accuracy"], "f1_macro": ov["f1_macro"],
                    "f1_sell": ov["f1_per_class"][0],
                    "f1_hold": ov["f1_per_class"][1],
                    "f1_buy":  ov["f1_per_class"][2],
                })

                for rule_name, fn in [
                    ("stateful",      backtest_stateful),
                    ("defensive",     backtest_defensive),
                    ("prob_weighted", backtest_prob_weighted),
                ]:
                    res, _ = fn(oof, prices, asset=asset, model=model_name)
                    row = res.to_dict()
                    row["k"] = k
                    backtest_rows.append(row)

                print(f"  k={k} {model_name:14s} F1m {ov['f1_macro']:.3f}  "
                      f"hold_F1 {ov['f1_per_class'][1]:.3f}", flush=True)

            # B&H per (asset, k) — same span as OOF
            first_oof_span_start = oof.index.min()
            first_oof_span_end = oof.index.max()
            bh_res, _ = backtest_buy_and_hold(prices, first_oof_span_start,
                                              first_oof_span_end, asset=asset)
            row = bh_res.to_dict(); row["k"] = k
            backtest_rows.append(row)

            print(f"  k={k} done in {time.time()-t_k:.0f}s", flush=True)

    pd.DataFrame(label_dist_rows).to_csv(out / "v5_p4_k_ablation_label_dist.csv", index=False)
    pd.DataFrame(overall_rows).to_csv(out / "v5_p4_k_ablation_overall.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(out / "v5_p4_k_ablation_backtest.csv", index=False)

    print(f"\n=== Phase C complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
