"""V5 Overnight Phase A — Architecture Ablation (V5_PLAN Phase 5 spec).

Compares 4 architectures using same Optuna budget per (arch, asset, model):

  Flat            : 6 oscillator features only
  2-Stage Trend   : + 6 Stage 1 prob (raw + smoothed10d)         = 12
  2-Stage Macro   : + 4 Stage 2 (3 hard one-hot + regime_age)    = 10
  3-Stage Full    : 16 features (current default)

For each (arch, asset):
  - Optuna 5-fold inner CV, 30 trial per model (4 models)
  - Tuned outer retrain
  - Backtest 3 trading rules + B&H

Outputs:
  data/processed/{asset}_stage3_oof_{model}_v5_tuned_{arch}.csv
  reports/Phase5.1_arch_ablation/v5_p5_arch_optuna_best.csv
  reports/Phase5.1_arch_ablation/v5_p5_arch_overall.csv
  reports/Phase5.1_arch_ablation/v5_p5_arch_backtest_summary.csv
  reports/Phase5.1_arch_ablation/v5_p5_arch_equity_curves_{asset}.csv
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
from src.models.v5_stage3_optuna import (
    InnerCVConfig, run_study,
)
from src.models.v5_stage3_trainer import (
    train_walk_forward, fold_results_to_df, overall_metrics, MODEL_FACTORIES,
)


ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]

OSC_COLS = ["RSI_14", "MACD_signal_diff", "Bollinger_pct_b",
            "Stochastic_K_14", "volume_zscore_20", "OBV_change_20d"]

S1_COLS = ["P1_down", "P1_range", "P1_up",
           "P1_down_smooth10", "P1_range_smooth10", "P1_up_smooth10"]

S2_COLS = ["P2_Bull", "P2_Neutral", "P2_Bear", "regime_age_days"]

ARCH_FEATURE_COLS = {
    "flat":          OSC_COLS,
    "2stage_trend":  S1_COLS + OSC_COLS,
    "2stage_macro":  S2_COLS + OSC_COLS,
    "3stage_full":   S1_COLS + S2_COLS + OSC_COLS,
}


def _hp_to_factory(model_name: str, best_params: dict) -> dict:
    """Translate Optuna best_params dict to factory kwargs."""
    hp = dict(best_params)
    if model_name == "mlp" and "hidden_layer_sizes" in hp:
        if isinstance(hp["hidden_layer_sizes"], list):
            hp["hidden_layer_sizes"] = tuple(hp["hidden_layer_sizes"])
    return hp


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out = PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation"
    out.mkdir(parents=True, exist_ok=True)

    inner_cfg = InnerCVConfig()  # 5 fold default

    overall_rows = []
    backtest_rows = []
    optuna_best_rows = []
    equity_records: dict[str, dict[str, pd.Series]] = {a: {} for a in ASSETS}

    for asset in ASSETS:
        full = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                           index_col=0, parse_dates=True)
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        prices = ohlcv["Close"]

        for arch_name, arch_cols in ARCH_FEATURE_COLS.items():
            X = full[arch_cols]
            y = full["signal_label"]
            print(f"\n[{asset.upper()}/{arch_name}] {X.shape}, "
                  f"span {X.index.min().date()} -> {X.index.max().date()}",
                  flush=True)

            for model_name in MODELS:
                # Optuna tune
                t0 = time.time()
                study, best = run_study(
                    asset=f"{asset}_{arch_name}",
                    model_name=model_name,
                    X=X, y=y,
                    n_trials=30, balanced=True,
                    inner_cfg=inner_cfg,
                    storage=None,  # in-memory for speed
                )
                tune_dt = time.time() - t0

                optuna_best_rows.append({
                    "asset":       asset,
                    "arch":        arch_name,
                    "model":       model_name,
                    "best_value":  study.best_value,
                    "best_trial":  study.best_trial.number,
                    "n_pruned":    sum(1 for t in study.trials if t.state.name == "PRUNED"),
                    "best_params": json.dumps(best, default=lambda o: list(o) if isinstance(o, tuple) else o),
                    "elapsed_s":   round(tune_dt, 1),
                })

                # Tuned outer retrain
                hp = _hp_to_factory(model_name, best)
                oof, folds = train_walk_forward(X, y, model_name, balanced=True, hp=hp)

                oof_path = proc / f"{asset}_stage3_oof_{model_name}_v5_tuned_{arch_name}.csv"
                oof.to_csv(oof_path)

                ov = overall_metrics(oof)
                overall_rows.append({
                    "asset":     asset,
                    "arch":      arch_name,
                    "model":     model_name,
                    "n_oof":     ov["n"],
                    "accuracy":  ov["accuracy"],
                    "f1_macro":  ov["f1_macro"],
                    "f1_sell":   ov["f1_per_class"][0],
                    "f1_hold":   ov["f1_per_class"][1],
                    "f1_buy":    ov["f1_per_class"][2],
                })

                # Backtest 3 rules
                for rule_name, fn in [
                    ("stateful",      backtest_stateful),
                    ("defensive",     backtest_defensive),
                    ("prob_weighted", backtest_prob_weighted),
                ]:
                    res, eq = fn(oof, prices, asset=asset, model=model_name)
                    row = res.to_dict()
                    row["arch"] = arch_name
                    backtest_rows.append(row)
                    equity_records[asset][f"{arch_name}_{model_name}_{rule_name}"] = eq

                print(f"  {model_name:14s}  inner_F1m {study.best_value:.3f}  "
                      f"outer_F1m {ov['f1_macro']:.3f}  ({tune_dt:.0f}s tune)",
                      flush=True)

        # B&H per asset
        first_oof_path = proc / f"{asset}_stage3_oof_xgboost_v5_tuned_3stage_full.csv"
        first_oof = pd.read_csv(first_oof_path, index_col=0, parse_dates=True)
        bh_res, bh_eq = backtest_buy_and_hold(prices, first_oof.index.min(),
                                              first_oof.index.max(), asset=asset)
        bh_row = bh_res.to_dict(); bh_row["arch"] = "buy_and_hold"
        backtest_rows.append(bh_row)
        equity_records[asset]["BUY_AND_HOLD"] = bh_eq

    # Save
    pd.DataFrame(overall_rows).to_csv(out / "v5_p5_arch_overall.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(out / "v5_p5_arch_backtest_summary.csv", index=False)
    pd.DataFrame(optuna_best_rows).to_csv(out / "v5_p5_arch_optuna_best.csv", index=False)
    for asset, eq_dict in equity_records.items():
        pd.DataFrame(eq_dict).to_csv(out / f"v5_p5_arch_equity_curves_{asset}.csv")

    print(f"\n=== Phase A complete ===")
    print(f"reports -> {out.relative_to(PROJECT_ROOT)}/")

    # Quick ablation summary
    bt = pd.DataFrame(backtest_rows)
    bt = bt[bt["arch"] != "buy_and_hold"]
    print("\n=== Best Sharpe per (asset, arch) — across all rules+models ===")
    best_per = bt.loc[bt.groupby(["asset", "arch"])["annualized_sharpe"].idxmax()]
    print(best_per[["asset", "arch", "rule", "model", "annualized_sharpe",
                    "total_return", "max_drawdown"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
