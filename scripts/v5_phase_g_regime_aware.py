"""V5 Phase G — Regime-aware (extended) Stage 3 ablation.

Tests a 5th architecture variant: 3-Stage Full + 6 regime-conditional
interaction features (RSI/MACD/BBpb × P_Bull/P_Bear), 22 features total.

Pipeline mirrors Phase 5.1 arch ablation:
  Optuna 5-fold inner CV, 30 trial × 4 model × 2 asset = 8 studies.
  Then tuned outer retrain (walk-forward expanding-window) + backtest
  with the 3 trading rules.

Outputs:
  data/processed/{asset}_stage3_oof_{model}_v5_tuned_3stage_full_regime.csv
  reports/Phase7_regime_aware/v5_p7_overall.csv
  reports/Phase7_regime_aware/v5_p7_optuna_best.csv
  reports/Phase7_regime_aware/v5_p7_backtest.csv
  reports/Phase7_regime_aware/v5_p7_equity_curves_{btc,eth}.csv
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
from src.features.v5_stage3_features import STAGE3_FEATURE_COLS_EXTENDED
from src.models.v5_stage3_optuna import InnerCVConfig, run_study
from src.models.v5_stage3_trainer import (
    MODEL_FACTORIES, train_walk_forward, overall_metrics,
)


ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
ARCH = "3stage_full_regime"


def _hp_to_factory(model_name: str, best: dict) -> dict:
    hp = dict(best)
    if model_name == "mlp" and isinstance(hp.get("hidden_layer_sizes"), list):
        hp["hidden_layer_sizes"] = tuple(hp["hidden_layer_sizes"])
    return hp


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out  = PROJECT_ROOT / "reports" / "Phase7_regime_aware"
    out.mkdir(parents=True, exist_ok=True)

    inner_cfg = InnerCVConfig()
    overall_rows, backtest_rows, optuna_best_rows = [], [], []
    equity_records: dict[str, dict[str, pd.Series]] = {a: {} for a in ASSETS}

    for asset in ASSETS:
        full = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                           index_col=0, parse_dates=True)
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        prices = ohlcv["Close"]

        X = full[STAGE3_FEATURE_COLS_EXTENDED]   # 22 features
        y = full["signal_label"]
        print(f"\n[{asset.upper()}/{ARCH}] {X.shape}, "
              f"span {X.index.min().date()} -> {X.index.max().date()}", flush=True)

        for model_name in MODELS:
            t0 = time.time()
            study, best = run_study(
                asset=f"{asset}_{ARCH}", model_name=model_name,
                X=X, y=y, n_trials=30, balanced=True,
                inner_cfg=inner_cfg, storage=None,
            )
            tune_dt = time.time() - t0

            optuna_best_rows.append({
                "asset":       asset,
                "arch":        ARCH,
                "model":       model_name,
                "best_value":  study.best_value,
                "best_trial":  study.best_trial.number,
                "n_pruned":    sum(1 for t in study.trials if t.state.name == "PRUNED"),
                "best_params": json.dumps(best, default=lambda o: list(o) if isinstance(o, tuple) else o),
                "elapsed_s":   round(tune_dt, 1),
            })

            hp = _hp_to_factory(model_name, best)
            oof, _ = train_walk_forward(X, y, model_name, balanced=True, hp=hp)

            oof_path = proc / f"{asset}_stage3_oof_{model_name}_v5_tuned_{ARCH}.csv"
            oof.to_csv(oof_path)

            ov = overall_metrics(oof)
            overall_rows.append({
                "asset":     asset,
                "arch":      ARCH,
                "model":     model_name,
                "n_oof":     ov["n"],
                "accuracy":  ov["accuracy"],
                "f1_macro":  ov["f1_macro"],
                "f1_sell":   ov["f1_per_class"][0],
                "f1_hold":   ov["f1_per_class"][1],
                "f1_buy":    ov["f1_per_class"][2],
            })

            for rule_name, fn in [
                ("stateful",      backtest_stateful),
                ("defensive",     backtest_defensive),
                ("prob_weighted", backtest_prob_weighted),
            ]:
                res, eq = fn(oof, prices, asset=asset, model=model_name)
                row = res.to_dict(); row["arch"] = ARCH
                backtest_rows.append(row)
                equity_records[asset][f"{ARCH}_{model_name}_{rule_name}"] = eq

            print(f"  {model_name:14s}  inner_F1m {study.best_value:.3f}  "
                  f"outer_F1m {ov['f1_macro']:.3f}  ({tune_dt:.0f}s)", flush=True)

        # B&H per asset (uses regime-aware OOF span)
        first_oof = pd.read_csv(
            proc / f"{asset}_stage3_oof_xgboost_v5_tuned_{ARCH}.csv",
            index_col=0, parse_dates=True,
        )
        bh_res, bh_eq = backtest_buy_and_hold(
            prices, first_oof.index.min(), first_oof.index.max(), asset=asset
        )
        row = bh_res.to_dict(); row["arch"] = "buy_and_hold"
        backtest_rows.append(row)
        equity_records[asset]["BUY_AND_HOLD"] = bh_eq

    pd.DataFrame(overall_rows).to_csv(out / "v5_p7_overall.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(out / "v5_p7_backtest.csv", index=False)
    pd.DataFrame(optuna_best_rows).to_csv(out / "v5_p7_optuna_best.csv", index=False)
    for asset, eq_dict in equity_records.items():
        pd.DataFrame(eq_dict).to_csv(out / f"v5_p7_equity_curves_{asset}.csv")

    print(f"\n=== Phase 7 (regime-aware) complete ===")
    print(f"reports -> {out.relative_to(PROJECT_ROOT)}/")

    bt = pd.DataFrame(backtest_rows)
    bt = bt[bt["arch"] != "buy_and_hold"]
    print("\n=== Best Sharpe per (asset) for regime-aware variant ===")
    best_per = bt.loc[bt.groupby(["asset"])["annualized_sharpe"].idxmax()]
    print(best_per[["asset", "arch", "rule", "model", "annualized_sharpe",
                    "total_return", "max_drawdown"]].to_string(
                        index=False, float_format=lambda v: f"{v:+.3f}"))

    # Compare with Phase 5.1 baseline for 3stage_full
    p51 = PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation" / "v5_p5_arch_backtest_summary.csv"
    if p51.exists():
        p51_df = pd.read_csv(p51)
        p51_df = p51_df[(p51_df["arch"] == "3stage_full") & (p51_df["model"] != "benchmark")]
        p51_best = p51_df.loc[p51_df.groupby(["asset"])["annualized_sharpe"].idxmax()]
        print("\n=== Phase 5.1 baseline (3stage_full, 16 features) for comparison ===")
        print(p51_best[["asset", "arch", "rule", "model", "annualized_sharpe",
                        "total_return", "max_drawdown"]].to_string(
                            index=False, float_format=lambda v: f"{v:+.3f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
