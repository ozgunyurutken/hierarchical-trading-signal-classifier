"""V5 Phase 5 — Backtest tuned Stage 3 OOF predictions.

For each (asset, model, trading_rule) combination:
  - run backtest on Stage 3 tuned OOF
  - compute total_return, Sharpe, MaxDD, n_trades, win_rate
  - save equity curves
B&H benchmark added per asset.

Outputs:
  reports/Phase5/v5_p5_backtest_summary.csv
  reports/Phase5/v5_p5_equity_curves_{asset}.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.v5_backtester import (
    backtest_stateful, backtest_defensive, backtest_prob_weighted,
    backtest_buy_and_hold,
)


ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
RULES = [
    ("stateful",      backtest_stateful),
    ("defensive",     backtest_defensive),
    ("prob_weighted", backtest_prob_weighted),
]


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out = PROJECT_ROOT / "reports" / "Phase5"
    out.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for asset in ASSETS:
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        prices = ohlcv["Close"]

        equity_curves = {}

        # First model's OOF span = backtest span (all 4 models share span)
        for model in MODELS:
            oof_path = proc / f"{asset}_stage3_oof_{model}_v5_tuned.csv"
            if not oof_path.exists():
                print(f"  ! skip {asset}/{model}: {oof_path.name} not found")
                continue
            oof = pd.read_csv(oof_path, index_col=0, parse_dates=True)

            for rule_name, fn in RULES:
                res, eq = fn(oof, prices, asset=asset, model=model)
                summary_rows.append(res.to_dict())
                equity_curves[f"{model}_{rule_name}"] = eq
                print(f"  {asset}/{model:14s}/{rule_name:14s}  "
                      f"return {res.total_return:+.2%}  "
                      f"Sharpe {res.annualized_sharpe:+.2f}  "
                      f"MaxDD {res.max_drawdown:.2%}  "
                      f"trades {res.n_trades:3d}  "
                      f"win {res.win_rate:.1%}")

        # B&H benchmark (use first available OOF span)
        first_oof = pd.read_csv(
            proc / f"{asset}_stage3_oof_xgboost_v5_tuned.csv",
            index_col=0, parse_dates=True,
        )
        bh_res, bh_eq = backtest_buy_and_hold(
            prices, first_oof.index.min(), first_oof.index.max(), asset=asset
        )
        summary_rows.append(bh_res.to_dict())
        equity_curves["BUY_AND_HOLD"] = bh_eq
        print(f"  {asset}/B&H                                "
              f"return {bh_res.total_return:+.2%}  "
              f"Sharpe {bh_res.annualized_sharpe:+.2f}  "
              f"MaxDD {bh_res.max_drawdown:.2%}")
        print()

        eq_df = pd.DataFrame(equity_curves)
        eq_path = out / f"v5_p5_equity_curves_{asset}.csv"
        eq_df.to_csv(eq_path)
        print(f"  saved {eq_path.relative_to(PROJECT_ROOT)}")
        print()

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out / "v5_p5_backtest_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path.relative_to(PROJECT_ROOT)}\n")

    # Pretty per-asset summary
    for asset in ASSETS:
        print(f"=== {asset.upper()} ===")
        sub = summary_df[summary_df["asset"] == asset].copy()
        # Sort by Sharpe descending
        sub = sub.sort_values("annualized_sharpe", ascending=False).reset_index(drop=True)
        cols = ["rule", "model", "total_return", "annualized_sharpe",
                "max_drawdown", "n_trades", "win_rate"]
        print(sub[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
