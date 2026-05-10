"""V5 Phase H — Stage 3 walk-forward isotonic calibration.

Re-runs the per-asset best (rule, model, arch) Stage 3 with isotonic
calibration applied at each outer fold. Calibration is fit on the last
~200 days of each fold's training span (held out from model fit, no
look-ahead leakage).

Then re-runs the 3 trading rules on BOTH the raw and calibrated OOF
predictions, comparing Sharpe / Return / MaxDD / # trades / win rate.

Outputs:
  data/processed/{asset}_stage3_oof_{model}_v5_calibrated_{arch}.csv
  reports/Phase8_calibration/v5_p8_calibration_metrics.csv
  reports/Phase8_calibration/v5_p8_backtest_compare.csv

Best per-asset arch+model is hard-coded from the Phase 5.1 + Phase 7
ablation results.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.v5_backtester import (
    backtest_stateful, backtest_defensive, backtest_prob_weighted,
)
from src.features.v5_stage3_features import (
    STAGE3_FEATURE_COLS, STAGE3_FEATURE_COLS_EXTENDED,
)
from src.models.v5_stage3_calibration import (
    CalibrationConfig, calibrated_walk_forward, calibration_metrics,
)


# Best per-asset arch + model (from Phase 5.1 + Phase 7)
# Will be expanded after Phase 7 finishes — for now we use the canonical
# Phase 5.1 winners (3stage_full + xgb for BTC, flat + lgbm for ETH).
TARGETS = [
    {"asset": "btc", "arch": "3stage_full", "model": "xgboost"},
    {"asset": "btc", "arch": "3stage_full", "model": "lightgbm"},
    {"asset": "btc", "arch": "3stage_full", "model": "random_forest"},
    {"asset": "btc", "arch": "3stage_full", "model": "mlp"},
    {"asset": "eth", "arch": "flat",        "model": "lightgbm"},
    {"asset": "eth", "arch": "flat",        "model": "xgboost"},
    {"asset": "eth", "arch": "flat",        "model": "random_forest"},
    {"asset": "eth", "arch": "flat",        "model": "mlp"},
]

ARCH_FEATURE_COLS = {
    "flat":          ["RSI_14", "MACD_signal_diff", "Bollinger_pct_b",
                      "Stochastic_K_14", "volume_zscore_20", "OBV_change_20d"],
    "2stage_trend":  STAGE3_FEATURE_COLS[:6] + STAGE3_FEATURE_COLS[10:],
    "2stage_macro":  STAGE3_FEATURE_COLS[6:10] + STAGE3_FEATURE_COLS[10:],
    "3stage_full":   STAGE3_FEATURE_COLS,
    "3stage_full_regime": STAGE3_FEATURE_COLS_EXTENDED,
}


def _load_best_params(asset: str, arch: str, model: str) -> dict:
    """Read tuned HP from Phase 5.1 (or Phase 7 if regime variant)."""
    if arch == "3stage_full_regime":
        path = PROJECT_ROOT / "reports" / "Phase7_regime_aware" / "v5_p7_optuna_best.csv"
    else:
        path = PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation" / "v5_p5_arch_optuna_best.csv"
    df = pd.read_csv(path)
    row = df[(df["asset"] == asset) & (df["arch"] == arch) & (df["model"] == model)]
    if not len(row):
        raise ValueError(f"No tuned params for {asset}/{arch}/{model} in {path.name}")
    hp = json.loads(row.iloc[0]["best_params"])
    if isinstance(hp.get("hidden_layer_sizes"), list):
        hp["hidden_layer_sizes"] = tuple(hp["hidden_layer_sizes"])
    return hp


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out  = PROJECT_ROOT / "reports" / "Phase8_calibration"
    out.mkdir(parents=True, exist_ok=True)

    cal_metrics_rows = []
    backtest_rows    = []

    for tgt in TARGETS:
        asset = tgt["asset"]; arch = tgt["arch"]; model_name = tgt["model"]
        print(f"\n[{asset.upper()}/{arch}/{model_name}] calibrating...", flush=True)

        full = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                           index_col=0, parse_dates=True)
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        prices = ohlcv["Close"]

        cols = ARCH_FEATURE_COLS[arch]
        X = full[cols]
        y = full["signal_label"]

        try:
            hp = _load_best_params(asset, arch, model_name)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ! skip: {e}")
            continue

        oof_cal = calibrated_walk_forward(
            X, y, model_name, hp=hp, balanced=True,
            cal_cfg=CalibrationConfig(method="isotonic", calib_size=200),
        )

        # Save OOF (raw + calibrated)
        oof_path = proc / f"{asset}_stage3_oof_{model_name}_v5_calibrated_{arch}.csv"
        oof_cal.to_csv(oof_path)

        # Calibration metrics (ECE + Brier per class)
        m = calibration_metrics(oof_cal)
        for variant in ["raw", "cal"]:
            for c_row in m[variant]:
                cal_metrics_rows.append({
                    "asset": asset, "arch": arch, "model": model_name,
                    "variant": variant, "class": c_row["class"],
                    "ece": c_row["ece"], "brier": c_row["brier"],
                })
        print(f"  ECE  raw={m['raw_macro_ece']:.4f}  cal={m['cal_macro_ece']:.4f}  "
              f"(reduction {(1 - m['cal_macro_ece']/max(m['raw_macro_ece'],1e-9))*100:+.1f}%)")
        print(f"  Brier raw={m['raw_macro_brier']:.4f} cal={m['cal_macro_brier']:.4f}")

        # Backtest both raw and calibrated; the prob_weighted rule is the most
        # calibration-sensitive (uses raw P_Buy − P_Sell directly).
        for variant_name, variant_oof in [
            ("raw", _build_raw_oof_df(oof_cal)),
            ("cal", _build_cal_oof_df(oof_cal)),
        ]:
            for rule_name, fn in [
                ("stateful",      backtest_stateful),
                ("defensive",     backtest_defensive),
                ("prob_weighted", backtest_prob_weighted),
            ]:
                res, _ = fn(variant_oof, prices, asset=asset, model=model_name)
                row = res.to_dict()
                row["arch"] = arch
                row["variant"] = variant_name
                backtest_rows.append(row)

    pd.DataFrame(cal_metrics_rows).to_csv(out / "v5_p8_calibration_metrics.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(out / "v5_p8_backtest_compare.csv", index=False)

    print(f"\n=== Phase 8 (calibration) complete ===")
    print(f"reports -> {out.relative_to(PROJECT_ROOT)}/")

    bt = pd.DataFrame(backtest_rows)
    print("\n=== Sharpe lift from calibration (per asset/arch/model, prob_weighted rule) ===")
    pw = bt[bt["rule"] == "prob_weighted"]
    pivot = pw.pivot_table(index=["asset","arch","model"], columns="variant",
                            values="annualized_sharpe").reset_index()
    pivot["lift"] = pivot["cal"] - pivot["raw"]
    print(pivot.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    return 0


def _build_raw_oof_df(oof_cal: pd.DataFrame) -> pd.DataFrame:
    """Subset to columns that mimic an uncalibrated OOF schema (used by backtester)."""
    return oof_cal.rename(columns={
        "P_Sell": "P_Sell", "P_Hold": "P_Hold", "P_Buy": "P_Buy",
        "pred_label": "pred_label",
    })[["P_Sell", "P_Hold", "P_Buy", "pred_label", "true_label", "fold"]]


def _build_cal_oof_df(oof_cal: pd.DataFrame) -> pd.DataFrame:
    df = oof_cal.copy()
    df["P_Sell"] = df["P_Sell_cal"]
    df["P_Hold"] = df["P_Hold_cal"]
    df["P_Buy"]  = df["P_Buy_cal"]
    df["pred_label"] = df["pred_label_cal"]
    return df[["P_Sell", "P_Hold", "P_Buy", "pred_label", "true_label", "fold"]]


if __name__ == "__main__":
    raise SystemExit(main())
