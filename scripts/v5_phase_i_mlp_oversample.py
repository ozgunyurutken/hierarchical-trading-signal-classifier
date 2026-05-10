"""V5 Phase I — MLP RandomOverSampler retrain.

scikit-learn's MLPClassifier does not natively support class_weight, so
on the heavily imbalanced Stage 3 label distribution (Buy 44% / Hold 22%
/ Sell 34%) the MLP under-predicts Hold (especially on ETH, where the
baseline MLP scores Hold-F1 ≈ 0.014). This script retrains the MLP for
each (asset, arch) target with imblearn.RandomOverSampler applied
inside each walk-forward training fold.

The RandomOverSampler is fit ONLY on the training fold of each split
(no leakage to validation). It duplicates minority-class samples until
all classes have equal counts.

Output:
  data/processed/{asset}_stage3_oof_mlp_v5_oversampled_{arch}.csv
  reports/Phase9_mlp_oversample/v5_p9_overall.csv
  reports/Phase9_mlp_oversample/v5_p9_backtest.csv
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
from src.models.v5_stage3_trainer import (
    train_walk_forward, overall_metrics,
)


# Target (asset, arch) combos where we want to retry MLP with oversample
TARGETS = [
    {"asset": "btc", "arch": "3stage_full"},        # BTC's main 3-stage
    {"asset": "btc", "arch": "flat"},
    {"asset": "eth", "arch": "flat"},               # ETH's main winner
    {"asset": "eth", "arch": "3stage_full"},
]

ARCH_FEATURE_COLS = {
    "flat":          ["RSI_14", "MACD_signal_diff", "Bollinger_pct_b",
                      "Stochastic_K_14", "volume_zscore_20", "OBV_change_20d"],
    "3stage_full":   STAGE3_FEATURE_COLS,
    "3stage_full_regime": STAGE3_FEATURE_COLS_EXTENDED,
}


def _load_best_params(asset: str, arch: str) -> dict:
    src = (PROJECT_ROOT / "reports" / "Phase7_regime_aware" / "v5_p7_optuna_best.csv"
           if arch == "3stage_full_regime"
           else PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation" / "v5_p5_arch_optuna_best.csv")
    df = pd.read_csv(src)
    row = df[(df["asset"] == asset) & (df["arch"] == arch) & (df["model"] == "mlp")]
    if not len(row):
        raise ValueError(f"No tuned MLP HP for {asset}/{arch}")
    hp = json.loads(row.iloc[0]["best_params"])
    if isinstance(hp.get("hidden_layer_sizes"), list):
        hp["hidden_layer_sizes"] = tuple(hp["hidden_layer_sizes"])
    return hp


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out  = PROJECT_ROOT / "reports" / "Phase9_mlp_oversample"
    out.mkdir(parents=True, exist_ok=True)

    overall_rows, backtest_rows = [], []

    for tgt in TARGETS:
        asset, arch = tgt["asset"], tgt["arch"]
        full = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                           index_col=0, parse_dates=True)
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        prices = ohlcv["Close"]

        cols = ARCH_FEATURE_COLS[arch]
        X = full[cols]
        y = full["signal_label"]

        try:
            hp = _load_best_params(asset, arch)
        except (FileNotFoundError, ValueError) as e:
            print(f"  ! skip {asset}/{arch}/mlp: {e}")
            continue

        print(f"\n[{asset.upper()}/{arch}/mlp] retraining with RandomOverSampler...", flush=True)

        oof, _ = train_walk_forward(
            X, y, "mlp", balanced=True, hp=hp, mlp_oversample=True,
        )

        oof_path = proc / f"{asset}_stage3_oof_mlp_v5_oversampled_{arch}.csv"
        oof.to_csv(oof_path)

        ov = overall_metrics(oof)
        overall_rows.append({
            "asset": asset, "arch": arch, "model": "mlp",
            "variant": "oversampled",
            "n_oof": ov["n"],
            "accuracy": ov["accuracy"],
            "f1_macro": ov["f1_macro"],
            "f1_sell": ov["f1_per_class"][0],
            "f1_hold": ov["f1_per_class"][1],
            "f1_buy":  ov["f1_per_class"][2],
        })
        print(f"  F1m {ov['f1_macro']:.3f}  hold_F1 {ov['f1_per_class'][1]:.3f}  "
              f"sell_F1 {ov['f1_per_class'][0]:.3f}  buy_F1 {ov['f1_per_class'][2]:.3f}")

        for rule_name, fn in [
            ("stateful",      backtest_stateful),
            ("defensive",     backtest_defensive),
            ("prob_weighted", backtest_prob_weighted),
        ]:
            res, _ = fn(oof, prices, asset=asset, model="mlp")
            row = res.to_dict()
            row["arch"] = arch
            row["variant"] = "oversampled"
            backtest_rows.append(row)

    pd.DataFrame(overall_rows).to_csv(out / "v5_p9_overall.csv", index=False)
    pd.DataFrame(backtest_rows).to_csv(out / "v5_p9_backtest.csv", index=False)

    print(f"\n=== Phase 9 (MLP oversample) complete ===")

    # Compare with Phase 5.1 MLP baseline
    p51 = PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation" / "v5_p5_arch_overall.csv"
    if p51.exists():
        b = pd.read_csv(p51)
        b = b[b["model"] == "mlp"]
        print("\n=== Comparison: Phase 5.1 MLP baseline vs oversampled ===")
        cmp = pd.DataFrame(overall_rows).merge(
            b.rename(columns={"f1_macro":"f1m_base","f1_hold":"hold_base"}),
            on=["asset","arch","model"], how="left",
        )[["asset","arch","f1_macro","f1m_base","f1_hold","hold_base"]]
        cmp.columns = ["asset","arch","f1m_oversample","f1m_base","hold_oversample","hold_base"]
        cmp["d_f1m"]  = cmp["f1m_oversample"] - cmp["f1m_base"]
        cmp["d_hold"] = cmp["hold_oversample"] - cmp["hold_base"]
        print(cmp.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
