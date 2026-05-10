"""V5 Phase F-prep — Save final-fit models for the What-If demo endpoint.

Reads Optuna best HPs from reports/Phase5.1_arch_ablation/v5_p5_arch_optuna_best.csv,
fits one final model per (asset x arch x model) on the full Stage-3 feature set,
and serializes a bundle (model + scaler + feature columns + class names) to
app/models/v5/{asset}_{arch}_{model}.joblib.

These bundles back the /predict_custom endpoint (interactive feature sliders).
Backtest curves shown elsewhere remain walk-forward OOF (Phase 5.1) — final-fit
models here are exclusively for the interactive what-if exploration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.v5_stage3_trainer import (
    MODEL_FACTORIES, TREE_MODELS, LABEL_TO_IDX, _balanced_sample_weight,
)


ASSETS = ["btc", "eth"]

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


def _hp_from_json(model_name: str, raw: str) -> dict:
    hp = json.loads(raw)
    if model_name == "mlp" and "hidden_layer_sizes" in hp:
        if isinstance(hp["hidden_layer_sizes"], list):
            hp["hidden_layer_sizes"] = tuple(hp["hidden_layer_sizes"])
    return hp


def fit_one(asset: str, arch: str, model_name: str, hp: dict, balanced: bool = True):
    proc = PROJECT_ROOT / "data" / "processed"
    full = pd.read_csv(proc / f"{asset}_features_stage3_v5.csv",
                       index_col=0, parse_dates=True)
    cols = ARCH_FEATURE_COLS[arch]
    X = full[cols]
    y = full["signal_label"]

    X_arr = X.to_numpy()
    y_idx = y.map(LABEL_TO_IDX).to_numpy()

    if model_name in TREE_MODELS:
        model = MODEL_FACTORIES[model_name](random_state=42, balanced=balanced, **hp)
        if model_name == "xgboost" and balanced:
            sw = _balanced_sample_weight(y_idx)
            model.fit(X_arr, y_idx, sample_weight=sw)
        else:
            model.fit(X_arr, y_idx)
        scaler = None
    else:
        scaler = StandardScaler().fit(X_arr)
        model = MODEL_FACTORIES[model_name](random_state=42, balanced=balanced, **hp)
        model.fit(scaler.transform(X_arr), y_idx)

    return {
        "model":        model,
        "scaler":       scaler,
        "feature_cols": cols,
        "classes":      ["Sell", "Hold", "Buy"],
        "asset":        asset,
        "arch":         arch,
        "model_name":   model_name,
        "n_train":      int(len(X_arr)),
        "trained_at":   pd.Timestamp.utcnow().isoformat(),
    }


def main() -> int:
    optuna_csv = (PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation"
                  / "v5_p5_arch_optuna_best.csv")
    if not optuna_csv.exists():
        print(f"!! Missing Optuna best HPs: {optuna_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(optuna_csv)
    print(f"Loaded {len(df)} (asset, arch, model) configs from Optuna best CSV.",
          flush=True)

    out_dir = PROJECT_ROOT / "app" / "models" / "v5"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    t_overall = time.time()
    for i, row in df.iterrows():
        asset      = str(row["asset"]).lower()
        arch       = str(row["arch"])
        model_name = str(row["model"])
        hp = _hp_from_json(model_name, row["best_params"])

        t0 = time.time()
        try:
            bundle = fit_one(asset, arch, model_name, hp, balanced=True)
        except Exception as e:
            print(f"  [{i+1:2d}/{len(df)}] {asset}/{arch}/{model_name} FAILED: {e}",
                  flush=True)
            continue

        out_path = out_dir / f"{asset}_{arch}_{model_name}.joblib"
        joblib.dump(bundle, out_path, compress=3)
        dt = time.time() - t0

        manifest_rows.append({
            "asset":       asset,
            "arch":        arch,
            "model":       model_name,
            "file":        out_path.name,
            "size_kb":     round(out_path.stat().st_size / 1024, 1),
            "fit_seconds": round(dt, 1),
            "n_train":     bundle["n_train"],
            "feature_count": len(bundle["feature_cols"]),
        })
        print(f"  [{i+1:2d}/{len(df)}] {asset}/{arch:>13}/{model_name:>13}  "
              f"{dt:5.1f}s  -> {out_path.name} "
              f"({manifest_rows[-1]['size_kb']:.0f} KB)", flush=True)

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"\n=== Done in {time.time() - t_overall:.0f}s. "
          f"{len(manifest_rows)} bundles -> {out_dir.relative_to(PROJECT_ROOT)}/",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
