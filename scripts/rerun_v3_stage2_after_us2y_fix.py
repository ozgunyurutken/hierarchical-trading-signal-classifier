"""
Step A of the v3 rerun: regenerate macro features and Stage 2 OOF regime
posterior using the patched aligned dataset (US2Y now from FRED DGS2).

Compares old vs new posterior to quantify the impact of the bug fix.
If the change is substantial, Step B (Stage 3 retrain) follows. If not,
the existing Stage 3 models can be kept with a footnote.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import shutil
import datetime as _dt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.features.macro_features import compute_macro_features
from src.labels.regime_labels import compute_oof_regime_posterior, predict_regime_posterior
from src.utils.config import cfg
from src.utils.helpers import save_csv, chronological_train_test_split

config = cfg()
TEST_SIZE = config["training"]["test_size"]
RANDOM_STATE = config["training"]["random_state"]

STAGE2_FEATURE_NAMES = [
    "macro_VIX",
    "macro_VIX_zscore_50",
    "macro_Yield_Curve_10Y_2Y",
    "macro_Credit_Spread_log",
    "macro_Gold_Silver_Ratio",
    "macro_SP500_VIX_ratio",
    "macro_DXY_zscore_50",
    "macro_SP500_roc_20",
]


def main() -> None:
    print("[1] Load patched aligned + recompute macro features")
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    print(f"  btc_aligned: {len(btc)} rows × {len(btc.columns)} cols, "
          f"{btc.index.min().date()} → {btc.index.max().date()}")

    macro_feats = compute_macro_features(btc)
    save_csv(macro_feats, PROJECT_ROOT / "data" / "processed" / "btc_features_macro.csv")
    print(f"  Recomputed macro features: {macro_feats.shape}")

    print("\n[2] Verify yield curve is in a sensible range now")
    yc = macro_feats["macro_Yield_Curve_10Y_2Y"].dropna()
    print(f"  macro_Yield_Curve_10Y_2Y: {yc.min():.2f} → {yc.max():.2f}, median {yc.median():.2f}")

    print("\n[3] Build Stage 2 input (8-feature subset, train/test split)")
    available = [c for c in STAGE2_FEATURE_NAMES if c in macro_feats.columns]
    missing = [c for c in STAGE2_FEATURE_NAMES if c not in macro_feats.columns]
    if missing:
        print(f"  WARN missing: {missing}")
    print(f"  Stage 2 features ({len(available)}): {available}")

    X_stage2 = macro_feats[available].dropna()
    print(f"  Stage 2 input shape: {X_stage2.shape}, "
          f"{X_stage2.index.min().date()} → {X_stage2.index.max().date()}")

    X_stage2_train, X_stage2_test = chronological_train_test_split(X_stage2, test_size=TEST_SIZE)
    print(f"  Train: {X_stage2_train.shape}, Test: {X_stage2_test.shape}")

    print("\n[4] OOF GMM regime posterior on TRAIN (n_folds=5)")
    oof_train_posterior, full_train_gmm, full_train_scaler = compute_oof_regime_posterior(
        X_stage2_train, method="gmm", n_clusters=3, n_folds=5, random_state=RANDOM_STATE,
    )
    print(f"  OOF train posterior: {oof_train_posterior.shape}")

    print("\n[5] Predict TEST posterior using full-train-fit GMM")
    test_posterior = predict_regime_posterior(
        X_stage2_test, full_train_gmm, full_train_scaler,
        method="gmm", n_clusters=3,
    )
    print(f"  Test posterior: {test_posterior.shape}")

    new_posterior = pd.concat([oof_train_posterior, test_posterior]).sort_index()
    print(f"  Total posterior coverage: {new_posterior.shape}")

    print("\n[6] Compare against the OLD posterior (pre-fix)")
    old_path = PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv"
    old_posterior = pd.read_csv(old_path, index_col=0, parse_dates=True)
    print(f"  Old posterior: {old_posterior.shape}")

    common_idx = new_posterior.index.intersection(old_posterior.index)
    old_aligned = old_posterior.loc[common_idx]
    new_aligned = new_posterior.loc[common_idx]
    print(f"  Common dates: {len(common_idx)}")

    # Hard-label argmax comparison (regime ID matters less than soft posteriors)
    old_hard = old_aligned.values.argmax(axis=1)
    new_hard = new_aligned.values.argmax(axis=1)
    ari = adjusted_rand_score(old_hard, new_hard)
    pct_same = (old_hard == new_hard).mean()
    print(f"  ARI (old vs new hard labels): {ari:.4f}")
    print(f"  Hard-label agreement: {pct_same:.1%}")

    # Soft posterior L2 distance (more sensitive)
    soft_diff = np.linalg.norm(old_aligned.values - new_aligned.values, axis=1)
    print(f"  Soft posterior L2 distance: mean={soft_diff.mean():.3f}, "
          f"median={np.median(soft_diff):.3f}, max={soft_diff.max():.3f}")
    # NOTE: cluster IDs may have permuted; that alone can drive ARI down even
    # if the partitioning is identical. We report both.

    print("\n[7] Backup old + write new posterior")
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = old_path.with_name(f"{old_path.stem}.backup_{ts}{old_path.suffix}")
    shutil.copy(old_path, backup)
    print(f"  Backup: {backup.name}")
    save_csv(new_posterior, old_path)
    print(f"  Wrote: {old_path.name} ({len(new_posterior)} rows)")

    print("\n[8] Decide on Stage 3 retrain necessity")
    if ari < 0.7 or soft_diff.mean() > 0.2:
        print("  >>> RETRAIN RECOMMENDED — clusters changed materially")
    else:
        print("  >>> RETRAIN OPTIONAL — clusters mostly stable, minor soft drift")

    print("\nStep A complete.")


if __name__ == "__main__":
    main()
