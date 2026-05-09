"""V5 Phase 4 — Build Stage 3 Signal Classifier dataset.

Joins per-asset:
  - Stage 1 OOF (RF tuned, _v5_tuned.csv)               -> 3 raw + 3 smoothed10d = 6
  - Stage 2 regime FSM (composite_macro_v5_v5.csv)      -> 3 hard + 1 days = 4
  - Stage 3 oscillators (computed from aligned OHLCV)   -> 6
  - Signal label (V5 spec: h=5, k=0.5, w=20)            -> Buy/Sell/Hold

Output:
  data/processed/{btc,eth}_features_stage3_v5.csv
    columns: STAGE3_FEATURE_COLS + [signal_label, forward_return, eps_threshold]

Usage:
  python scripts/v5_build_stage3_dataset.py
  python scripts/v5_build_stage3_dataset.py --stage1-source mlp  # use MLP OOF instead of RF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.v5_stage3_features import (
    STAGE3_FEATURE_COLS, FEATURE_GROUPS, build_stage3_features,
)
from src.labels.v5_signal_labels import (
    generate_v5_signal_labels, label_distribution, assert_no_lookahead_leakage,
)


ASSETS = ["btc", "eth"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-source", default="random_forest",
                    choices=["xgboost", "lightgbm", "random_forest", "mlp"],
                    help="Which Stage 1 model's tuned OOF to use as Stage 3 input.")
    ap.add_argument("--smooth-window", type=int, default=10,
                    help="Rolling window for Stage 1 OOF smoothing (default 10).")
    ap.add_argument("--h", type=int, default=5,
                    help="Forward horizon for signal label (default 5).")
    ap.add_argument("--k", type=float, default=0.5,
                    help="Volatility multiplier for adaptive threshold (default 0.5).")
    ap.add_argument("--w", type=int, default=20,
                    help="Rolling window for daily-return std (default 20).")
    args = ap.parse_args()

    proc = PROJECT_ROOT / "data" / "processed"
    print(f"[CONFIG] stage1_source={args.stage1_source}, smooth={args.smooth_window}d, "
          f"label h={args.h}, k={args.k}, w={args.w}")
    print()

    for asset in ASSETS:
        print(f"=== {asset.upper()} ===")
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        s1 = pd.read_csv(proc / f"{asset}_stage1_oof_{args.stage1_source}_v5_tuned.csv",
                         index_col=0, parse_dates=True)
        s2 = pd.read_csv(proc / f"{asset}_regime_labels_composite_macro_v5_v5.csv",
                         index_col=0, parse_dates=True)

        feats = build_stage3_features(ohlcv, s1, s2, smooth_window=args.smooth_window)

        labels = generate_v5_signal_labels(ohlcv["Close"], h=args.h, k=args.k, window=args.w)

        # Causality check (sample 1/500 rows)
        assert_no_lookahead_leakage(ohlcv["Close"], labels, h=args.h, window=args.w)

        # Inner-join: keep only dates present in ALL of features + labels + Stage 1 OOF
        joined = feats.join(labels, how="inner").dropna(subset=STAGE3_FEATURE_COLS)
        joined = joined.dropna(subset=["signal_label"])

        # Save
        out_path = proc / f"{asset}_features_stage3_v5.csv"
        joined.to_csv(out_path)
        print(f"  saved: {out_path.relative_to(PROJECT_ROOT)}")
        print(f"  shape: {joined.shape}")
        print(f"  span:  {joined.index.min().date()} -> {joined.index.max().date()}")
        print(f"  feature groups:")
        for name, cols in FEATURE_GROUPS.items():
            n_nan = joined[cols].isna().sum().sum()
            print(f"    {name:30s} ({len(cols)} feats, {n_nan} NaN)")
        print(f"  label distribution:")
        for line in label_distribution(joined["signal_label"]).to_string().splitlines():
            print(f"    {line}")

        # Quick leakage hint: forward_return statistics by class
        print(f"  forward_return by class (sanity, should match label rule):")
        gb = joined.groupby("signal_label")["forward_return"]
        print(f"    Buy   median {gb.median().get('Buy', float('nan')):+.4f}")
        print(f"    Hold  median {gb.median().get('Hold', float('nan')):+.4f}")
        print(f"    Sell  median {gb.median().get('Sell', float('nan')):+.4f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
