"""V5 Phase 3 — Build Stage 1 dataset (features + ZigZag labels).

Outputs:
  data/processed/btc_features_stage1_v5_zz.csv
  data/processed/eth_features_stage1_v5_zz.csv

Each CSV has 14 features + 1 label column. NaN warm-up rows dropped.
ZigZag deviation_pct = 0.10, min_segment_days = 10, range_amplitude = 0.05.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.features.v5_trend_features import build_stage1_features, STAGE1_FEATURE_COLS
from src.labels.v5_trend_labels import zigzag_trend_label

DEV_PCT = 0.10
MIN_SEG = 15
AMP = 0.075


def build(asset: str):
    proc = PROJECT_ROOT / "data" / "processed"
    df = pd.read_csv(proc / f"{asset}_aligned_v5.csv", index_col=0, parse_dates=True)

    feats = build_stage1_features(df[["Open", "High", "Low", "Close", "Volume"]])
    labels = zigzag_trend_label(df["Close"], deviation_pct=DEV_PCT,
                                min_segment_days=MIN_SEG, range_amplitude=AMP)

    out = feats.copy()
    out["trend_label"] = labels["label"]
    out["segment_id"] = labels["segment_id"]
    out = out.dropna(subset=STAGE1_FEATURE_COLS + ["trend_label"])

    out_path = proc / f"{asset}_features_stage1_v5_zz.csv"
    out.to_csv(out_path)
    n = len(out)
    dist = out["trend_label"].value_counts(normalize=True)
    print(f"{asset.upper()}: {n} rows, span {out.index.min().date()} -> {out.index.max().date()}")
    print(f"  Distribution: up {dist.get('uptrend', 0):.1%} / down {dist.get('downtrend', 0):.1%} / range {dist.get('range', 0):.1%}")
    print(f"  Saved: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    build("btc")
    build("eth")


if __name__ == "__main__":
    main()
