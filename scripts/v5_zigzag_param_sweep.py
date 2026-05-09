"""V5 Phase 3 — ZigZag parameter distribution sweep (no training).

Generates ZigZag labels under multiple parameter combinations and reports
the resulting class distribution + segment count for BTC and ETH.

User picks the configuration that yields the most balanced 3-class
distribution; that config feeds into the full 4-classifier retrain step.

Output: console table.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.labels.v5_trend_labels import zigzag_trend_label

CONFIGS = [
    # (deviation_pct, min_segment_days, range_amplitude, name)
    (0.10, 10, 0.05,  "A_baseline"),
    (0.10, 10, 0.075, "B_amp075"),
    (0.10, 10, 0.10,  "C_amp10"),
    (0.10, 15, 0.075, "D_min15_amp075"),
    (0.10, 15, 0.10,  "E_min15_amp10"),
    (0.075, 10, 0.05, "F_dev075"),
    (0.125, 15, 0.10, "G_dev125_min15_amp10"),
    (0.10, 20, 0.10,  "H_min20_amp10"),
    (0.10, 20, 0.125, "I_min20_amp125"),
]


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print(f"{'config':22s}  {'asset':5s}  {'segs':>5s}  {'uptrend':>8s}  {'downtrend':>10s}  {'range':>6s}  {'imbalance':>10s}")
    print("-" * 90)

    for dev, min_seg, amp, name in CONFIGS:
        for asset_label, df in [("BTC", btc), ("ETH", eth)]:
            labels = zigzag_trend_label(df["Close"], deviation_pct=dev,
                                         min_segment_days=min_seg, range_amplitude=amp)
            dist = labels["label"].value_counts(normalize=True)
            up = dist.get("uptrend", 0); down = dist.get("downtrend", 0); rng = dist.get("range", 0)
            n_segs = labels["segment_id"].nunique()
            # imbalance = std of class proportions; lower = more balanced
            import numpy as np
            imbalance = float(np.std([up, down, rng]))
            print(f"{name:22s}  {asset_label:5s}  {n_segs:>5d}  {up:>7.1%}  {down:>9.1%}  {rng:>5.1%}  {imbalance:>9.4f}")
        print()


if __name__ == "__main__":
    main()
