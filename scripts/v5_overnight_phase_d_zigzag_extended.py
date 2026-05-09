"""V5 Overnight Phase D — Extended ZigZag config sweep (distribution analysis).

Current Stage 1 uses config D (deviation_pct=0.10, min_segment=15, range_amp=0.075).
This phase tests 16 wider config grid for label distribution balance.

If a more balanced config is found (closer to 33/33/33), Phase E may retrain Stage 1
(if time permits). Otherwise just report findings for paper Discussion.

Outputs:
  reports/Phase3.6_zigzag_extended/v5_p3_zigzag_extended_dist.csv
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.labels.v5_trend_labels import zigzag_trend_label, label_distribution


CONFIG_GRID = list(product(
    [0.07, 0.10, 0.12, 0.15],            # deviation_pct
    [10, 15, 20, 25],                     # min_segment_days
    [0.05, 0.075, 0.10],                  # range_amplitude
))[:16]  # cap at 16


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out = PROJECT_ROOT / "reports" / "Phase3.6_zigzag_extended"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for asset in ["btc", "eth"]:
        ohlcv = pd.read_csv(proc / f"{asset}_aligned_v5.csv",
                            index_col=0, parse_dates=True)
        close = ohlcv["Close"]
        for dev, min_seg, amp in CONFIG_GRID:
            label_df = zigzag_trend_label(close, deviation_pct=dev,
                                           min_segment_days=min_seg,
                                           range_amplitude=amp)
            # zigzag_trend_label returns DataFrame[label, segment_id]
            label_series = label_df["label"] if "label" in label_df.columns else label_df.iloc[:, 0]
            shares = label_series.dropna().value_counts(normalize=True).reindex(
                ["downtrend", "range", "uptrend"], fill_value=0.0
            )
            counts = label_series.dropna().value_counts().reindex(
                ["downtrend", "range", "uptrend"], fill_value=0
            )
            # Balance score: how close to 33/33/33 (lower = better)
            target = pd.Series([1/3]*3, index=["downtrend", "range", "uptrend"])
            imbalance = float((shares - target).abs().sum())
            rows.append({
                "asset": asset,
                "deviation_pct": dev,
                "min_segment_days": min_seg,
                "range_amplitude": amp,
                "n_total": int(counts.sum()),
                "n_down": int(counts["downtrend"]),
                "n_range": int(counts["range"]),
                "n_up":   int(counts["uptrend"]),
                "share_down": float(shares["downtrend"]),
                "share_range": float(shares["range"]),
                "share_up":   float(shares["uptrend"]),
                "imbalance":  imbalance,
            })

    df = pd.DataFrame(rows).sort_values(["asset", "imbalance"])
    df.to_csv(out / "v5_p3_zigzag_extended_dist.csv", index=False)

    print(f"\n=== Phase D complete ===")
    print(f"reports -> {(out / 'v5_p3_zigzag_extended_dist.csv').relative_to(PROJECT_ROOT)}")
    print()

    for asset in ["btc", "eth"]:
        sub = df[df["asset"] == asset].head(5)
        print(f"=== {asset.upper()} TOP 5 most balanced configs ===")
        print(sub[["deviation_pct", "min_segment_days", "range_amplitude",
                   "share_down", "share_range", "share_up", "imbalance"]].to_string(index=False))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
