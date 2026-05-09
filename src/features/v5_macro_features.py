"""
V5 Faz 2 — Macro feature engineering.

Stage 2 K-Means input için 10 derived feature üretir:

  Long-term static z-score (baseline 2000-2025):
    VIX_zscore_long, DXY_zscore_long

  Rate-of-Change (RoC):
    FEDFUNDS_change_60d, UNRATE_change_180d
    SP500_log_return_5d, Gold_log_return_20d

  Year-over-Year (YoY %):
    CPI_yoy_change, M2_yoy_change

  Derived spreads:
    Yield_Curve_10Y_2Y (= US10Y - US2Y)
    Gold_Silver_Ratio (= Gold / Silver, optional ablation)

  Raw level (sabit, kıyas için):
    VIX (kept raw — level itself meaningful "fear gauge")

10 final stage 2 features (config'te listelendiği gibi).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRETRAIN_START = "2000-01-01"   # config.yaml > stage2_macro_pretrain.start_date


def long_term_zscore(series: pd.Series,
                     baseline_start: str = PRETRAIN_START,
                     baseline_end: str | None = None) -> pd.Series:
    """Sabit baseline mean+std üzerinden z-score."""
    baseline = series.loc[baseline_start:baseline_end].dropna()
    mu, sigma = baseline.mean(), baseline.std()
    return (series - mu) / sigma


def rate_of_change(series: pd.Series, window: int) -> pd.Series:
    """n-gün yüzdesel değişim (raw level seriler için)."""
    return series.pct_change(window)


def log_return(series: pd.Series, window: int) -> pd.Series:
    """n-gün log return (price seriler için)."""
    return np.log(series / series.shift(window))


def yoy_change(series: pd.Series, window: int = 252) -> pd.Series:
    """Year-over-Year % değişim. window = trading days/year."""
    return series.pct_change(window)


def build_macro_features(aligned: pd.DataFrame,
                         baseline_start: str = PRETRAIN_START) -> pd.DataFrame:
    """
    Build 10 stage 2 macro features from raw aligned dataset.

    Aligned dataset must contain raw columns:
      SP500, VIX, DXY, Gold, Silver, US10Y, US2Y,
      FEDFUNDS, CPIAUCSL, UNRATE, WM2NS

    Returns: DataFrame with 10 derived features + raw VIX (kept for reference).
    """
    out = pd.DataFrame(index=aligned.index)

    # 1. Long-term static z-scores (stable across pre-train and crypto era)
    out["VIX"] = aligned["VIX"]   # raw level kept (fear gauge interpretable)
    out["VIX_zscore_long"] = long_term_zscore(aligned["VIX"], baseline_start)
    out["DXY_zscore_long"] = long_term_zscore(aligned["DXY"], baseline_start)

    # 2. Log returns (price momentum)
    out["SP500_log_return_5d"] = log_return(aligned["SP500"], 5)
    out["Gold_log_return_20d"] = log_return(aligned["Gold"], 20)
    out["Oil_log_return_20d"] = log_return(aligned["Oil"], 20)

    # 3. Rate-of-Change for monthly/level series
    out["FEDFUNDS_change_60d"] = rate_of_change(aligned["FEDFUNDS"], 60)
    out["UNRATE_change_180d"] = rate_of_change(aligned["UNRATE"], 180)

    # 4. Year-over-Year (CPI + M2 growth)
    out["CPI_yoy_change"] = yoy_change(aligned["CPIAUCSL"], 252)
    out["M2_yoy_change"] = yoy_change(aligned["WM2NS"], 252)

    # 5. Derived spread (yield curve)
    out["Yield_Curve_10Y_2Y"] = aligned["US10Y"] - aligned["US2Y"]

    # 6. Optional ablation: Gold/Silver ratio
    out["Gold_Silver_Ratio"] = aligned["Gold"] / aligned["Silver"]

    return out


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned_v5.csv",
                      index_col=0, parse_dates=True)
    feats = build_macro_features(btc)
    print(f"Macro features shape: {feats.shape}")
    print(f"Columns: {list(feats.columns)}")
    print(f"NaN counts:\n{feats.isna().sum()}")
    print(f"\nDescriptive stats (sample):")
    print(feats[["VIX_zscore_long", "FEDFUNDS_change_60d", "CPI_yoy_change", "M2_yoy_change"]].describe().round(3))
