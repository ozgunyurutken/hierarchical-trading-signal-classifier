"""
V5 Faz 2.1.5 — Build macro-only aligned dataset (2000-2025) + derived features.

Crypto-independent macro daily aligned for Stage 2 K-Means cluster fit.
Uses raw macro CSVs from data/raw/v5_macro_*.csv (already 2000-2025).

Outputs:
  data/processed/macro_aligned_pretrain_v5.csv      (raw aligned, 2000-2025)
  data/processed/macro_derived_pretrain_v5.csv      (9 derived stage 2 features)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data.data_aligner import FRED_RELEASE_DELAYS
from src.features.v5_macro_features import build_macro_features


def align_to_business_daily(macro_dfs: dict, fred_monthly: pd.DataFrame,
                            start: str, end: str) -> pd.DataFrame:
    """Align all macro sources to NYSE business-day index, weekly→daily fwd-fill,
    monthly with publication-release lag."""
    bday_idx = pd.bdate_range(start=start, end=end)
    out = pd.DataFrame(index=bday_idx)

    # 1. Daily yfinance series (risk + commodities + yields)
    for cat, df in macro_dfs.items():
        # Reindex to bdays + forward-fill
        df_d = df.reindex(bday_idx).ffill()
        for col in df_d.columns:
            out[col] = df_d[col]

    # 2. FRED monthly + WM2NS weekly with release lag
    for col, lag_days in FRED_RELEASE_DELAYS.items():
        if col not in fred_monthly.columns:
            continue
        s = fred_monthly[col].dropna()
        if len(s) == 0:
            continue
        # Shift index by release lag → "available from" date
        s_shifted = s.copy()
        s_shifted.index = s.index + pd.Timedelta(days=lag_days)
        # Reindex to bdays + forward-fill
        s_d = s_shifted.reindex(bday_idx, method="ffill")
        out[col] = s_d

    return out


def main():
    print("V5 Faz 2.1.5 — Macro Pre-Train Dataset Builder")
    print("=" * 60)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Load raw macro CSVs (already 2000-2025 from v5_build_dataset.py)
    macro_dfs = {}
    for cat in ["risk", "commodities", "yields"]:
        df = pd.read_csv(raw_dir / f"v5_macro_{cat}.csv",
                         index_col=0, parse_dates=True)
        macro_dfs[cat] = df
        print(f"  {cat:12s} {df.shape}  {df.index.min().date()} → {df.index.max().date()}")

    fred_monthly = pd.read_csv(raw_dir / "v5_macro_fred_monthly.csv",
                               index_col=0, parse_dates=True)
    print(f"  fred monthly  {fred_monthly.shape}  {fred_monthly.index.min().date()} → {fred_monthly.index.max().date()}")

    # ---- Align to business-day index 2000-2025 ----
    start = "2000-01-01"
    end = "2025-12-31"
    print(f"\n[align] business-day index {start} → {end}")
    aligned = align_to_business_daily(macro_dfs, fred_monthly, start, end)
    print(f"  aligned shape: {aligned.shape}")
    print(f"  cols ({len(aligned.columns)}): {list(aligned.columns)}")

    # Drop initial NaN rows (warm-up)
    n_before = len(aligned)
    aligned = aligned.dropna(subset=["SP500", "VIX"])  # require core macro
    n_after = len(aligned)
    print(f"  dropped {n_before - n_after} initial NaN rows ({n_before} → {n_after})")
    print(f"  range: {aligned.index.min().date()} → {aligned.index.max().date()}")

    # Save raw aligned
    out_raw = proc_dir / "macro_aligned_pretrain_v5.csv"
    aligned.to_csv(out_raw)
    print(f"\n  saved: {out_raw.relative_to(PROJECT_ROOT)}")

    # ---- Derived features ----
    print("\n[derive] 11 stage 2 derived features (incl. raw VIX kept)")
    derived = build_macro_features(aligned, baseline_start="2000-01-01")
    print(f"  derived shape: {derived.shape}")
    print(f"  NaN per col (warm-up):\n{derived.isna().sum().to_string()}")

    out_der = proc_dir / "macro_derived_pretrain_v5.csv"
    derived.to_csv(out_der)
    print(f"\n  saved: {out_der.relative_to(PROJECT_ROOT)}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Pre-train derived ready for Stage 2 K-Means fit")
    print(f"  Coverage: {derived.index.min().date()} → {derived.index.max().date()}  ({len(derived)} bdays)")
    print(f"  After dropna full set: {derived.dropna().shape[0]} rows usable")


if __name__ == "__main__":
    main()
