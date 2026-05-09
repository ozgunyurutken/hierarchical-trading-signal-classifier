"""
Data alignment module.
Aligns crypto OHLCV with daily macro data on a common daily timeline.

Key design decisions:
  1. Forward-fill: weekends/holidays filled from last trading day.
  2. Timezone lag: macro data shifted by macro_lag_days (default 1 day).
     NYSE closes at 21:00 UTC, crypto daily candle closes at 00:00 UTC.
     So T-day macro close is only usable for T+1 crypto signals.
  3. NaN rows at the start (before first macro data point) are dropped.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger, save_csv

logger = setup_logger(__name__)

# BLS / Federal Reserve realistic release schedule (V5 onayı 2026-05-08).
# Önceki V4 değerleri (CPI 45g, UNRATE 35g) ekstra-konservatifti.
# Realistic schedule:
#   CPI:      sonraki ayın 10-13. günü release → 14g lag
#   UNRATE:   sonraki ayın 1. Cuması release → 7g lag
#   FFR:      FOMC kararı anında → 1g lag
#   M2 WM2NS: weekly + 2-hafta gecikme → 14g lag
#   ICSA:     weekly + ~5g → 5g lag
# Reference: López de Prado AFML 2018 [12] "use most recent publicly
# available data with embargo equal to publication delay only".
FRED_RELEASE_DELAYS = {
    "FEDFUNDS": 1,
    "CPIAUCSL": 14,
    "UNRATE": 7,
    "WM2NS": 14,
    "ICSA": 5,
}


def align_market_data_to_crypto(
    market_df: pd.DataFrame,
    crypto_index: pd.DatetimeIndex,
    name: str = "market",
    lag_days: int = 0,
) -> pd.DataFrame:
    """
    Align market data (weekday-only) to crypto daily index (24/7).

    Steps:
      1. If lag_days > 0, shift market data forward (preventing look-ahead).
      2. Forward-fill weekends and holidays from last trading day.

    Parameters
    ----------
    market_df : pd.DataFrame
        Market data with DatetimeIndex (weekday-only).
    crypto_index : pd.DatetimeIndex
        Full daily crypto index (24/7).
    name : str
        Label for logging.
    lag_days : int
        Number of days to shift market data forward.
        Default 0 (no shift). Set to 1 for timezone alignment.
    """
    df = market_df.copy()

    # Apply timezone lag: shift market data forward so T-day macro
    # is only available on T+lag day for crypto
    if lag_days > 0:
        df.index = df.index + pd.Timedelta(days=lag_days)
        logger.info(f"  {name}: applied {lag_days}-day forward shift (timezone lag)")

    # Reindex to crypto daily calendar with forward-fill
    aligned = df.reindex(crypto_index, method="ffill")

    # Logging: how many gaps were filled
    original_non_nan = df.reindex(crypto_index).notna().sum().sum()
    aligned_non_nan = aligned.notna().sum().sum()
    n_filled = aligned_non_nan - original_non_nan
    n_still_nan = aligned.isna().sum().sum()
    logger.info(f"  {name}: forward-filled {n_filled} gaps, {n_still_nan} NaN remaining")

    return aligned


def align_fred_monthly_to_daily(
    monthly_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align monthly FRED data to daily using release-date logic.
    Each value only becomes 'available' after its estimated release date.
    """
    aligned_frames = {}

    for col in monthly_df.columns:
        series = monthly_df[col].dropna()
        delay_days = FRED_RELEASE_DELAYS.get(col, 40)

        release_dates = series.index + pd.Timedelta(days=delay_days)
        shifted = pd.Series(series.values, index=release_dates, name=col)
        daily_series = shifted.reindex(daily_index, method="ffill")
        aligned_frames[col] = daily_series

        logger.info(f"  {col}: {len(series)} monthly -> daily, delay={delay_days}d")

    return pd.DataFrame(aligned_frames)


def create_aligned_dataset(
    price_df: pd.DataFrame,
    macro_dfs: dict[str, pd.DataFrame],
    fred_monthly: pd.DataFrame | None = None,
    coin_name: str = "btc",
    save: bool = True,
) -> pd.DataFrame:
    """
    Create a fully aligned dataset for one coin.

    Combines:
      - Crypto OHLCV (24/7 daily)
      - All macro categories (forward-filled + timezone-lagged)
      - (Optional) FRED monthly — release-date-shifted, forward-filled

    Parameters
    ----------
    price_df : pd.DataFrame
        Crypto OHLCV data with DatetimeIndex.
    macro_dfs : dict[str, pd.DataFrame]
        Dict of macro category DataFrames.
        Keys like 'risk', 'commodities', 'yields', 'credit'.
    fred_monthly : pd.DataFrame or None
        Optional monthly FRED data.
    coin_name : str
        Coin identifier for file naming ('btc', 'eth').
    save : bool
        Whether to save the result to CSV.
    """
    config = cfg()
    root = get_project_root()
    lag_days = config["data"].get("macro_lag_days", 1)
    crypto_index = price_df.index

    logger.info(f"Aligning data for {coin_name.upper()}: {len(crypto_index)} trading days, lag={lag_days}d")

    # Start with OHLCV
    aligned = price_df.copy()

    # Align each macro category with timezone lag
    for category_name, macro_df in macro_dfs.items():
        if macro_df is not None and not macro_df.empty:
            cat_aligned = align_market_data_to_crypto(
                macro_df, crypto_index,
                name=category_name,
                lag_days=lag_days,
            )
            aligned = aligned.join(cat_aligned, how="left")

    # Align FRED monthly (if available)
    if fred_monthly is not None and not fred_monthly.empty:
        fred_aligned = align_fred_monthly_to_daily(fred_monthly, crypto_index)
        aligned = aligned.join(fred_aligned, how="left")
        logger.info(f"  FRED monthly aligned: {list(fred_aligned.columns)}")

    # Report NaN before dropping
    nan_per_col = aligned.isna().sum()
    nan_cols = nan_per_col[nan_per_col > 0]
    if len(nan_cols) > 0:
        logger.info(f"  NaN per column before dropna:\n{nan_cols}")

    rows_before = len(aligned)
    aligned = aligned.dropna()
    rows_after = len(aligned)
    rows_lost = rows_before - rows_after

    pct_lost = (rows_lost / rows_before * 100) if rows_before > 0 else 0
    logger.info(
        f"  Dropped {rows_lost} rows ({rows_before} -> {rows_after}, "
        f"lost {pct_lost:.1f}%)"
    )
    logger.info(f"  Final columns ({len(aligned.columns)}): {list(aligned.columns)}")

    if save:
        out_path = root / config["paths"]["data_processed"] / f"{coin_name}_aligned.csv"
        save_csv(aligned, out_path)
        logger.info(f"  Saved: {out_path}")

    return aligned
