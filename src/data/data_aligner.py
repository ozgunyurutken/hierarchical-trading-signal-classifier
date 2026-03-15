"""
Data alignment module.
Aligns crypto OHLCV with daily macro and yield proxy data on a common daily timeline.
Handles forward-fill for weekends/holidays and NaN cleanup.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger, save_csv

logger = setup_logger(__name__)

# Approximate FRED release delays (days after report period end)
FRED_RELEASE_DELAYS = {
    "FEDFUNDS": 1,
    "CPIAUCSL": 45,
    "UNRATE": 35,
}


def align_market_data_to_crypto(
    market_df: pd.DataFrame,
    crypto_index: pd.DatetimeIndex,
    name: str = "market",
) -> pd.DataFrame:
    """
    Align market data (weekday-only) to crypto daily index (24/7).
    Forward-fills weekends and holidays from last trading day.
    """
    aligned = market_df.reindex(crypto_index, method="ffill")

    original_non_nan = market_df.reindex(crypto_index).notna().sum().sum()
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
    macro_daily: pd.DataFrame,
    macro_yields: pd.DataFrame,
    fred_monthly: pd.DataFrame | None = None,
    coin_name: str = "btc",
    save: bool = True,
) -> pd.DataFrame:
    """
    Create a fully aligned dataset for one coin.

    Combines:
      - Crypto OHLCV (24/7 daily)
      - Market macro (S&P, Gold, DXY, VIX — forward-filled)
      - Yield proxies (10Y, 5Y, 3M, Spread — forward-filled)
      - (Optional) FRED monthly — release-date-shifted, forward-filled
    """
    config = cfg()
    root = get_project_root()
    crypto_index = price_df.index

    logger.info(f"Aligning data for {coin_name.upper()}: {len(crypto_index)} trading days")

    # Start with OHLCV
    aligned = price_df.copy()

    # Align daily macro
    if not macro_daily.empty:
        daily_aligned = align_market_data_to_crypto(macro_daily, crypto_index, "macro_daily")
        aligned = aligned.join(daily_aligned, how="left")

    # Align yield proxies
    if not macro_yields.empty:
        yields_aligned = align_market_data_to_crypto(macro_yields, crypto_index, "yields")
        aligned = aligned.join(yields_aligned, how="left")

    # Align FRED monthly (if available)
    if fred_monthly is not None and not fred_monthly.empty:
        fred_aligned = align_fred_monthly_to_daily(fred_monthly, crypto_index)
        aligned = aligned.join(fred_aligned, how="left")
        logger.info(f"  FRED monthly aligned")

    # Report NaN before dropping
    nan_per_col = aligned.isna().sum()
    nan_cols = nan_per_col[nan_per_col > 0]
    if len(nan_cols) > 0:
        logger.info(f"  NaN per column:\n{nan_cols}")

    rows_before = len(aligned)
    aligned = aligned.dropna()
    rows_after = len(aligned)
    rows_lost = rows_before - rows_after

    logger.info(
        f"  Dropped {rows_lost} rows ({rows_before} -> {rows_after}, "
        f"lost {rows_lost / rows_before * 100:.1f}%)"
    )

    if save:
        out_path = root / config["paths"]["data_processed"] / f"{coin_name}_aligned.csv"
        save_csv(aligned, out_path)
        logger.info(f"  Saved: {out_path}")

    return aligned
