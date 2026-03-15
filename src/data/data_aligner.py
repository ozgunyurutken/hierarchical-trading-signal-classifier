"""
Data alignment module.
Aligns crypto OHLCV, daily macro, and monthly macro data on a common daily timeline.
Handles forward-fill, release-date alignment, and NaN cleanup.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger, save_csv

logger = setup_logger(__name__)

# Approximate FRED release delays (days after report period end)
# E.g., January CPI is released mid-February (~45 days after Jan 1)
FRED_RELEASE_DELAYS = {
    "FEDFUNDS": 1,      # Available almost immediately (daily effective rate)
    "CPIAUCSL": 45,     # Released ~15th of month+1 (so ~45 days from period start)
    "UNRATE": 35,       # Released ~first Friday of month+1
}


def align_macro_monthly_to_daily(
    monthly_df: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align monthly FRED data to daily frequency using release-date logic.

    Each monthly observation becomes available only after its estimated release date,
    then is forward-filled until the next release.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly macro data with DatetimeIndex (period start dates).
    daily_index : pd.DatetimeIndex
        Target daily date index to align to.

    Returns
    -------
    pd.DataFrame
        Daily-frequency macro data aligned by release dates.
    """
    aligned_frames = {}

    for col in monthly_df.columns:
        series = monthly_df[col].dropna()
        delay_days = FRED_RELEASE_DELAYS.get(col, 40)  # Default 40 days

        # Shift observation dates forward by release delay
        release_dates = series.index + pd.Timedelta(days=delay_days)
        shifted = pd.Series(series.values, index=release_dates, name=col)

        # Reindex to daily and forward-fill
        daily_series = shifted.reindex(daily_index, method="ffill")
        aligned_frames[col] = daily_series

        available_from = release_dates.min()
        logger.info(
            f"  {col}: {len(series)} monthly obs -> daily, "
            f"release delay={delay_days}d, first available={available_from.date()}"
        )

    return pd.DataFrame(aligned_frames)


def align_daily_macro(
    macro_daily: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Align daily market macro data (S&P, VIX, Gold, DXY) to crypto daily index.
    Forward-fills weekends and holidays from last trading day.
    """
    aligned = macro_daily.reindex(daily_index, method="ffill")

    missing_before = macro_daily.isna().sum().sum()
    missing_after = aligned.isna().sum().sum()
    logger.info(
        f"  Daily macro aligned: {len(aligned)} rows, "
        f"NaN before={missing_before}, NaN after ffill={missing_after}"
    )

    return aligned


def create_aligned_dataset(
    price_df: pd.DataFrame,
    macro_monthly: pd.DataFrame,
    macro_daily: pd.DataFrame,
    coin_name: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Create a fully aligned dataset combining price, daily macro, and monthly macro.

    Parameters
    ----------
    price_df : pd.DataFrame
        OHLCV data for a single coin.
    macro_monthly : pd.DataFrame
        Monthly FRED data.
    macro_daily : pd.DataFrame
        Daily market macro data.
    coin_name : str
        Name of the coin (e.g., 'btc', 'eth').
    save : bool
        Whether to save the result.

    Returns
    -------
    pd.DataFrame
        Aligned dataset with all columns.
    """
    config = cfg()
    root = get_project_root()

    daily_index = price_df.index
    logger.info(f"Aligning data for {coin_name.upper()}: {len(daily_index)} trading days")

    # Align monthly macro to daily
    monthly_aligned = pd.DataFrame()
    if not macro_monthly.empty:
        monthly_aligned = align_macro_monthly_to_daily(macro_monthly, daily_index)

    # Align daily macro
    daily_aligned = align_daily_macro(macro_daily, daily_index)

    # Combine
    aligned = price_df.copy()
    if not monthly_aligned.empty:
        aligned = aligned.join(monthly_aligned, how="left")
    aligned = aligned.join(daily_aligned, how="left")

    # Report NaN
    nan_counts = aligned.isna().sum()
    total_nans = nan_counts.sum()
    logger.info(f"  Pre-drop NaN count per column:\n{nan_counts[nan_counts > 0]}")

    rows_before = len(aligned)
    aligned = aligned.dropna()
    rows_after = len(aligned)
    rows_lost = rows_before - rows_after

    logger.info(
        f"  Dropped {rows_lost} rows with NaN "
        f"({rows_before} -> {rows_after}, lost {rows_lost/rows_before*100:.1f}%)"
    )

    if save:
        out_path = root / config["paths"]["data_processed"] / f"{coin_name}_aligned.csv"
        save_csv(aligned, out_path)
        logger.info(f"  Saved aligned data to {out_path}")

    return aligned
