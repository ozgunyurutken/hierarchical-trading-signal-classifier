"""
Macroeconomic feature engineering.
Computes transformations on macro data: rate-of-change, rolling z-scores, derived features.
"""

import pandas as pd
import numpy as np

from src.utils.config import cfg
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def compute_rate_of_change(series: pd.Series, periods: list[int]) -> pd.DataFrame:
    """Compute rate of change for multiple periods."""
    features = pd.DataFrame(index=series.index)
    for n in periods:
        features[f"{series.name}_roc_{n}"] = series.pct_change(periods=n)
    return features


def compute_rolling_zscore(
    series: pd.Series,
    windows: list[int],
) -> pd.DataFrame:
    """Compute rolling z-score for multiple window sizes."""
    features = pd.DataFrame(index=series.index)
    for w in windows:
        rolling_mean = series.rolling(window=w).mean()
        rolling_std = series.rolling(window=w).std()
        features[f"{series.name}_zscore_{w}"] = (series - rolling_mean) / rolling_std
    return features


def compute_rolling_mean(
    series: pd.Series,
    windows: list[int],
) -> pd.DataFrame:
    """Compute rolling means for multiple windows."""
    features = pd.DataFrame(index=series.index)
    for w in windows:
        features[f"{series.name}_sma_{w}"] = series.rolling(window=w).mean()
    return features


def compute_macro_features(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all macro features from an aligned dataset.

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned dataset containing macro columns.

    Returns
    -------
    pd.DataFrame
        All macro features (raw + transformations).
    """
    config = cfg()
    rolling_windows = config["features"]["macro_transformations"]["rolling_windows"]

    # Identify macro columns (non-OHLCV columns)
    ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
    macro_cols = [c for c in aligned_df.columns if c not in ohlcv_cols]

    if not macro_cols:
        logger.warning("No macro columns found in aligned data")
        return pd.DataFrame(index=aligned_df.index)

    logger.info(f"Computing macro features for columns: {macro_cols}")

    features = pd.DataFrame(index=aligned_df.index)

    # Raw macro values
    for col in macro_cols:
        features[f"macro_{col}"] = aligned_df[col]

    # Transformations for each macro column
    for col in macro_cols:
        series = aligned_df[col]
        series.name = f"macro_{col}"

        # Rate of change
        roc = compute_rate_of_change(series, [5, 20, 50])
        features = features.join(roc)

        # Rolling z-score
        zscore = compute_rolling_zscore(series, rolling_windows)
        features = features.join(zscore)

        # Rolling mean
        rmean = compute_rolling_mean(series, rolling_windows)
        features = features.join(rmean)

    # Derived features
    if "FEDFUNDS" in macro_cols and "CPIAUCSL" in macro_cols:
        # Real interest rate = FFR - CPI annual change
        cpi_annual_change = aligned_df["CPIAUCSL"].pct_change(periods=252) * 100
        features["macro_real_interest_rate"] = aligned_df["FEDFUNDS"] - cpi_annual_change

    logger.info(f"Computed {len(features.columns)} macro features total")
    return features
