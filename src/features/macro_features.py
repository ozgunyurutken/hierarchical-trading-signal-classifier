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


def compute_derived_spreads(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived spread / ratio features that capture cross-asset relationships.

    Each spread is economically meaningful for risk-on/risk-off regime detection:
        - Yield_Curve_10Y_2Y: positive in expansion, inverts before recession
        - Credit_Spread:      log(HY) - log(IG), widens in stress
        - Gold_Silver_Ratio:  high in safe-haven flight (gold demand > silver industrial)
        - SP500_VIX_ratio:    risk appetite proxy (price up & vol down = bullish)

    Returns DataFrame with only derived columns (caller joins with rest).
    """
    derived = pd.DataFrame(index=aligned_df.index)
    cols = set(aligned_df.columns)

    if "US10Y" in cols and "US2Y" in cols:
        derived["macro_Yield_Curve_10Y_2Y"] = aligned_df["US10Y"] - aligned_df["US2Y"]

    if "HY_Bond" in cols and "IG_Bond" in cols:
        # Log spread is more stationary than raw price ratio
        derived["macro_Credit_Spread_log"] = (
            np.log(aligned_df["HY_Bond"]) - np.log(aligned_df["IG_Bond"])
        )

    if "Gold" in cols and "Silver" in cols:
        derived["macro_Gold_Silver_Ratio"] = aligned_df["Gold"] / aligned_df["Silver"]

    if "SP500" in cols and "VIX" in cols:
        derived["macro_SP500_VIX_ratio"] = aligned_df["SP500"] / aligned_df["VIX"]

    if derived.empty:
        logger.warning("No derived spreads could be computed (missing source columns)")
    else:
        logger.info(f"Computed {len(derived.columns)} derived spread features")

    return derived


def compute_macro_features(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all macro features from an aligned dataset.

    Pipeline:
        1. Raw macro levels
        2. Per-column transformations: rate-of-change, rolling z-score, rolling mean
        3. Derived cross-asset spreads (yield curve, credit, gold/silver, sp500/vix)
        4. Z-score transformations of the derived spreads as well

    Parameters
    ----------
    aligned_df : pd.DataFrame
        Aligned dataset containing macro columns.

    Returns
    -------
    pd.DataFrame
        All macro features (raw + transformations + derived).
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

        roc = compute_rate_of_change(series, [5, 20, 50])
        features = features.join(roc)

        zscore = compute_rolling_zscore(series, rolling_windows)
        features = features.join(zscore)

        rmean = compute_rolling_mean(series, rolling_windows)
        features = features.join(rmean)

    # Derived cross-asset spreads
    derived = compute_derived_spreads(aligned_df)
    features = features.join(derived)

    # Z-score the derived spreads (these are typically the most predictive)
    for col in derived.columns:
        series = derived[col]
        series.name = col  # already prefixed with 'macro_'
        zscore = compute_rolling_zscore(series, rolling_windows)
        features = features.join(zscore)

    # Optional FRED-derived: real interest rate (only if monthly FRED data was added)
    if "FEDFUNDS" in macro_cols and "CPIAUCSL" in macro_cols:
        cpi_annual_change = aligned_df["CPIAUCSL"].pct_change(periods=252) * 100
        features["macro_real_interest_rate"] = aligned_df["FEDFUNDS"] - cpi_annual_change

    logger.info(f"Computed {len(features.columns)} macro features total")
    return features
