"""
Stage 1 — Trend label generation.
Labels: Uptrend, Downtrend, Sideways
Based on SMA crossover rules with persistence filter.
"""

import pandas as pd
import numpy as np

from src.utils.config import cfg
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def generate_trend_labels(
    df: pd.DataFrame,
    sma_short: int | None = None,
    sma_long: int | None = None,
    min_persistence: int | None = None,
) -> pd.Series:
    """
    Generate trend labels based on SMA crossover rules.

    Rules:
        - Uptrend:   SMA(short) > SMA(long) AND Close > SMA(long)
        - Downtrend: SMA(short) < SMA(long) AND Close < SMA(long)
        - Sideways:  everything else

    Noise filter: Labels must persist for at least `min_persistence` days.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'Close' column and SMA columns, or raw OHLCV.
    sma_short : int
        Short SMA period. Default from config.
    sma_long : int
        Long SMA period. Default from config.
    min_persistence : int
        Minimum consecutive days for a label to hold.

    Returns
    -------
    pd.Series
        Trend labels indexed by date.
    """
    config = cfg()
    sma_short = sma_short or config["labels"]["trend"]["sma_short"]
    sma_long = sma_long or config["labels"]["trend"]["sma_long"]
    min_persistence = min_persistence or config["labels"]["trend"]["min_persistence_days"]

    close = df["Close"]

    # Compute SMAs if not present
    sma_s = close.rolling(window=sma_short).mean()
    sma_l = close.rolling(window=sma_long).mean()

    # Apply rules
    labels = pd.Series("Sideways", index=df.index, name="trend_label")
    labels[(sma_s > sma_l) & (close > sma_l)] = "Uptrend"
    labels[(sma_s < sma_l) & (close < sma_l)] = "Downtrend"

    # Drop rows where SMA is NaN (warm-up period)
    labels = labels[sma_l.notna()]

    # Apply persistence filter
    labels = _apply_persistence_filter(labels, min_persistence)

    # Report
    dist = labels.value_counts()
    logger.info(
        f"Trend labels generated: {len(labels)} rows, "
        f"SMA({sma_short}/{sma_long}), persistence={min_persistence}\n"
        f"  Distribution: {dist.to_dict()}"
    )

    return labels


def generate_trend_labels_zigzag(
    close: pd.Series,
    threshold: float = 0.10,
    sideways_band: float = 0.03,
) -> pd.Series:
    """
    Causal ZigZag-style trend labelling.

    State machine:
        - Up:       price keeps making higher pivots; reversal triggered if price
                    drops `threshold` from the pivot high.
        - Down:     mirror.
        - Sideways: bootstrapped initial state and any time the running pivot has
                    not moved more than `sideways_band` in either direction over
                    the last 20 bars (cheap bootstrap).

    The label at time t depends ONLY on close prices up to and including t — no
    look-ahead. This is critical: classical ZigZag implementations back-fill
    pivots once a swing is confirmed, which leaks future information into the
    label.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed chronologically.
    threshold : float
        Reversal trigger as a fraction of the pivot price (e.g., 0.10 = 10%).
    sideways_band : float
        If max(close[-20:]) / min(close[-20:]) - 1 < sideways_band, force Sideways.

    Returns
    -------
    pd.Series of {"Uptrend", "Downtrend", "Sideways"} indexed like `close`.
    """
    if len(close) == 0:
        return pd.Series([], dtype=object, name="trend_label")

    n = len(close)
    labels: list[str] = []
    state = "Sideways"
    pivot = float(close.iloc[0])

    for i, p in enumerate(close.values):
        p = float(p)

        if state == "Sideways":
            if p >= pivot * (1 + threshold):
                state = "Uptrend"
                pivot = p
            elif p <= pivot * (1 - threshold):
                state = "Downtrend"
                pivot = p
            else:
                # update pivot to running max in early phase
                pivot = max(pivot, p) if pivot < p else min(pivot, p) if pivot > p else pivot
        elif state == "Uptrend":
            if p > pivot:
                pivot = p
            elif p <= pivot * (1 - threshold):
                state = "Downtrend"
                pivot = p
        elif state == "Downtrend":
            if p < pivot:
                pivot = p
            elif p >= pivot * (1 + threshold):
                state = "Uptrend"
                pivot = p

        # Sideways override: tight 20-bar range ⇒ Sideways
        if i >= 20:
            window = close.iloc[i - 19 : i + 1]
            wmax, wmin = float(window.max()), float(window.min())
            if (wmax / max(wmin, 1e-12) - 1.0) < sideways_band:
                labels.append("Sideways")
                continue

        labels.append(state)

    out = pd.Series(labels, index=close.index, name="trend_label")
    dist = out.value_counts()
    logger.info(
        f"ZigZag labels: threshold={threshold:.0%}, sideways_band={sideways_band:.0%}, "
        f"dist={dist.to_dict()}"
    )
    return out


def _apply_persistence_filter(labels: pd.Series, min_days: int) -> pd.Series:
    """
    Remove label flips that don't persist for at least `min_days`.
    Reverts short-lived labels to the previous stable label.
    """
    if min_days <= 1:
        return labels

    filtered = labels.copy()
    values = filtered.values
    n = len(values)

    i = 0
    while i < n:
        # Find the start of a run
        current_label = values[i]
        run_start = i
        while i < n and values[i] == current_label:
            i += 1
        run_length = i - run_start

        # If run is too short and not at the start, revert to previous label
        if run_length < min_days and run_start > 0:
            prev_label = values[run_start - 1]
            values[run_start:i] = prev_label

    filtered = pd.Series(values, index=labels.index, name=labels.name)

    changes_before = (labels != labels.shift()).sum()
    changes_after = (filtered != filtered.shift()).sum()
    logger.info(
        f"  Persistence filter: label changes {changes_before} -> {changes_after}"
    )

    return filtered
