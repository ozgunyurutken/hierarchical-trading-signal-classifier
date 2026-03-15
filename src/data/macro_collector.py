"""
Macroeconomic data collection from FRED API and yfinance.
Collects both monthly (FRED) and daily (market) macro indicators.
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger, save_csv

logger = setup_logger(__name__)


def download_fred_data(
    api_key: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download monthly macroeconomic data from FRED.

    Parameters
    ----------
    api_key : str, optional
        FRED API key. If None, attempts to read from FRED_API_KEY env var.
    start_date : str, optional
        Start date. Defaults to config value.
    end_date : str, optional
        End date. Defaults to config value.

    Returns
    -------
    pd.DataFrame
        Monthly macro indicators with DatetimeIndex.
    """
    import os

    config = cfg()
    start = start_date or config["data"]["start_date"]
    end = end_date or config["data"]["end_date"]
    key = api_key or os.environ.get("FRED_API_KEY")

    if key is None:
        logger.warning(
            "No FRED API key provided. Set FRED_API_KEY env var or pass api_key. "
            "Falling back to yfinance proxies for macro data."
        )
        return _fallback_macro_monthly(start, end)

    from fredapi import Fred
    fred = Fred(api_key=key)

    series_ids = config["data"]["macro_monthly"]
    frames = {}

    for series_id, description in series_ids.items():
        logger.info(f"Downloading FRED series: {series_id} ({description})")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = series_id
            frames[series_id] = s
        except Exception as e:
            logger.error(f"  Failed to download {series_id}: {e}")

    if not frames:
        raise ValueError("No FRED data could be downloaded")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    logger.info(f"FRED monthly data: {len(df)} rows, {df.columns.tolist()}")
    return df


def _fallback_macro_monthly(start: str, end: str) -> pd.DataFrame:
    """
    Fallback: use yfinance proxies for macro indicators when FRED API is unavailable.
    Downloads monthly resampled data for approximate macro representation.
    """
    logger.info("Using fallback monthly macro proxies from yfinance")
    # We'll use Treasury yields and other proxies
    tickers = {
        "^TNX": "US10Y_Yield",   # 10-Year Treasury Yield (proxy for rates)
        "^IRX": "US3M_Yield",    # 3-Month Treasury Yield
    }

    frames = {}
    for ticker, name in tickers.items():
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
            if not data.empty:
                monthly = data["Close"].resample("MS").last()
                monthly.name = name
                frames[name] = monthly
        except Exception as e:
            logger.warning(f"  Fallback failed for {ticker}: {e}")

    if frames:
        df = pd.DataFrame(frames)
        df.index.name = "Date"
        return df

    return pd.DataFrame()


def download_daily_macro(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download daily market macro indicators from yfinance.

    Returns
    -------
    pd.DataFrame
        Daily macro indicators (S&P 500, Gold, DXY, VIX).
    """
    config = cfg()
    start = start_date or config["data"]["start_date"]
    end = end_date or config["data"]["end_date"]

    tickers = config["data"]["macro_daily"]
    frames = {}

    for ticker, description in tickers.items():
        logger.info(f"Downloading daily macro: {ticker} ({description})")
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
            if not data.empty:
                series = data["Close"].copy()
                series.name = description.replace(" ", "_")
                frames[series.name] = series
            else:
                logger.warning(f"  No data for {ticker}")
        except Exception as e:
            logger.error(f"  Failed to download {ticker}: {e}")

    if not frames:
        raise ValueError("No daily macro data could be downloaded")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    logger.info(f"Daily macro data: {len(df)} rows, {df.columns.tolist()}")
    return df


def collect_all_macro(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Download all macro data (monthly + daily).

    Returns
    -------
    dict with keys 'monthly' and 'daily'.
    """
    config = cfg()
    root = get_project_root()
    results = {}

    # Monthly FRED data
    try:
        monthly = download_fred_data()
        results["monthly"] = monthly
        if save:
            out_path = root / config["paths"]["data_raw"] / "macro_monthly.csv"
            save_csv(monthly, out_path)
            logger.info(f"  Saved monthly macro to {out_path}")
    except Exception as e:
        logger.error(f"Monthly macro collection failed: {e}")
        results["monthly"] = pd.DataFrame()

    # Daily market data
    daily = download_daily_macro()
    results["daily"] = daily
    if save:
        out_path = root / config["paths"]["data_raw"] / "macro_daily.csv"
        save_csv(daily, out_path)
        logger.info(f"  Saved daily macro to {out_path}")

    return results
