"""
Macroeconomic data collection.

Primary source: yfinance (no API key needed).
Optional: FRED API via fredapi (requires FRED_API_KEY env var).

Collects:
  - Daily market indicators: S&P 500, Gold, DXY, VIX
  - Rate environment proxies: US 10Y/5Y/3M Treasury Yields
  - (Optional) Monthly FRED data: FFR, CPI, Unemployment
"""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path

from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger, save_csv

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Daily macro data (yfinance — always available)
# ---------------------------------------------------------------------------

def download_daily_macro(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download daily market macro indicators from yfinance.
    Includes S&P 500, Gold, DXY, VIX.
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
                series.index = pd.to_datetime(series.index).tz_localize(None)
                frames[series.name] = series
            else:
                logger.warning(f"  No data for {ticker}")
        except Exception as e:
            logger.error(f"  Failed to download {ticker}: {e}")

    if not frames:
        raise ValueError("No daily macro data could be downloaded")

    df = pd.DataFrame(frames)
    df.index.name = "Date"

    logger.info(f"Daily macro: {len(df)} rows, columns={list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Yield proxies (yfinance — always available, replaces FRED rate data)
# ---------------------------------------------------------------------------

def download_yield_proxies(
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download Treasury yield data from yfinance as rate-environment proxies.
    These replace FRED's Federal Funds Rate when no API key is available.

    Returns daily data: US 10Y, 5Y, 3M yields.
    """
    config = cfg()
    start = start_date or config["data"]["start_date"]
    end = end_date or config["data"]["end_date"]

    tickers = config["data"]["macro_yield_proxies"]
    frames = {}

    for ticker, name in tickers.items():
        logger.info(f"Downloading yield proxy: {ticker} ({name})")
        try:
            data = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
            if not data.empty:
                series = data["Close"].copy()
                series.name = name
                series.index = pd.to_datetime(series.index).tz_localize(None)
                frames[name] = series
            else:
                logger.warning(f"  No data for {ticker}")
        except Exception as e:
            logger.error(f"  Failed: {ticker}: {e}")

    if not frames:
        logger.warning("No yield proxy data downloaded")
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "Date"

    # Derived: yield curve spread (10Y - 3M) — classic recession indicator
    if "US10Y_Yield" in df.columns and "US3M_Yield" in df.columns:
        df["Yield_Spread_10Y_3M"] = df["US10Y_Yield"] - df["US3M_Yield"]

    logger.info(f"Yield proxies: {len(df)} rows, columns={list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# FRED monthly data (optional — requires fredapi + API key)
# ---------------------------------------------------------------------------

def download_fred_monthly(
    api_key: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame | None:
    """
    Download monthly data from FRED. Returns None if unavailable.
    """
    config = cfg()
    start = start_date or config["data"]["start_date"]
    end = end_date or config["data"]["end_date"]
    key = api_key or os.environ.get("FRED_API_KEY")

    if key is None:
        logger.info("No FRED API key — skipping FRED monthly data (yield proxies used instead)")
        return None

    try:
        from fredapi import Fred
    except ImportError:
        logger.info("fredapi not installed — skipping FRED monthly data")
        return None

    fred = Fred(api_key=key)
    series_ids = config["data"]["macro_monthly_fred"]
    frames = {}

    for series_id, description in series_ids.items():
        logger.info(f"Downloading FRED: {series_id} ({description})")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = series_id
            frames[series_id] = s
        except Exception as e:
            logger.error(f"  Failed {series_id}: {e}")

    if not frames:
        return None

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    logger.info(f"FRED monthly: {len(df)} rows, columns={list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def collect_all_macro(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Download all macro data.

    Returns dict with keys:
      'daily'  — S&P 500, Gold, DXY, VIX (daily)
      'yields' — US 10Y, 5Y, 3M yields + spread (daily)
      'fred'   — FFR, CPI, Unemployment (monthly, or None if unavailable)
    """
    config = cfg()
    root = get_project_root()
    results = {}

    # 1. Daily market macro
    daily = download_daily_macro()
    results["daily"] = daily
    if save:
        out = root / config["paths"]["data_raw"] / "macro_daily.csv"
        save_csv(daily, out)
        logger.info(f"Saved: {out}")

    # 2. Yield proxies
    yields = download_yield_proxies()
    results["yields"] = yields
    if save and not yields.empty:
        out = root / config["paths"]["data_raw"] / "macro_yields.csv"
        save_csv(yields, out)
        logger.info(f"Saved: {out}")

    # 3. FRED monthly (optional)
    fred = download_fred_monthly()
    results["fred"] = fred
    if save and fred is not None:
        out = root / config["paths"]["data_raw"] / "macro_fred_monthly.csv"
        save_csv(fred, out)
        logger.info(f"Saved: {out}")

    return results
