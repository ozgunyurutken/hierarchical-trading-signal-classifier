"""
Macroeconomic data collection.

Primary source: yfinance (no API key needed).
Optional: FRED API via fredapi (requires FRED_API_KEY env var).

Collects 4 categories of daily macro data:
  1. Risk Appetite & Volatility:  S&P 500, VIX, DXY
  2. Commodities & Store of Value: Gold, Silver, Oil WTI
  3. Bond Yields (rate environment): US 10Y, 5Y, 3M, 30Y, 2Y
  4. Credit & Inflation Proxies:   HYG, LQD, TLT, TIP

Optional monthly FRED data: FFR, CPI, Unemployment, M2, Jobless Claims
"""

import os
import pandas as pd
import yfinance as yf
from pathlib import Path

from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger, save_csv

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Generic yfinance downloader for a ticker group
# ---------------------------------------------------------------------------

def _download_ticker_group(
    tickers: dict[str, str],
    start_date: str,
    end_date: str,
    group_name: str = "macro",
) -> pd.DataFrame:
    """
    Download daily close prices for a group of yfinance tickers.

    Parameters
    ----------
    tickers : dict
        Mapping of yfinance ticker symbol to friendly column name.
        e.g. {"^GSPC": "SP500", "^VIX": "VIX"}
    start_date : str
        Start date YYYY-MM-DD.
    end_date : str
        End date YYYY-MM-DD.
    group_name : str
        Human-readable group name for logging.

    Returns
    -------
    pd.DataFrame
        DataFrame with Date index and one column per ticker.
    """
    frames = {}

    for ticker, col_name in tickers.items():
        logger.info(f"  [{group_name}] Downloading {ticker} → {col_name}")
        try:
            data = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1d")
            if not data.empty:
                series = data["Close"].copy()
                series.name = col_name
                series.index = pd.to_datetime(series.index).tz_localize(None)
                frames[col_name] = series
                logger.info(f"    ✓ {col_name}: {len(series)} rows ({series.index.min().date()} → {series.index.max().date()})")
            else:
                logger.warning(f"    ✗ No data for {ticker}")
        except Exception as e:
            logger.error(f"    ✗ Failed {ticker}: {e}")

    if not frames:
        logger.warning(f"  [{group_name}] No data downloaded for any ticker")
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Category downloaders
# ---------------------------------------------------------------------------

def download_macro_risk(start_date: str, end_date: str) -> pd.DataFrame:
    """Download risk appetite & volatility indicators: S&P 500, VIX, DXY."""
    tickers = cfg()["data"]["macro_risk"]
    return _download_ticker_group(tickers, start_date, end_date, "risk")


def download_macro_commodities(start_date: str, end_date: str) -> pd.DataFrame:
    """Download commodities & store of value: Gold, Silver, Oil WTI."""
    tickers = cfg()["data"]["macro_commodities"]
    return _download_ticker_group(tickers, start_date, end_date, "commodities")


def download_macro_yields(start_date: str, end_date: str) -> pd.DataFrame:
    """Download bond yields: US 10Y, 5Y, 3M, 30Y, 2Y."""
    tickers = cfg()["data"]["macro_yields"]
    return _download_ticker_group(tickers, start_date, end_date, "yields")


def download_macro_credit(start_date: str, end_date: str) -> pd.DataFrame:
    """Download credit & inflation proxies: HYG, LQD, TLT, TIP."""
    tickers = cfg()["data"]["macro_credit"]
    return _download_ticker_group(tickers, start_date, end_date, "credit")


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
    start = start_date or config["data"].get("macro_start_date", "2007-01-01")
    end = end_date or config["data"]["end_date"]
    key = api_key or os.environ.get("FRED_API_KEY")

    if key is None:
        logger.info("No FRED API key — skipping FRED monthly data")
        return None

    try:
        from fredapi import Fred
    except ImportError:
        logger.info("fredapi not installed — skipping FRED monthly data")
        return None

    fred = Fred(api_key=key)
    series_ids = config["data"].get("macro_fred_optional", {})
    frames = {}

    for series_id, description in series_ids.items():
        logger.info(f"  [FRED] Downloading {series_id} ({description})")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s.name = series_id
            frames[series_id] = s
        except Exception as e:
            logger.error(f"    ✗ Failed {series_id}: {e}")

    if not frames:
        return None

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    logger.info(f"  [FRED] Monthly: {len(df)} rows, columns={list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def collect_all_macro(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Download all macro data across 4 categories + optional FRED.

    Uses macro_start_date from config for maximum historical coverage.

    Returns dict with keys:
      'risk'        — S&P 500, VIX, DXY (daily)
      'commodities' — Gold, Silver, Oil WTI (daily)
      'yields'      — US 10Y, 5Y, 3M, 30Y, 2Y yields (daily)
      'credit'      — HYG, LQD, TLT, TIP (daily)
      'fred'        — FRED monthly (or None if unavailable)
    """
    config = cfg()
    root = get_project_root()
    start = config["data"]["macro_start_date"]
    end = config["data"]["end_date"]
    results = {}

    logger.info(f"=== Macro Data Collection: {start} → {end} ===")

    # 1. Risk Appetite & Volatility
    logger.info("Category 1/4: Risk Appetite & Volatility")
    risk = download_macro_risk(start, end)
    results["risk"] = risk
    if save and not risk.empty:
        out = root / config["paths"]["data_raw"] / "macro_risk.csv"
        save_csv(risk, out)
        logger.info(f"  Saved: {out}")

    # 2. Commodities & Store of Value
    logger.info("Category 2/4: Commodities & Store of Value")
    commodities = download_macro_commodities(start, end)
    results["commodities"] = commodities
    if save and not commodities.empty:
        out = root / config["paths"]["data_raw"] / "macro_commodities.csv"
        save_csv(commodities, out)
        logger.info(f"  Saved: {out}")

    # 3. Bond Yields
    logger.info("Category 3/4: Bond Yields")
    yields = download_macro_yields(start, end)
    results["yields"] = yields
    if save and not yields.empty:
        out = root / config["paths"]["data_raw"] / "macro_yields.csv"
        save_csv(yields, out)
        logger.info(f"  Saved: {out}")

    # 4. Credit & Inflation Proxies
    logger.info("Category 4/4: Credit & Inflation Proxies")
    credit = download_macro_credit(start, end)
    results["credit"] = credit
    if save and not credit.empty:
        out = root / config["paths"]["data_raw"] / "macro_credit.csv"
        save_csv(credit, out)
        logger.info(f"  Saved: {out}")

    # 5. FRED monthly (optional)
    logger.info("Optional: FRED Monthly Data")
    fred = download_fred_monthly()
    results["fred"] = fred
    if save and fred is not None:
        out = root / config["paths"]["data_raw"] / "macro_fred_monthly.csv"
        save_csv(fred, out)
        logger.info(f"  Saved: {out}")

    # Summary
    total_tickers = sum(
        len(df.columns) for key, df in results.items()
        if df is not None and not isinstance(df, type(None)) and hasattr(df, 'columns') and not df.empty
    )
    logger.info(f"=== Macro collection complete: {total_tickers} tickers downloaded ===")

    return results
