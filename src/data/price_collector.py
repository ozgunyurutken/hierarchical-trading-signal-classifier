"""
Price data collection from yfinance.
Downloads OHLCV data for BTC-USD and ETH-USD.

Each coin has its own start date configured in config.yaml:
  data.symbols.<SYMBOL>.start_date
"""

import pandas as pd
import yfinance as yf
from pathlib import Path

from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger, save_csv

logger = setup_logger(__name__)


def download_price_data(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Download OHLCV data for a single symbol from yfinance.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., "BTC-USD", "ETH-USD").
    start_date : str, optional
        Start date in YYYY-MM-DD format. Defaults to per-coin config value.
    end_date : str, optional
        End date in YYYY-MM-DD format. Defaults to config value.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    """
    config = cfg()

    # Per-coin start date from config (symbols is now a dict)
    if start_date is None:
        symbol_cfg = config["data"]["symbols"].get(symbol, {})
        start = symbol_cfg.get("start_date", "2014-01-01")
    else:
        start = start_date

    end = end_date or config["data"]["end_date"]

    logger.info(f"Downloading {symbol} from {start} to {end}")
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")

    if df.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Keep only OHLCV columns
    cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[cols].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    logger.info(f"  {symbol}: {len(df)} rows, {df.index.min()} to {df.index.max()}")

    # Check for missing days (crypto trades 24/7)
    date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    missing = date_range.difference(df.index)
    if len(missing) > 0:
        logger.warning(f"  {symbol}: {len(missing)} missing days detected")

    return df


def collect_all_prices(save: bool = True) -> dict[str, pd.DataFrame]:
    """
    Download price data for all configured symbols.
    Each symbol reads its own start_date from config.yaml.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of symbol name (e.g., 'btc', 'eth') to OHLCV DataFrame.
    """
    config = cfg()
    root = get_project_root()
    results = {}

    # config["data"]["symbols"] is now a dict: {"BTC-USD": {"start_date": ...}, ...}
    for symbol in config["data"]["symbols"]:
        df = download_price_data(symbol)
        name = symbol.split("-")[0].lower()  # "BTC-USD" -> "btc"
        results[name] = df

        if save:
            out_path = root / config["paths"]["data_raw"] / f"{name}_ohlcv.csv"
            save_csv(df, out_path)
            logger.info(f"  Saved to {out_path}")

    return results
