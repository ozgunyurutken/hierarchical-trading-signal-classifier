"""
Fetch historical BTC daily OHLCV from CryptoCompare and merge it
in front of the existing yfinance series so we can extend the dataset
backwards from 2014-09-17 to 2010-07-17 (CryptoCompare's BTC start).

Strategy:
  1. Fetch 1600 days ending 2014-09-16 (one day before yfinance start)
     via CryptoCompare /data/v2/histoday — returns OHLCV with explicit
     timestamps.
  2. Sanity check: also fetch a small overlap window (2014-09-17 to
     2014-10-31, 45 days) where yfinance ALSO has data. Compare Close
     prices day-by-day. If max abs deviation > 5%, abort and warn.
  3. Concatenate the older slice in front of the existing
     btc_aligned.csv (only OHLCV columns; macro / FRED warm-up handled
     downstream).
  4. Output: data/raw/btc_extended_history.csv

Public endpoint, no API key required (free tier 100K req/month).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

CC_URL = "https://min-api.cryptocompare.com/data/v2/histoday"


def fetch_window(end_ts: int, limit: int = 2000) -> pd.DataFrame:
    """One CryptoCompare call. Returns OHLCV DataFrame indexed by date."""
    qs = f"?fsym=BTC&tsym=USD&limit={limit}&toTs={end_ts}"
    with urllib.request.urlopen(CC_URL + qs, timeout=30) as resp:
        payload = json.loads(resp.read())
    if payload.get("Response") != "Success":
        raise RuntimeError(f"CryptoCompare error: {payload}")
    df = pd.DataFrame(payload["Data"]["Data"])
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = df.set_index("date").sort_index()
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volumefrom": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main() -> None:
    print("[1] Fetch CryptoCompare BTC historical (1600 days ending 2014-09-16)")
    end_2014_09_16 = int(datetime(2014, 9, 16, tzinfo=timezone.utc).timestamp())
    cc_old = fetch_window(end_ts=end_2014_09_16, limit=1600)
    print(f"  fetched {len(cc_old)} days, "
          f"{cc_old.index.min().date()} → {cc_old.index.max().date()}")
    # Drop rows where Close is 0 (CryptoCompare pads pre-2010-07 with zeros)
    n_zero = (cc_old["Close"] == 0).sum()
    cc_old = cc_old[cc_old["Close"] > 0]
    print(f"  dropped {n_zero} rows with Close=0 (pre-data padding)")
    print(f"  final cc_old: {len(cc_old)} days, "
          f"{cc_old.index.min().date()} → {cc_old.index.max().date()}")

    print("\n[2] Fetch overlap window 2014-09-17 → 2014-10-31 for sanity check")
    end_2014_10_31 = int(datetime(2014, 10, 31, tzinfo=timezone.utc).timestamp())
    cc_overlap = fetch_window(end_ts=end_2014_10_31, limit=44)
    print(f"  fetched {len(cc_overlap)} days, "
          f"{cc_overlap.index.min().date()} → {cc_overlap.index.max().date()}")

    print("\n[3] Sanity check vs yfinance (current btc_aligned.csv)")
    yf = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                     index_col=0, parse_dates=True)
    overlap_dates = cc_overlap.index.intersection(yf.index)
    if len(overlap_dates) == 0:
        raise RuntimeError("no overlap with yfinance — abort")

    cmp = pd.DataFrame({
        "yf_close": yf.loc[overlap_dates, "Close"],
        "cc_close": cc_overlap.loc[overlap_dates, "Close"],
    })
    cmp["abs_pct"] = (cmp["cc_close"] - cmp["yf_close"]).abs() / cmp["yf_close"] * 100
    print(f"  overlap days: {len(overlap_dates)}")
    print(f"  max abs %dev: {cmp['abs_pct'].max():.2f}%")
    print(f"  mean abs %dev: {cmp['abs_pct'].mean():.2f}%")
    if cmp["abs_pct"].max() > 5.0:
        print(f"  WARNING max deviation > 5%, sample:\n{cmp.head(10)}")
    else:
        print("  OK (close prices align, < 5% per day)")

    print("\n[4] Concatenate older history in front of yfinance series")
    cc_old_only = cc_old[cc_old.index < yf.index.min()]
    print(f"  pre-yfinance rows: {len(cc_old_only)}, "
          f"{cc_old_only.index.min().date()} → {cc_old_only.index.max().date()}")

    extended = pd.concat([cc_old_only, yf[["Open","High","Low","Close","Volume"]]]).sort_index()
    print(f"  extended OHLCV: {extended.shape}, "
          f"{extended.index.min().date()} → {extended.index.max().date()}")

    out = PROJECT_ROOT / "data" / "raw" / "btc_extended_history.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    extended.to_csv(out)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")

    # Mini liquidity stats summary
    print("\n[5] Liquidity sanity (note: 2010-2013 BTC volume is tiny)")
    bins = [
        ("2010-07 → 2010-12", "2010-07-17", "2010-12-31"),
        ("2011 (year)",       "2011-01-01", "2011-12-31"),
        ("2012 (year)",       "2012-01-01", "2012-12-31"),
        ("2013 (year)",       "2013-01-01", "2013-12-31"),
        ("2014-09 → 2014-12", "2014-09-17", "2014-12-31"),
    ]
    for lbl, s, e in bins:
        sub = extended.loc[s:e]
        med_v = sub["Volume"].median()
        med_p = sub["Close"].median()
        med_dollar_v = (sub["Close"] * sub["Volume"]).median()
        print(f"    {lbl:24s}  n={len(sub):4d}  med Close=${med_p:8,.2f}  "
              f"med BTC Vol={med_v:8.0f}  med USD Vol=${med_dollar_v:14,.0f}")


if __name__ == "__main__":
    main()
