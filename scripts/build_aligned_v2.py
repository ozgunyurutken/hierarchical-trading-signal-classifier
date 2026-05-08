"""
v2/bigger-dataset orchestrator: build btc_aligned_v2.csv that starts
2013-01-01 (vs v1 start 2014-09-17).

Inputs:
  - data/raw/btc_extended_history.csv  (CryptoCompare 2010 + yfinance,
    written by scripts/fetch_cryptocompare_btc.py)
  - yfinance live: SP500, VIX, DXY, Gold, Silver, Oil, ^TNX, HY, IG,
                   Treasury20Y, TIPS  (downloaded fresh)
  - FRED API: DGS2 (daily 2Y yield) + 5 monthly (FEDFUNDS, CPIAUCSL,
              UNRATE, WM2NS, ICSA)

Output: data/processed/btc_aligned_v2.csv
        (~4750 rows × 22 cols, 2013-01-01 → 2025-12-30)

Cuts off the BTC series at 2013-01-01 — although CryptoCompare goes
back to 2010-07, the FRED daily DGS2 only starts 2013-01-02, so the
oldest dates would carry a NaN block. 2013 also marks the inflection
point where BTC daily USD volume crossed $1M (median).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import urllib.request
import urllib.parse
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.utils.config import cfg
from src.utils.helpers import save_csv

START = "2013-01-01"
END = "2025-12-31"
FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: set FRED_API_KEY", file=sys.stderr); sys.exit(1)

config = cfg()


# ---------- yfinance via the macro_collector helpers ----------
from src.data.macro_collector import _download_ticker_group


def fetch_yfinance(start: str, end: str) -> dict[str, pd.DataFrame]:
    print("\n[2] yfinance: 12 macro tickers (slow, expect ~30-60s)")
    out = {}
    for cat_name, tickers in [
        ("risk", config["data"]["macro_risk"]),
        ("commodities", config["data"]["macro_commodities"]),
        ("yields", config["data"]["macro_yields"]),
        ("credit", config["data"]["macro_credit"]),
    ]:
        df = _download_ticker_group(tickers, start, end, group_name=cat_name)
        print(f"  {cat_name:12s}  {df.shape}  {df.index.min().date()} → {df.index.max().date()}")
        out[cat_name] = df
    return out


# ---------- FRED daily DGS2 ----------
def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    qs = urllib.parse.urlencode({
        "series_id": series_id, "api_key": FRED_API_KEY, "file_type": "json",
        "observation_start": start, "observation_end": end,
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{qs}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read())
    df = pd.DataFrame(payload["observations"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].sort_index().dropna()
    s.name = series_id
    return s


# ---------- main ----------
def main() -> None:
    print(f"=== Build aligned_v2  ({START} → {END}) ===")

    print("\n[1] Load CryptoCompare extended BTC OHLCV, slice from START")
    btc_path = PROJECT_ROOT / "data" / "raw" / "btc_extended_history.csv"
    btc_full = pd.read_csv(btc_path, index_col=0, parse_dates=True)
    btc = btc_full.loc[START:END].copy()
    print(f"  BTC: {btc.shape}  {btc.index.min().date()} → {btc.index.max().date()}")

    yf_macro = fetch_yfinance(START, END)

    print("\n[3] FRED daily DGS2 (US2Y)")
    dgs2 = fetch_fred_series("DGS2", START, END)
    print(f"  DGS2: {len(dgs2)} obs, {dgs2.index.min().date()} → {dgs2.index.max().date()}")
    yf_macro["yields"] = yf_macro["yields"].join(dgs2.rename("US2Y"), how="left")
    print(f"  yields after DGS2 join: {yf_macro['yields'].shape}")

    print("\n[4] FRED monthly (5 series)")
    fred_monthly_dict = {}
    for sid in ["FEDFUNDS", "CPIAUCSL", "UNRATE", "WM2NS", "ICSA"]:
        s = fetch_fred_series(sid, START, END)
        fred_monthly_dict[sid] = s
        print(f"  {sid:9s}  {len(s)} obs, {s.index.min().date()} → {s.index.max().date()}")
    fred_monthly = pd.DataFrame(fred_monthly_dict)

    print("\n[5] data_aligner.create_aligned_dataset")
    from src.data.data_aligner import create_aligned_dataset
    aligned = create_aligned_dataset(
        price_df=btc,
        macro_dfs=yf_macro,
        fred_monthly=fred_monthly,
        coin_name="btc_v2",
        save=False,
    )
    print(f"  aligned_v2: {aligned.shape}, {aligned.index.min().date()} → {aligned.index.max().date()}")
    print(f"  cols: {list(aligned.columns)}")

    out = PROJECT_ROOT / "data" / "processed" / "btc_aligned_v2.csv"
    save_csv(aligned, out)
    print(f"\n  saved: {out.relative_to(PROJECT_ROOT)}")

    # Compare vs v1
    v1 = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                     index_col=0, parse_dates=True)
    print(f"\n[6] v1 vs v2 sanity:")
    print(f"  v1: {v1.shape}, {v1.index.min().date()} → {v1.index.max().date()}")
    print(f"  v2: {aligned.shape}, {aligned.index.min().date()} → {aligned.index.max().date()}")
    print(f"  v2 - v1 row delta: +{len(aligned) - len(v1)}")

    # Last 3 days head-to-head
    common_tail = aligned.index.intersection(v1.index)
    if len(common_tail) > 0:
        last5 = common_tail[-5:]
        diff = (aligned.loc[last5, "Close"] - v1.loc[last5, "Close"]).abs()
        print(f"  Close-price diff on last 5 common days: max={diff.max():.4f}, "
              f"mean={diff.mean():.4f}")


if __name__ == "__main__":
    main()
