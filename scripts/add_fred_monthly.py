"""
Fetch 5 monthly/weekly FRED series and add them to btc_aligned.csv / eth_aligned.csv.

Series (config.yaml > macro_fred_optional):
  FEDFUNDS  — Federal Funds Rate         (monthly,   release lag  1d)
  CPIAUCSL  — CPI All Urban Consumers    (monthly,   release lag 45d)
  UNRATE    — Unemployment Rate          (monthly,   release lag 35d)
  WM2NS     — M2 Money Supply (weekly)   (weekly,    release lag 14d)
  ICSA      — Initial Jobless Claims     (weekly,    release lag  5d)

Each series is published at low frequency, then forward-filled onto the
crypto daily index with a per-series release lag (so day-T value reflects
what was actually known by day T, no look-ahead).

Inputs : data/processed/{btc,eth}_aligned.csv  (v3 — 17 cols)
Outputs: same paths, in-place rewrite (with backup), 22 cols
"""
from __future__ import annotations

import os
import sys
import shutil
import json
import datetime as _dt
import urllib.request
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: set FRED_API_KEY in env first", file=sys.stderr)
    sys.exit(1)

# matches src/data/data_aligner.py FRED_RELEASE_DELAYS
RELEASE_LAG_DAYS = {
    "FEDFUNDS": 1,
    "CPIAUCSL": 45,
    "UNRATE": 35,
    "WM2NS": 14,
    "ICSA": 5,
}


def fetch_fred(series_id: str, start: str = "2013-01-01", end: str = "2026-01-01") -> pd.Series:
    qs = urllib.parse.urlencode({
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{qs}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read())
    obs = payload["observations"]
    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].sort_index().dropna()
    s.name = series_id
    return s


def align_series(s: pd.Series, crypto_index: pd.DatetimeIndex, release_lag: int) -> pd.Series:
    # 1. Shift the (low-frequency) observation date forward by release_lag days
    #    so that the value first appears in the daily index on the date it would
    #    actually have been published.
    s_shifted = s.copy()
    s_shifted.index = s_shifted.index + pd.Timedelta(days=release_lag)
    # 2. Forward-fill onto the crypto daily index (24/7).
    return s_shifted.reindex(crypto_index, method="ffill")


def patch_aligned(path: Path, fetched: dict[str, pd.Series]) -> None:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    print(f"\n  {path.name}: {df.shape}, cols={list(df.columns)}")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    shutil.copy(path, backup)
    print(f"  Backup: {backup.name}")

    for series_id, s in fetched.items():
        if series_id in df.columns:
            print(f"  WARN {series_id} already in df — skipping")
            continue
        lag = RELEASE_LAG_DAYS[series_id]
        aligned = align_series(s, df.index, release_lag=lag)
        n_nan = aligned.isna().sum()
        df[series_id] = aligned
        print(f"  + {series_id:9s}  lag={lag:3d}d  NaN(warm-up)={n_nan:4d}  "
              f"min={aligned.min():.3f}  last={aligned.iloc[-1]:.3f}")

    n_before = len(df)
    df = df.dropna(subset=list(fetched.keys()))
    n_lost = n_before - len(df)
    if n_lost > 0:
        print(f"  Dropped {n_lost} rows that have any FRED warm-up NaN")
    df.to_csv(path)
    print(f"  Wrote {path.name}: {df.shape}")


def main() -> None:
    print("[1] Fetch 5 FRED series")
    fetched = {}
    for sid in RELEASE_LAG_DAYS.keys():
        s = fetch_fred(sid)
        print(f"  {sid:9s}: {len(s)} obs, {s.index.min().date()} → {s.index.max().date()}, "
              f"min={s.min():.3f}, max={s.max():.3f}")
        fetched[sid] = s

    print("\n[2] Patch BTC aligned")
    patch_aligned(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv", fetched)

    print("\n[3] Patch ETH aligned")
    patch_aligned(PROJECT_ROOT / "data" / "processed" / "eth_aligned.csv", fetched)

    print("\nFRED monthly add-on complete.")


if __name__ == "__main__":
    main()
