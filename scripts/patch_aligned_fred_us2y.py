"""
Patch the aligned dataset: replace ZT=F-derived `US2Y` with FRED DGS2 (real 2Y yield)
and drop US5Y / US3M / US30Y columns that were carried over from an older config.

Outputs: in-place rewrite of data/processed/btc_aligned.csv and eth_aligned.csv.
A backup with timestamp is saved alongside.

This script does NOT re-download the rest of the macro / OHLCV data — it patches
only the yield columns. Useful when yfinance is rate-limited but FRED works.
"""
from __future__ import annotations

import os
import sys
import shutil
import datetime as _dt
import urllib.request
import urllib.parse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

DROP_YIELDS = ["US5Y", "US3M", "US30Y"]
FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: set FRED_API_KEY in env first", file=sys.stderr)
    sys.exit(1)


def fetch_fred_daily(series_id: str, start: str = "2014-01-01", end: str = "2026-01-01") -> pd.Series:
    """Fetch a daily FRED series via REST (no fredapi dep)."""
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
    s = df.set_index("date")["value"].sort_index()
    s.name = series_id
    print(f"  FRED {series_id}: {len(s)} obs, {s.index.min().date()} → {s.index.max().date()}, "
          f"non-null={s.notna().sum()}")
    return s


def patch_aligned_csv(path: Path, fred_us2y: pd.Series, lag_days: int = 1) -> None:
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # Backup
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.backup_{ts}{path.suffix}")
    shutil.copy(path, backup)
    print(f"  Backup: {backup.name}")

    # Drop superseded yield columns
    dropped = [c for c in DROP_YIELDS if c in df.columns]
    if dropped:
        df = df.drop(columns=dropped)
        print(f"  Dropped: {dropped}")

    # Replace US2Y with FRED DGS2 — apply same +1 day lag the original aligner used
    if lag_days > 0:
        fred_shifted = fred_us2y.copy()
        fred_shifted.index = fred_shifted.index + pd.Timedelta(days=lag_days)
    else:
        fred_shifted = fred_us2y

    # Forward-fill onto the crypto daily index (matches data_aligner behaviour)
    fred_aligned = fred_shifted.reindex(df.index, method="ffill")
    n_nan = fred_aligned.isna().sum()
    print(f"  US2Y(DGS2) reindexed onto crypto calendar: {len(fred_aligned)} rows, "
          f"NaN={n_nan} (warm-up before DGS2 starts)")

    # Replace column
    df["US2Y"] = fred_aligned

    # Drop any rows where the new US2Y is still NaN (start-of-history warm-up)
    n_before = len(df)
    df = df.dropna(subset=["US2Y"])
    n_lost = n_before - len(df)
    if n_lost > 0:
        print(f"  Dropped {n_lost} rows where DGS2 hadn't started yet")

    df.to_csv(path)
    print(f"  Wrote {path.name}: {len(df)} rows × {len(df.columns)} cols")
    print(f"  US2Y now: min={df['US2Y'].min():.3f}, max={df['US2Y'].max():.3f}, "
          f"last={df['US2Y'].iloc[-1]:.3f}")


def main() -> None:
    print("[1] Fetch FRED DGS2 (US 2-Year Constant-Maturity Yield)")
    us2y = fetch_fred_daily("DGS2", start="2014-01-01", end="2026-01-01")

    print("\n[2] Patch BTC aligned")
    patch_aligned_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv", us2y)

    print("\n[3] Patch ETH aligned")
    patch_aligned_csv(PROJECT_ROOT / "data" / "processed" / "eth_aligned.csv", us2y)

    print("\n[4] Sanity: yield curve should now be in a sensible range")
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    yc = btc["US10Y"] - btc["US2Y"]
    print(f"  Yield_Curve_10Y_2Y: {yc.min():.2f} → {yc.max():.2f}  "
          f"(median {yc.median():.2f}, last {yc.iloc[-1]:.2f})")
    print("  Patch complete.")


if __name__ == "__main__":
    main()
