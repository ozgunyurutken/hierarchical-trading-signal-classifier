"""
V5 Faz 1.4 — Dataset orchestrator.

Build BTC + ETH aligned datasets (~22 columns) from yfinance + FRED for
the 2018-01-01 → 2025-12-31 V5 window (macro warm-up from 2017-01-01).

Outputs:
  data/processed/btc_aligned_v5.csv
  data/processed/eth_aligned_v5.csv
  data/raw/v5_*.csv  (raw price + macro snapshots)

Reuses existing config-driven src/data modules:
  src.data.price_collector   (yfinance OHLCV)
  src.data.macro_collector   (yfinance macro + FRED daily/monthly)
  src.data.data_aligner      (alignment + forward-fill + release lag)

V5-specific overrides via config.yaml > data block.
"""
from __future__ import annotations

import os
import json
import sys
import urllib.request
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.macro_collector import _download_ticker_group
from src.data.data_aligner import create_aligned_dataset
from src.data.price_collector import download_price_data
from src.utils.config import cfg
from src.utils.helpers import save_csv, setup_logger

logger = setup_logger(__name__)
config = cfg()

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY env var required", file=sys.stderr)
    sys.exit(1)


def fetch_fred_series(series_id: str, start: str, end: str) -> pd.Series:
    """REST API fetch, fredapi-free (works in sandboxes without optional deps)."""
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


def collect_macro(start: str, end: str) -> dict[str, pd.DataFrame]:
    print("\n[macro] Downloading 7 yfinance macro tickers")
    out = {}
    for cat_name, tickers in [
        ("risk", config["data"]["macro_risk"]),
        ("commodities", config["data"]["macro_commodities"]),
        ("yields", config["data"]["macro_yields"]),
    ]:
        df = _download_ticker_group(tickers, start, end, group_name=cat_name)
        print(f"  {cat_name:12s}  {df.shape}  {df.index.min().date()} → {df.index.max().date()}")
        out[cat_name] = df

    print("\n[macro] FRED daily DGS2 (US2Y)")
    dgs2 = fetch_fred_series("DGS2", start, end)
    print(f"  DGS2: {len(dgs2)} obs, {dgs2.index.min().date()} → {dgs2.index.max().date()}")
    out["yields"] = out["yields"].join(dgs2.rename("US2Y"), how="left")

    print("\n[macro] FRED monthly + weekly (4 series including M2)")
    fred_monthly_dict = {}
    for sid in ["FEDFUNDS", "CPIAUCSL", "UNRATE", "WM2NS"]:
        s = fetch_fred_series(sid, start, end)
        fred_monthly_dict[sid] = s
        print(f"  {sid:9s}  {len(s)} obs")
    fred_monthly = pd.DataFrame(fred_monthly_dict)
    return out, fred_monthly


def main() -> None:
    print("=" * 70)
    print("V5 Faz 1.4 — Dataset orchestrator")
    print("=" * 70)

    macro_start = config["data"]["macro_start_date"]
    end_date = config["data"]["end_date"]

    # ---------- 1. Macro data ----------
    macro_dfs, fred_monthly = collect_macro(macro_start, end_date)

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for cat_name, df in macro_dfs.items():
        save_csv(df, raw_dir / f"v5_macro_{cat_name}.csv")
    save_csv(fred_monthly, raw_dir / "v5_macro_fred_monthly.csv")

    # ---------- 2. Price data per coin ----------
    aligned_outputs = {}
    for coin, cfg_block in config["data"]["symbols"].items():
        coin_label = cfg_block["label"].lower()
        coin_start = cfg_block["start_date"]
        print(f"\n[price] {coin_label.upper()}: {coin_start} → {end_date}")
        price_df = download_price_data(coin, coin_start, end_date)
        save_csv(price_df, raw_dir / f"v5_price_{coin_label}.csv")

        # ---------- 3. Align ----------
        print(f"\n[align] {coin_label.upper()}")
        aligned = create_aligned_dataset(
            price_df=price_df,
            macro_dfs=macro_dfs,
            fred_monthly=fred_monthly,
            coin_name=f"{coin_label}_v5",
            save=False,  # we save explicitly with v5 suffix
        )
        out_path = PROJECT_ROOT / "data" / "processed" / f"{coin_label}_aligned_v5.csv"
        save_csv(aligned, out_path)
        aligned_outputs[coin_label] = aligned

    # ---------- 4. Sanity summary ----------
    print("\n" + "=" * 70)
    print("V5 Aligned Dataset Summary")
    print("=" * 70)
    for coin, df in aligned_outputs.items():
        print(f"\n{coin.upper()}_aligned_v5.csv")
        print(f"  shape: {df.shape}")
        print(f"  range: {df.index.min().date()} → {df.index.max().date()}")
        print(f"  cols ({len(df.columns)}): {list(df.columns)}")
        n_train = int(len(df) * config["training"]["train_size"])
        n_val = int(len(df) * config["training"]["validation_size"])
        n_test = len(df) - n_train - n_val
        train_end = df.index[n_train - 1]
        val_end = df.index[n_train + n_val - 1]
        print(f"  Split 70/15/15:")
        print(f"    train: {df.index[0].date()} → {train_end.date()}  ({n_train} days)")
        print(f"    val:   {df.index[n_train].date()} → {val_end.date()}  ({n_val} days)")
        print(f"    test:  {df.index[n_train + n_val].date()} → {df.index[-1].date()}  ({n_test} days)")

    print("\nV5 Faz 1.4 complete. Run notebook v5_01_eda.ipynb for review.")


if __name__ == "__main__":
    main()
