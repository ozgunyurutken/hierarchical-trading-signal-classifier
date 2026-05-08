"""
Generate one time-series PNG per data column for manual data-quality review.

Reads from data/processed/{btc,eth}_aligned.csv since the raw weekday-only files
are not in this workspace. The aligned files have forward-filled weekends/holidays
and a 1-day shift for macro data (NYSE close -> next-day crypto signal). Plots
are annotated accordingly so the reviewer knows what they are looking at.

Output: reports/raw_data_visuals/{asset_or_indicator}.png
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

OUT_DIR = PROJECT_ROOT / "reports" / "raw_data_visuals"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Annotated metadata: nicer titles, units, log/linear, and category for grouping
COL_META: dict[str, dict] = {
    # Price OHLCV (rendered as combined panel per coin)
    "Open": {"title": "Open", "unit": "USD"},
    "High": {"title": "High", "unit": "USD"},
    "Low": {"title": "Low", "unit": "USD"},
    "Close": {"title": "Close", "unit": "USD"},
    "Volume": {"title": "Volume", "unit": "shares/coins"},
    # Risk appetite
    "SP500": {"title": "S&P 500 (^GSPC)", "unit": "index", "category": "Risk", "log": False},
    "VIX": {"title": "Volatility Index (^VIX)", "unit": "%", "category": "Risk", "log": False},
    "DXY": {"title": "US Dollar Index (DX-Y.NYB)", "unit": "index", "category": "Risk", "log": False},
    # Commodities
    "Gold": {"title": "Gold Futures (GC=F)", "unit": "USD/oz", "category": "Commodity", "log": False},
    "Silver": {"title": "Silver Futures (SI=F)", "unit": "USD/oz", "category": "Commodity", "log": False},
    "Oil_WTI": {"title": "WTI Crude Oil (CL=F)", "unit": "USD/bbl", "category": "Commodity", "log": False},
    # Yields
    "US10Y": {"title": "US 10-Year Treasury Yield (^TNX)", "unit": "%", "category": "Yield", "log": False},
    "US5Y": {"title": "US 5-Year Treasury Yield", "unit": "%", "category": "Yield", "log": False},
    "US3M": {"title": "US 3-Month T-Bill Yield", "unit": "%", "category": "Yield", "log": False},
    "US30Y": {"title": "US 30-Year Treasury Yield", "unit": "%", "category": "Yield", "log": False},
    "US2Y": {"title": "US 2-Year Treasury Futures (ZT=F)", "unit": "price", "category": "Yield", "log": False},
    # Credit / inflation
    "HY_Bond": {"title": "High Yield Corporate Bonds (HYG)", "unit": "USD", "category": "Credit", "log": False},
    "IG_Bond": {"title": "Investment Grade Corporate Bonds (LQD)", "unit": "USD", "category": "Credit", "log": False},
    "Treasury20Y": {"title": "Long-Term Treasury (TLT)", "unit": "USD", "category": "Credit", "log": False},
    "TIPS": {"title": "Inflation-Protected Securities (TIP)", "unit": "USD", "category": "Credit", "log": False},
}

CATEGORY_COLOR = {
    "Risk": "#E74C3C",
    "Commodity": "#F39C12",
    "Yield": "#16A085",
    "Credit": "#8E44AD",
    "Crypto": "#2C3E50",
}


def annotate_axes(ax, title: str, unit: str, df_index, source_note: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel(unit)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    n = len(df_index)
    start = df_index.min().strftime("%Y-%m-%d")
    end = df_index.max().strftime("%Y-%m-%d")
    ax.text(
        0.01, 0.98,
        f"{start} → {end}  |  {n:,} rows  |  {source_note}",
        transform=ax.transAxes,
        fontsize=8, color="#555555",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="#cccccc"),
    )


def plot_ohlcv_panel(df: pd.DataFrame, coin: str, source_note: str) -> Path:
    """Combined OHLC + Volume panel for a coin."""
    fig, (ax_p, ax_v) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    color = CATEGORY_COLOR["Crypto"]
    ax_p.plot(df.index, df["Close"], color=color, linewidth=0.9, label="Close")
    ax_p.fill_between(df.index, df["Low"], df["High"], color=color, alpha=0.15, label="Daily High–Low range")
    ax_p.set_yscale("log")
    annotate_axes(ax_p, f"{coin} Price (OHLC, log scale)", "USD (log)", df.index, source_note)
    ax_p.legend(loc="upper left", fontsize=9)

    ax_v.bar(df.index, df["Volume"], color=color, alpha=0.5, width=1.0)
    ax_v.set_ylabel("Volume")
    ax_v.set_xlabel("Date")
    ax_v.grid(True, alpha=0.3)
    ax_v.set_title(f"{coin} Volume", fontsize=11)

    plt.tight_layout()
    path = OUT_DIR / f"{coin.lower()}_price_volume.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_single(series: pd.Series, col: str, source_note: str) -> Path:
    meta = COL_META.get(col, {})
    title = meta.get("title", col)
    unit = meta.get("unit", "")
    category = meta.get("category", "Other")
    log = meta.get("log", False)
    color = CATEGORY_COLOR.get(category, "#34495E")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(series.index, series.values, color=color, linewidth=0.85)
    if log:
        ax.set_yscale("log")
    annotate_axes(ax, title, unit, series.index, source_note)
    plt.tight_layout()
    path = OUT_DIR / f"{col.lower()}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    eth = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "eth_aligned.csv",
                      index_col=0, parse_dates=True)

    btc_note = "btc_aligned (forward-filled, +1d macro lag)"
    eth_note = "eth_aligned (forward-filled, +1d macro lag)"

    written: list[str] = []

    print("[1] Crypto OHLC+Volume panels")
    written.append(plot_ohlcv_panel(btc, "BTC", btc_note).name)
    written.append(plot_ohlcv_panel(eth, "ETH", eth_note).name)

    print("[2] BTC macro columns (one PNG per indicator, full BTC date range)")
    macro_cols = [
        "SP500", "VIX", "DXY",
        "Gold", "Silver", "Oil_WTI",
        "US10Y", "US5Y", "US3M", "US30Y", "US2Y",
        "HY_Bond", "IG_Bond", "Treasury20Y", "TIPS",
    ]
    for col in macro_cols:
        if col not in btc.columns:
            print(f"  skip {col} (not in btc_aligned)")
            continue
        s = btc[col].dropna()
        if len(s) == 0:
            print(f"  skip {col} (empty)")
            continue
        written.append(plot_single(s, col, btc_note).name)

    print(f"\nWrote {len(written)} PNG files to {OUT_DIR}")
    for name in written:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
