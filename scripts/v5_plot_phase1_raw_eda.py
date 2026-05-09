"""V5 Phase 1 — Raw data EDA plots.

Outputs:
  reports/Phase1/v5_p1_btc_eth_overview.png       — BTC/ETH price + volume
  reports/Phase1/v5_p1_macro_raw_grid.png         — 12 raw macro features timeline
  reports/Phase1/v5_p1_macro_raw_correlation.png  — raw macro correlation matrix

Phase 1 = raw data inventory + EDA. Henüz feature engineering yok.
Phase 1.5 = derived/engineered features.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_MACRO_ORDER = [
    # Risk (3)
    ("SP500",     "v5_macro_risk",        "S&P 500 index"),
    ("VIX",       "v5_macro_risk",        "VIX (volatility)"),
    ("DXY",       "v5_macro_risk",        "DXY (dollar index)"),
    # Yields (2)
    ("US10Y",     "v5_macro_yields",      "US 10-Year Treasury"),
    ("US2Y",      "v5_macro_yields",      "US 2-Year Treasury"),
    # Commodities (3)
    ("Gold",      "v5_macro_commodities", "Gold futures"),
    ("Silver",    "v5_macro_commodities", "Silver futures"),
    ("Oil",       "v5_macro_commodities", "Oil (WTI futures)"),
    # FRED monthly (4)
    ("FEDFUNDS",  "v5_macro_fred_monthly", "Fed Funds Rate (monthly)"),
    ("CPIAUCSL",  "v5_macro_fred_monthly", "CPI (monthly)"),
    ("UNRATE",    "v5_macro_fred_monthly", "Unemployment Rate (monthly)"),
    ("WM2NS",     "v5_macro_fred_monthly", "M2 Money Supply (monthly)"),
]


def plot_btc_eth_overview(out_dir: Path):
    raw = PROJECT_ROOT / "data" / "raw"
    btc = pd.read_csv(raw / "v5_price_btc.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(raw / "v5_price_eth.csv", index_col=0, parse_dates=True)

    fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    axes[0, 0].semilogy(btc.index, btc["Close"], color="#F7931A", lw=1.0)
    axes[0, 0].set_title(f"BTC Close (log)  —  {btc.index.min().date()} → {btc.index.max().date()}, {len(btc)} days",
                         fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("USD (log scale)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(btc.index, btc["Volume"] / 1e9, color="#F7931A", lw=0.7)
    axes[0, 1].set_title("BTC Volume (billions USD)", fontsize=11, fontweight="bold")
    axes[0, 1].set_ylabel("Volume ($B)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(eth.index, eth["Close"], color="#627EEA", lw=1.0)
    axes[1, 0].set_title(f"ETH Close (log)  —  {eth.index.min().date()} → {eth.index.max().date()}, {len(eth)} days",
                         fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("USD (log scale)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(eth.index, eth["Volume"] / 1e9, color="#627EEA", lw=0.7)
    axes[1, 1].set_title("ETH Volume (billions USD)", fontsize=11, fontweight="bold")
    axes[1, 1].set_ylabel("Volume ($B)")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Phase 1 — BTC / ETH raw price overview", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = out_dir / "v5_p1_btc_eth_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def load_raw_macro() -> pd.DataFrame:
    raw = PROJECT_ROOT / "data" / "raw"
    cols = []
    for col, file_stem, _ in RAW_MACRO_ORDER:
        df = pd.read_csv(raw / f"{file_stem}.csv", index_col=0, parse_dates=True)
        cols.append(df[col].rename(col))
    return pd.concat(cols, axis=1).sort_index()


def plot_macro_raw_grid(out_dir: Path):
    macro = load_raw_macro()

    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex=True)
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})

    for ax, (col, _, desc) in zip(axes.flat, RAW_MACRO_ORDER):
        s = macro[col].dropna()
        ax.plot(s.index, s.values, color="#3a6fb0", lw=0.8)
        ax.set_title(f"{col}  —  {desc}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle("Phase 1 — 12 raw macro features (no engineering applied)\n"
                 "Risk (3) + Yields (2) + Commodities (3) + FRED monthly (4)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_dir / "v5_p1_macro_raw_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def plot_macro_raw_correlation(out_dir: Path):
    macro = load_raw_macro()
    # Daily resample: monthly FRED forward-filled; correlation on common daily index
    macro = macro.ffill().dropna()
    corr = macro.corr()
    n = len(corr.columns)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(corr.columns, fontsize=10)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.55 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Phase 1 — Raw macro feature correlation matrix\n"
                 "Pearson on forward-filled daily series (FRED monthly upsampled)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    out = out_dir / "v5_p1_macro_raw_correlation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    out_dir = PROJECT_ROOT / "reports" / "Phase1"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_btc_eth_overview(out_dir)
    plot_macro_raw_grid(out_dir)
    plot_macro_raw_correlation(out_dir)


if __name__ == "__main__":
    main()
