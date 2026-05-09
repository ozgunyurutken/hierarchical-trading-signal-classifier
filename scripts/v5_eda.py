"""
V5 Faz 1.5 — Exploratory Data Analysis (Decision Gate 1).

Reads btc_aligned_v5.csv + eth_aligned_v5.csv, produces summary plots
for user review before proceeding to feature engineering (Faz 2).

Outputs:
  reports/v5_eda_overview.png            — BTC + ETH price + train/val/test split
  reports/v5_eda_macro_panels.png        — 9 macro time series
  reports/v5_eda_distributions.png       — return distribution histograms
  reports/v5_eda_correlation.png         — feature correlation matrix
  data/processed/v5_eda_summary.csv      — summary stats table
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.utils.config import cfg

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False})

config = cfg()
TRAIN_FRAC = config["training"]["train_size"]
VAL_FRAC = config["training"]["validation_size"]
TEST_FRAC = config["training"]["test_size"]


def load_aligned():
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned_v5.csv",
                      index_col=0, parse_dates=True)
    eth = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "eth_aligned_v5.csv",
                      index_col=0, parse_dates=True)
    return btc, eth


def get_split_dates(df):
    n = len(df)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return df.index[n_train - 1], df.index[n_train + n_val - 1]


def plot_overview(btc, eth, out: Path):
    """Top: BTC log price + train/val/test shading. Bottom: ETH log price."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    train_end, val_end = get_split_dates(btc)
    test_start = btc.index[btc.index > val_end][0]

    for ax, df, label, color in [(axes[0], btc, "BTC", "#f7931a"),
                                  (axes[1], eth, "ETH", "#627eea")]:
        ax.semilogy(df.index, df["Close"], color=color, lw=1.2, label=f"{label} Close")
        # split shading
        ax.axvspan(df.index[0], train_end, color="#cfe5ff", alpha=0.4, label="Train (70%)")
        ax.axvspan(train_end, val_end, color="#ffe5b4", alpha=0.4, label="Val (15%)")
        ax.axvspan(val_end, df.index[-1], color="#ffb3b3", alpha=0.4, label="Test (15%)")
        ax.set_ylabel(f"{label} (USD, log)")
        # train/val/test cutoff lines
        for d, lbl in [(train_end, "train end"), (val_end, "val end")]:
            ax.axvline(d, color="black", ls=":", lw=0.7, alpha=0.7)
        ax.legend(loc="upper left", fontsize=8, ncol=4)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Tarih")

    n_btc = len(btc); n_eth = len(eth)
    n_btc_tr = int(n_btc * TRAIN_FRAC); n_btc_v = int(n_btc * VAL_FRAC); n_btc_te = n_btc - n_btc_tr - n_btc_v
    n_eth_tr = int(n_eth * TRAIN_FRAC); n_eth_v = int(n_eth * VAL_FRAC); n_eth_te = n_eth - n_eth_tr - n_eth_v
    fig.suptitle(f"V5 Aligned Dataset Overview — BTC ({n_btc}d, {btc.index.min().date()}→) + "
                 f"ETH ({n_eth}d, {eth.index.min().date()}→) → 2025-12\n"
                 f"BTC train/val/test: {n_btc_tr}/{n_btc_v}/{n_btc_te}d  ·  "
                 f"ETH train/val/test: {n_eth_tr}/{n_eth_v}/{n_eth_te}d  ·  chronological 70/15/15",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_macro(btc, out: Path):
    """10-panel macro time series (5x2) with split shading on top panel."""
    macro_cols = ["SP500", "VIX", "DXY", "Gold", "US10Y", "US2Y",
                  "FEDFUNDS", "CPIAUCSL", "UNRATE", "WM2NS"]
    train_end, val_end = get_split_dates(btc)

    fig, axes = plt.subplots(5, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    for i, col in enumerate(macro_cols):
        ax = axes[i]
        ax.plot(btc.index, btc[col], lw=0.9, color=f"C{i % 10}")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.axvline(train_end, color="black", ls=":", lw=0.6, alpha=0.6)
        ax.axvline(val_end, color="black", ls=":", lw=0.6, alpha=0.6)

    for ax in axes[-2:]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)

    fig.suptitle("V5 Macro Features Time Series — 10 raw inputs (incl. M2 / WM2NS)\n"
                 "(dotted lines: train/val/test split boundaries)",
                 fontsize=11.5, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_distributions(btc, eth, out: Path):
    """Daily log returns + distribution per regime period."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    for col_idx, (df, label, color) in enumerate([(btc, "BTC", "#f7931a"),
                                                   (eth, "ETH", "#627eea")]):
        ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        # Daily returns histogram
        ax = axes[0, col_idx]
        ax.hist(ret, bins=80, color=color, alpha=0.7, edgecolor="black", lw=0.3)
        ax.axvline(ret.mean(), color="red", lw=1, label=f"mean={ret.mean()*100:.2f}%")
        ax.axvline(0, color="black", lw=0.5, alpha=0.5)
        ax.set_title(f"{label} Daily Log Returns Distribution", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Log return"); ax.legend(fontsize=8)
        # Annualised vol per year
        ax2 = axes[1, col_idx]
        rolling_vol = ret.rolling(20).std() * np.sqrt(252) * 100
        ax2.plot(rolling_vol.index, rolling_vol, color=color, lw=0.8)
        ax2.axhline(rolling_vol.mean(), color="red", lw=0.8, ls="--", alpha=0.7,
                    label=f"mean={rolling_vol.mean():.1f}%")
        ax2.set_title(f"{label} 20-day Realized Volatility (annualized %)", fontsize=10.5, fontweight="bold")
        ax2.set_ylabel("Annualized vol (%)")
        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax2.legend(fontsize=8)

    fig.suptitle("V5 Return + Volatility Profile  (BTC vs ETH)",
                 fontsize=11.5, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_macro_pretrain(out: Path, crypto_start: str = "2014-09-17"):
    """25-yıllık (2000-2025) macro pre-train tarihçesi — Stage 2 K-Means fit window.
    Crypto era highlighted; major regime events annotated."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    risk = pd.read_csv(raw_dir / "v5_macro_risk.csv", index_col=0, parse_dates=True)
    com = pd.read_csv(raw_dir / "v5_macro_commodities.csv", index_col=0, parse_dates=True)
    yld = pd.read_csv(raw_dir / "v5_macro_yields.csv", index_col=0, parse_dates=True)
    fred = pd.read_csv(raw_dir / "v5_macro_fred_monthly.csv", index_col=0, parse_dates=True)

    crypto_dt = pd.Timestamp(crypto_start)

    fig, axes = plt.subplots(5, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    panels = [
        (axes[0], risk["SP500"], "S&P 500", "C0"),
        (axes[1], risk["VIX"], "VIX (log)", "C1"),
        (axes[2], risk["DXY"], "DXY", "C2"),
        (axes[3], com["Gold"], "Gold", "C3"),
        (axes[4], yld["US10Y"], "US 10Y Yield", "C4"),
        (axes[5], yld["US2Y"], "US 2Y Yield (FRED DGS2)", "C5"),
        (axes[6], fred["FEDFUNDS"], "FEDFUNDS", "C6"),
        (axes[7], fred["CPIAUCSL"], "CPI Index", "C7"),
        (axes[8], fred["UNRATE"], "UNRATE", "C8"),
        (axes[9], fred["WM2NS"], "M2 Money Supply (WM2NS, log)", "C9"),
    ]

    for ax, series, title, color in panels:
        s = series.dropna()  # FRED monthly + WM2NS weekly farklı tarihler, NaN'leri at
        if title.startswith("VIX") or title.startswith("M2"):
            ax.semilogy(s.index, s.values, lw=0.9, color=color)
        else:
            ax.plot(s.index, s.values, lw=0.9, color=color)

        # Crypto-era shading (V5+ training start)
        ax.axvspan(crypto_dt, series.index.max(), color="#ffe5b4", alpha=0.4)
        ax.axvline(crypto_dt, color="black", ls=":", lw=0.7)
        ax.set_title(title, fontsize=10, fontweight="bold")

    # Major regime annotations on first panel
    events = [
        (pd.Timestamp("2000-03-10"), "dot-com peak", -0.05),
        (pd.Timestamp("2008-09-15"), "Lehman", 0.10),
        (pd.Timestamp("2020-03-23"), "COVID", -0.05),
        (pd.Timestamp("2022-03-16"), "Fed hike\nbegins", 0.10),
    ]
    sp_ax = axes[0]
    for date, label, ypos_frac in events:
        if date < risk.index.max():
            sp_ax.axvline(date, color="red", ls="--", lw=0.8, alpha=0.7)
            ymin, ymax = sp_ax.get_ylim()
            ypos = ymin + (ymax - ymin) * (0.85 + ypos_frac)
            sp_ax.annotate(label, xy=(date, ypos), fontsize=7,
                           ha="center", color="red",
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", lw=0.5))

    # Add legend hint
    handles = [
        Patch(facecolor="#ffe5b4", alpha=0.4, label=f"Crypto era (V5+ training, post-{crypto_dt.year})"),
        plt.Line2D([], [], color="red", ls="--", lw=0.8, label="Major regime events"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.005))

    for ax in axes[-2:]:
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)

    n_pretrain = (risk.index.max() - risk.index.min()).days
    fig.suptitle(f"V5+ Macro Pre-Training Window — {risk.index.min().date()} → {risk.index.max().date()}  "
                 f"(~{n_pretrain // 365} yıl, {len(risk)} gün) — 10 features (incl. M2)\n"
                 f"Stage 2 K-Means cluster fit covers 4 regime cycle: dot-com + GFC 2008 + COVID 2020 + Fed hike 2022\n"
                 f"Inference yapılırken sadece crypto era (highlighted) kullanılır",
                 fontsize=11.5, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_correlation(btc, out: Path):
    """Correlation heatmap of FEATURE columns on BTC.
    Open/High/Low excluded (~1.0 with Close, redundant)."""
    feature_cols = [c for c in btc.columns if c not in ("Open", "High", "Low")]
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = btc[feature_cols].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5, color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"BTC V5 Feature Correlation Matrix (Pearson)\n"
                 f"{len(corr)} feature columns: Close + Volume + {len(corr) - 2} macro "
                 f"(Open/High/Low excluded — ~1.0 with Close, redundant)",
                 fontsize=11, fontweight="bold")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def write_summary(btc, eth, out: Path):
    """Per-coin descriptive stats."""
    rows = []
    for label, df in [("BTC", btc), ("ETH", eth)]:
        ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
        train_end, val_end = get_split_dates(df)
        rows.append({
            "coin": label,
            "n_days": len(df),
            "first_date": df.index[0].date(),
            "last_date": df.index[-1].date(),
            "n_columns": len(df.columns),
            "train_end": train_end.date(),
            "val_end": val_end.date(),
            "test_start": df.index[df.index > val_end][0].date(),
            "close_min": round(float(df["Close"].min()), 2),
            "close_max": round(float(df["Close"].max()), 2),
            "close_mean": round(float(df["Close"].mean()), 2),
            "ret_mean_daily_pct": round(float(ret.mean() * 100), 4),
            "ret_std_daily_pct": round(float(ret.std() * 100), 4),
            "ret_annual_vol_pct": round(float(ret.std() * np.sqrt(252) * 100), 2),
            "ret_min_daily_pct": round(float(ret.min() * 100), 2),
            "ret_max_daily_pct": round(float(ret.max() * 100), 2),
            "vix_mean": round(float(df["VIX"].mean()), 2),
            "vix_max": round(float(df["VIX"].max()), 2),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out, index=False)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")
    return summary


def main():
    print("V5 Faz 1.5 — Exploratory Data Analysis")
    print("=" * 60)

    btc, eth = load_aligned()

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    proc = PROJECT_ROOT / "data" / "processed"

    print("\n[1] Plot overview (price + train/val/test split)")
    plot_overview(btc, eth, reports / "v5_eda_overview.png")

    print("\n[2a] Plot macro panels (9 inputs, crypto-aligned 2014-2025)")
    plot_macro(btc, reports / "v5_eda_macro_panels.png")

    print("\n[2b] Plot macro pre-train window (2000-2025, Stage 2 K-Means fit data)")
    plot_macro_pretrain(reports / "v5_eda_macro_pretrain.png")

    print("\n[3] Plot return + volatility distributions")
    plot_distributions(btc, eth, reports / "v5_eda_distributions.png")

    print("\n[4] Correlation matrix")
    plot_correlation(btc, reports / "v5_eda_correlation.png")

    print("\n[5] Summary statistics CSV")
    summary = write_summary(btc, eth, proc / "v5_eda_summary.csv")
    print("\n=== Summary ===")
    print(summary.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
