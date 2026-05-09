"""
V5 Faz 2.2 detail — Regime overlay on BTC + ETH + S&P + VIX time series.

User request: 'Stage 2 regime sonuçlarını BTC, S&P, VIX gibi grafiklerin üzerinde
görmek isterim'.

Output: reports/Phase2/v5_p2_regime_overlay_assets.png
4-panel: BTC log + ETH log + S&P 500 + VIX (log) — all with regime shading
across full pre-train + crypto era (2000-2025).
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.patches import Patch

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})

REGIME_COLORS = {"Risk-On": "#7ec27e", "Risk-Off": "#e07e7e", "Neutral": "#f0c870"}


def shade_regime(ax, price_series, regime_series, log=False, line_color="black", lw=1.0):
    """Draw price + paint background by regime runs."""
    if log:
        ax.semilogy(price_series.index, price_series.values, color=line_color, lw=lw)
    else:
        ax.plot(price_series.index, price_series.values, color=line_color, lw=lw)
    if regime_series.empty:
        return
    cur, start = regime_series.iloc[0], regime_series.index[0]
    for i in range(1, len(regime_series)):
        val = regime_series.iloc[i]
        if val != cur:
            if pd.notna(cur):
                ax.axvspan(start, regime_series.index[i],
                           color=REGIME_COLORS.get(cur, "white"), alpha=0.30, lw=0)
            cur, start = val, regime_series.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime_series.index[-1],
                   color=REGIME_COLORS.get(cur, "white"), alpha=0.30, lw=0)


def main():
    proc = PROJECT_ROOT / "data" / "processed"
    raw = PROJECT_ROOT / "data" / "raw"

    # Load regime labels (pretrain has full 2000-2025 coverage)
    pretrain_regime = pd.read_csv(proc / "macro_pretrain_regime_labels_v5.csv",
                                  index_col=0, parse_dates=True)

    # Macro raw assets (2000-2025)
    risk = pd.read_csv(raw / "v5_macro_risk.csv", index_col=0, parse_dates=True)
    sp500 = risk["SP500"].dropna()
    vix = risk["VIX"].dropna()

    # BTC + ETH (different start dates)
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)
    btc_close = btc["Close"]
    eth_close = eth["Close"]

    # Get regime labels aligned to each asset's index
    def regime_aligned_to(asset_idx):
        return pretrain_regime["regime_label"].reindex(asset_idx, method="ffill")

    btc_regime = regime_aligned_to(btc_close.index)
    eth_regime = regime_aligned_to(eth_close.index)
    sp500_regime = regime_aligned_to(sp500.index)
    vix_regime = regime_aligned_to(vix.index)

    # 4-panel plot
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    # Panel 1: S&P 500 log price (full pretrain coverage)
    ax = axes[0]
    shade_regime(ax, sp500, sp500_regime, log=True, line_color="#1f77b4", lw=1.0)
    ax.set_ylabel("S&P 500 (log)")
    ax.set_title("S&P 500 (full 2000-2025) + Stage 2 regime shading", fontsize=10.5, fontweight="bold")

    # Panel 2: VIX log
    ax = axes[1]
    shade_regime(ax, vix, vix_regime, log=True, line_color="#ff7f0e", lw=1.0)
    ax.set_ylabel("VIX (log)")
    ax.axhline(20, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.axhline(30, color="red", lw=0.5, ls="--", alpha=0.5)
    ax.set_title("VIX (log scale; ref 20 = elevated, 30 = stress) + regime shading",
                 fontsize=10.5, fontweight="bold")

    # Panel 3: BTC log
    ax = axes[2]
    shade_regime(ax, btc_close, btc_regime, log=True, line_color="#f7931a", lw=1.0)
    ax.set_ylabel("BTC (USD, log)")
    ax.set_title("BTC (2014-09 → 2025-12) + regime shading", fontsize=10.5, fontweight="bold")

    # Panel 4: ETH log
    ax = axes[3]
    shade_regime(ax, eth_close, eth_regime, log=True, line_color="#627eea", lw=1.0)
    ax.set_ylabel("ETH (USD, log)")
    ax.set_title("ETH (2017-11 → 2025-12) + regime shading", fontsize=10.5, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Tarih")

    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("V5 Faz 2.2 — Stage 2 Regime Overlay on Multiple Assets\n"
                 "K-Means fit on pre-train (2000-2025), inference applied to BTC + ETH + S&P + VIX\n"
                 "Risk-Off bands: 2002 dot-com bust, 2008-09 GFC, 2020 COVID, 2022 hike (varies)",
                 fontsize=11.5, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    out = PROJECT_ROOT / "reports" / "Phase2" / "v5_p2_regime_overlay_assets.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
