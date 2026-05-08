"""
Visualize 5 monthly FRED series + Fisher real interest rate on the
BTC daily timeline, with Stage 2 v4 GMM cluster shading for context.

Output: reports/monthly_fred_overview.png  (single 7-panel figure)
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

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})

REGIME_COLORS = {0: "#bde0c2", 1: "#f8d6a4", 2: "#f7b3b3"}  # green/orange/red
REGIME_LABEL = {0: "Calm", 1: "Transition", 2: "Stress"}


def main() -> None:
    btc = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_aligned.csv",
                      index_col=0, parse_dates=True)
    macro = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "btc_features_macro.csv",
                        index_col=0, parse_dates=True)
    s2 = pd.read_csv(PROJECT_ROOT / "data" / "labels" / "btc_oof_regime_posterior.csv",
                     index_col=0, parse_dates=True)
    common = btc.index.intersection(macro.index).intersection(s2.index)
    btc = btc.loc[common]; macro = macro.loc[common]; s2 = s2.loc[common]
    hard = s2.values.argmax(1)

    # CPI YoY (% change 252-day) — needed for plot
    cpi_yoy = btc["CPIAUCSL"].pct_change(252) * 100

    fig, axes = plt.subplots(7, 1, figsize=(13, 18), sharex=True,
                             gridspec_kw={"height_ratios": [1.4, 1, 1, 1, 1, 1, 1]})

    # ---------------- Panel 0: BTC log price + regime shading ----------------
    ax = axes[0]
    ax.semilogy(btc.index, btc["Close"], color="#222", lw=1.0, label="BTC Close (log scale)")
    # cluster shading: paint only on regime change, span [t_start, t_end]
    cur, start = hard[0], common[0]
    for i in range(1, len(common)):
        if hard[i] != cur:
            ax.axvspan(start, common[i], color=REGIME_COLORS[cur], alpha=0.35, lw=0)
            cur, start = hard[i], common[i]
    ax.axvspan(start, common[-1], color=REGIME_COLORS[cur], alpha=0.35, lw=0)
    ax.set_ylabel("BTC (USD, log)")
    ax.set_title("BTC fiyatı + Stage 2 v4 GMM rejimi (3-küme)  /  arka plan v4 (11-feature) cluster",
                 fontsize=10, fontweight="bold")
    legend_handles = [Patch(facecolor=c, alpha=0.4, label=f"Regime {k} ({REGIME_LABEL[k]})")
                      for k, c in REGIME_COLORS.items()]
    legend_handles.insert(0, plt.Line2D([], [], color="#222", lw=1, label="BTC Close"))
    ax.legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=2)

    # ---------------- Panel 1: FEDFUNDS + UNRATE (twin axis) ----------------
    ax = axes[1]
    ax.plot(btc.index, btc["FEDFUNDS"], color="#1f77b4", lw=1.4, label="FEDFUNDS (%, lag 1d)")
    ax.set_ylabel("FEDFUNDS (%)", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(btc.index, btc["UNRATE"], color="#d62728", lw=1.4, label="UNRATE (%, lag 35d)")
    ax2.set_ylabel("UNRATE (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax.set_title("Federal Funds Rate (mavi)  +  Unemployment Rate (kırmızı)", fontsize=9.5)

    # ---------------- Panel 2: CPI raw level ----------------
    ax = axes[2]
    ax.plot(btc.index, btc["CPIAUCSL"], color="#2ca02c", lw=1.4, label="CPIAUCSL (lag 45d)")
    ax.set_ylabel("CPI Index (1982-84=100)")
    ax.set_title("CPI All Urban Consumers (raw seviye, durmadan artar — level olarak GMM'e değil, türev olarak)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)

    # ---------------- Panel 3: CPI YoY % + Fisher real rate ----------------
    ax = axes[3]
    ax.plot(btc.index, cpi_yoy, color="#9467bd", lw=1.4, label="CPI YoY (%, 252-gün pct change)")
    ax.plot(btc.index, macro["macro_real_interest_rate"], color="#ff7f0e", lw=1.4,
            label="macro_real_interest_rate (Fisher: FFR − CPI YoY)")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_ylabel("%")
    ax.set_title("CPI YoY enflasyon  +  Fisher reel faiz  (Stage 2 GMM v4 girdi #10)",
                 fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)

    # ---------------- Panel 4: WM2NS (M2 Money Supply) ----------------
    ax = axes[4]
    ax.plot(btc.index, btc["WM2NS"]/1000, color="#8c564b", lw=1.4, label="WM2NS (Trilyon $, lag 14d)")
    ax.set_ylabel("M2 (Trilyon $)")
    ax.set_title("M2 Para Arzı (haftalık, ABD Fed bilanço genişlemesi)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)

    # ---------------- Panel 5: ICSA Initial Claims (log scale) ----------------
    ax = axes[5]
    ax.semilogy(btc.index, btc["ICSA"], color="#e377c2", lw=1.0, label="ICSA (log scale, lag 5d)")
    ax.set_ylabel("Claims (log)")
    ax.set_title("Initial Jobless Claims  (log skala — COVID Mart 2020'de 6.13M zirve)",
                 fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)

    # ---------------- Panel 6: All standardised together (z-score 252) ----------------
    ax = axes[6]
    def z252(s):
        return (s - s.rolling(252).mean()) / s.rolling(252).std()
    for col, color, lbl in [
        ("FEDFUNDS", "#1f77b4", "FEDFUNDS"),
        ("UNRATE", "#d62728", "UNRATE"),
        ("WM2NS", "#8c564b", "WM2NS"),
    ]:
        ax.plot(btc.index, z252(btc[col]), color=color, lw=1.0, alpha=0.85, label=f"{lbl} z₂₅₂")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_ylabel("z-score (252-gün)")
    ax.set_title("Karşılaştırılabilir z-score görünümü (252-gün rolling)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8, ncol=3)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Tarih")

    fig.suptitle(
        "Aylık FRED Makro Serileri (v4) — BTC zaman ekseninde, publication-release lag ile shifted",
        fontsize=11.5, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.99])

    out = PROJECT_ROOT / "reports" / "monthly_fred_overview.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out.relative_to(PROJECT_ROOT)}")
    print(f"  date range: {common[0].date()} → {common[-1].date()}  ({len(common)} days)")
    cnt = pd.Series(hard).value_counts().sort_index()
    print(f"  cluster freq: {{0: {cnt.get(0,0)}, 1: {cnt.get(1,0)}, 2: {cnt.get(2,0)}}}")


if __name__ == "__main__":
    main()
