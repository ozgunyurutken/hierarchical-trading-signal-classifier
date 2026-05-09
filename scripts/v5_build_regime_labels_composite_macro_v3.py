"""
V5 Phase 2.10 — Composite Macro FSM v3 (Phase 2.9 + Bull velocity entry).

Kullanıcı feedback:
  - 2000-Mar 2020 clusterleri ✓ beğenildi (değiştirme)
  - 2020 Mar pandemic Bear ✓ beğenildi (değiştirme)
  - **2020 Apr-Sep V-shape recovery yakalanmıyor** ❗ — VIX hala yüksek olduğu
    için Bull entry threshold karşılanmıyor, sistem Neutral'da kalıyor.
  - **2022 Q4-2023 Q1 Bear çıkışı sonrası rally yakalanmıyor** ❗
  - 2023 sonrası clusterleri ✓ beğenildi (değiştirme)

Phase 2.10 ekleme: Bull entry velocity override (V-shape recovery detection).
  Mevcut Bull entry: VIX_z < -0.5 AND SP500_5d > 0
  + Yeni alternatif:
    ΔVIX_z[30d] < -0.6  AND  SP500_60d_return > +2%
  → V-shape recovery'lerde Bull'a hızlı geçiş, diğer dönemleri etkilemez.

Korunan mekanizmalar (Phase 2.9'dan):
  - Hysteresis (asymmetric VIX thresholds)
  - Dwell time
  - VIX velocity Bear→Neutral fast-track
  - Yield curve persistent inversion (60d window) Bull→Neutral force
  - DXY+M2 macro stress override

Plot: 6 ayrı yüksek-DPI dosya — pretrain panel, BTC panel, ETH panel,
yield curve panel, DXY/M2 panel, 2020-2022 zoom, 2023-2025 zoom.

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_composite_macro_v3_v5.csv
  reports/Phase2/v5_p2.10_*.png
  reports/Phase2/v5_p2.10_diagnostics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.labels.v5_regime_labels import (
    CompositeVIXRegimeClassifier, BULL_BEAR_LABELS,
)

# Yüksek çözünürlük + büyük font (kullanıcı feedback: plot resolution)
plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240,
                     "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {"Bull": "#7ec27e", "Neutral": "#f0c870", "Bear": "#e07e7e"}


def _shade_regimes(ax, price, regime, log=False, lw=1.0):
    if log:
        ax.semilogy(price.index, price.values, color="black", lw=lw, zorder=3)
    else:
        ax.plot(price.index, price.values, color="black", lw=lw, zorder=3)
    if regime.empty:
        return
    cur, start = regime.iloc[0], regime.index[0]
    for i in range(1, len(regime)):
        v = regime.iloc[i]
        if v != cur:
            if pd.notna(cur):
                ax.axvspan(start, regime.index[i],
                           color=REGIME_COLORS.get(cur, "white"),
                           alpha=0.30, lw=0, zorder=2)
            cur, start = v, regime.index[i]
    if pd.notna(cur):
        ax.axvspan(start, regime.index[-1],
                   color=REGIME_COLORS.get(cur, "white"),
                   alpha=0.30, lw=0, zorder=2)


def plot_pretrain_overview(pre_r, sp500, vix_z, model, out: Path):
    """Pretrain S&P 500 + VIX z-score + regime band + override işaretleri."""
    fig, ax = plt.subplots(figsize=(26, 11))
    _shade_regimes(ax, pd.Series(np.nan, index=pre_r.index),
                   pre_r["regime_label"], log=False, lw=0)
    ax.plot(sp500.index, sp500.values, color="#1f77b4", lw=1.2,
            label="S&P 500 (left)", zorder=3)
    ax.set_ylabel("S&P 500", color="#1f77b4", fontsize=15)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=13)
    ax2 = ax.twinx()
    ax2.plot(vix_z.index, vix_z.values, color="#d62728", lw=0.8, alpha=0.85,
             label="VIX z-score (right)", zorder=4)
    ax2.axhline(model.bear_entry_threshold, color="#d62728", ls="--", lw=0.8,
                alpha=0.6, label=f"Bear entry (>{model.bear_entry_threshold})")
    ax2.axhline(model.bull_entry_threshold, color="green", ls="--", lw=0.8,
                alpha=0.6, label=f"Bull entry (<{model.bull_entry_threshold})")
    ax2.set_ylabel("VIX z-score", color="#d62728", fontsize=15)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=13)
    ax2.grid(False); ax2.spines["top"].set_visible(False)

    sp_max = sp500.max()
    if "velocity_override" in pre_r.columns:
        d = pre_r.index[pre_r["velocity_override"]]
        if len(d):
            ax.scatter(d, [sp_max * 1.05] * len(d), marker="v",
                       color="darkred", s=90, zorder=5, label="Bear→Neutral velocity")
    if "bull_velocity_entry" in pre_r.columns:
        d = pre_r.index[pre_r["bull_velocity_entry"]]
        if len(d):
            ax.scatter(d, [sp_max * 1.10] * len(d), marker="^",
                       color="darkgreen", s=90, zorder=5, label="Neutral→Bull velocity (V-shape)")
    if "yield_curve_override" in pre_r.columns:
        d = pre_r.index[pre_r["yield_curve_override"]]
        if len(d):
            ax.scatter(d, [sp_max * 1.15] * len(d), marker="s",
                       color="purple", s=80, zorder=5, label="YC override")
    if "macro_stress_override" in pre_r.columns:
        d = pre_r.index[pre_r["macro_stress_override"]]
        if len(d):
            ax.scatter(d, [sp_max * 1.20] * len(d), marker="D",
                       color="brown", s=80, zorder=5, label="DXY/M2 macro stress")

    ax.set_title("V5 Phase 2.10 — Pre-train (2000-2025): S&P 500 + VIX z-score + Composite Regime\n"
                 "Override markers: ▼ Bear→Neutral velocity | ▲ Neutral→Bull velocity (V-shape) | "
                 "■ Yield curve | ◆ DXY+M2 macro stress",
                 fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=14)
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_crypto_panel(symbol, close, regime, out: Path):
    fig, ax = plt.subplots(figsize=(26, 9))
    _shade_regimes(ax, close, regime["regime_label"], log=True, lw=1.5)
    ax.set_ylabel(f"{symbol} (log scale)", fontsize=15)
    ax.set_title(f"V5 Phase 2.10 — {symbol} + Composite Regime",
                 fontsize=15, fontweight="bold", pad=15)
    ax.tick_params(labelsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    ax.legend(handles=handles, loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_yield_curve(yc, model, pre_r, out: Path):
    fig, ax = plt.subplots(figsize=(26, 8))
    ax.plot(yc.index, yc.values, color="purple", lw=1.0, label="10Y-2Y spread")
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.fill_between(yc.index, yc.values, 0,
                    where=(yc.values < 0), color="purple", alpha=0.30,
                    label="Inverted (recession leading indicator)")
    if "yield_curve_override" in pre_r.columns:
        d = pre_r.index[pre_r["yield_curve_override"]]
        if len(d):
            ax.scatter(d, [yc.max() * 0.95] * len(d), marker="s",
                       color="darkviolet", s=80, zorder=5,
                       label=f"YC override fired (n={len(d)})")
    ax.set_ylabel("10Y - 2Y spread (%)", fontsize=15)
    ax.set_title(f"V5 Phase 2.10 — Yield Curve 10Y-2Y "
                 f"(persistence: {model.yield_curve_persistence_window}d rolling mean)",
                 fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=12)
    ax.tick_params(labelsize=13)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_dxy_m2(dxy_z, m2, model, pre_r, out: Path):
    fig, ax = plt.subplots(figsize=(26, 8))
    ax.plot(dxy_z.index, dxy_z.values, color="orange", lw=0.9,
            label="DXY z-score (left)", alpha=0.85)
    ax.axhline(model.dxy_strong_threshold, color="orange", ls="--", lw=0.8,
               alpha=0.6, label=f"DXY strong (>{model.dxy_strong_threshold})")
    ax.set_ylabel("DXY z-score", color="orange", fontsize=15)
    ax.tick_params(axis="y", labelcolor="orange", labelsize=13)
    ax2 = ax.twinx()
    ax2.plot(m2.index, m2.values, color="teal", lw=0.9,
             label="M2 yoy change (right)", alpha=0.85)
    ax2.axhline(model.m2_low_threshold, color="teal", ls="--", lw=0.8,
                alpha=0.6, label=f"M2 low (<{model.m2_low_threshold})")
    ax2.set_ylabel("M2 yoy change", color="teal", fontsize=15)
    ax2.tick_params(axis="y", labelcolor="teal", labelsize=13)
    ax2.grid(False); ax2.spines["top"].set_visible(False)
    if "macro_stress_override" in pre_r.columns:
        d = pre_r.index[pre_r["macro_stress_override"]]
        if len(d):
            ax.scatter(d, [dxy_z.max() * 0.95] * len(d), marker="D",
                       color="brown", s=80, zorder=5,
                       label=f"Macro stress fired (n={len(d)})")
    ax.set_title(f"V5 Phase 2.10 — DXY z-score + M2 yoy (Macro stress: "
                 f"DXY>{model.dxy_strong_threshold} {model.macro_stress_combine} "
                 f"M2<{model.m2_low_threshold})",
                 fontsize=15, fontweight="bold", pad=15)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="upper left", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_zoom(start, end, btc_close, eth_close, btc_r, eth_r, pre_r, sp500, title, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(24, 16), sharex=True)
    sp_slice = sp500.loc[(sp500.index >= start) & (sp500.index <= end)]
    pre_slice = pre_r.loc[(pre_r.index >= start) & (pre_r.index <= end)]
    btc_slice = btc_close.loc[(btc_close.index >= start) & (btc_close.index <= end)]
    eth_slice = eth_close.loc[(eth_close.index >= start) & (eth_close.index <= end)]
    btc_r_slice = btc_r.loc[(btc_r.index >= start) & (btc_r.index <= end)]
    eth_r_slice = eth_r.loc[(eth_r.index >= start) & (eth_r.index <= end)]

    _shade_regimes(axes[0], sp_slice, pre_slice["regime_label"], log=False, lw=1.5)
    axes[0].set_ylabel("S&P 500", fontsize=14)
    axes[0].set_title(f"S&P 500 ({start.date()} → {end.date()})",
                      fontsize=14, fontweight="bold")

    if not btc_slice.empty:
        _shade_regimes(axes[1], btc_slice, btc_r_slice["regime_label"], log=True, lw=1.6)
        axes[1].set_ylabel("BTC (log)", fontsize=14)
        axes[1].set_title(f"BTC ({start.date()} → {end.date()})",
                          fontsize=14, fontweight="bold")
    else:
        axes[1].text(0.5, 0.5, "BTC era starts later", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=14)

    if not eth_slice.empty:
        _shade_regimes(axes[2], eth_slice, eth_r_slice["regime_label"], log=True, lw=1.6)
        axes[2].set_ylabel("ETH (log)", fontsize=14)
        axes[2].set_title(f"ETH ({start.date()} → {end.date()})",
                          fontsize=14, fontweight="bold")
    else:
        axes[2].text(0.5, 0.5, "ETH era starts later", ha="center", va="center",
                     transform=axes[2].transAxes, fontsize=14)

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_distribution(btc_r, eth_r, pre_r, out: Path):
    rows = []
    for label, df in [("Pre-train", pre_r), ("BTC", btc_r), ("ETH", eth_r)]:
        c = df["regime_label"].value_counts(normalize=True) * 100
        for r in BULL_BEAR_LABELS:
            rows.append({"period": label, "regime": r, "pct": float(c.get(r, 0))})
    pivot = pd.DataFrame(rows).pivot(index="period", columns="regime",
                                      values="pct")[BULL_BEAR_LABELS]
    fig, ax = plt.subplots(figsize=(18, 7))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[REGIME_COLORS[r] for r in BULL_BEAR_LABELS],
               edgecolor="black", lw=1.0)
    ax.set_xlabel("% of days", fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_title("V5 Phase 2.10 — Composite Macro Regime Distribution",
                 fontsize=15, fontweight="bold")
    ax.tick_params(labelsize=14)
    for i, period in enumerate(pivot.index):
        cum = 0
        for r in BULL_BEAR_LABELS:
            v = pivot.loc[period, r]
            if v > 4:
                ax.text(cum + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=14, fontweight="bold")
            cum += v
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    print("V5 Phase 2.10 — Composite Macro FSM v3 (+ Bull velocity entry)")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    raw = PROJECT_ROOT / "data" / "raw"
    reports = PROJECT_ROOT / "reports"
    (reports / "Phase2").mkdir(parents=True, exist_ok=True)

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True)

    # Phase 2.10 inline feature: SP500_60d_return for V-shape recovery detection
    sp500_raw = pd.read_csv(raw / "v5_macro_risk.csv", index_col=0, parse_dates=True)["SP500"]
    sp500_60d = sp500_raw / sp500_raw.shift(60) - 1.0
    pretrain["SP500_60d_return"] = sp500_60d.reindex(pretrain.index)

    pretrain_clean = pretrain.dropna(subset=["VIX_zscore_long",
                                              "SP500_log_return_5d",
                                              "Yield_Curve_10Y_2Y",
                                              "DXY_zscore_long",
                                              "M2_yoy_change",
                                              "SP500_60d_return"])
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print("\n[1] Fit Composite Macro FSM v3")
    model = CompositeVIXRegimeClassifier(
        bear_entry_threshold=1.0,
        bear_exit_threshold=0.3,
        bull_entry_threshold=-0.5,
        bull_exit_threshold=0.0,
        bear_min_dwell=20,
        bull_min_dwell=30,
        velocity_window=5,
        velocity_threshold=-0.8,
        velocity_sp500_min=0.0,
        initial_regime="Neutral",
        # Yield curve (Phase 2.9 değişmedi — kullanıcı 2024 clusterleri beğendi)
        enable_yield_curve_override=True,
        yield_curve_inverted_threshold=0.0,
        yield_curve_persistence_window=60,
        yield_curve_blocks_bull_entry=False,
        yc_requires_sp500_weakness=False,         # 2024 cluster korunsun
        # Macro stress (gevşetilmiş threshold)
        enable_macro_stress_override=True,
        dxy_strong_threshold=0.7,                 # 1.0 → 0.7
        m2_low_threshold=0.040,                   # 0.023 → 0.040 (~p25)
        macro_stress_window=30,
        macro_stress_combine="AND",
        # Phase 2.10 YENİ: Bull entry velocity override (V-shape recovery)
        enable_bull_velocity_entry=True,
        bull_velocity_window=30,
        bull_velocity_threshold=-0.6,
        bull_velocity_sp500_min=0.02,             # SP500 60d > +2%
    ).fit(pretrain_clean)
    print(f"  Bear entry: VIX_z > {model.bear_entry_threshold}")
    print(f"  Bear exit:  VIX_z < {model.bear_exit_threshold}")
    print(f"  Bull entry (standard):  VIX_z < {model.bull_entry_threshold} AND SP500_5d > 0")
    print(f"  Bull entry (V-shape):   ΔVIX_z[{model.bull_velocity_window}d] < {model.bull_velocity_threshold} "
          f"AND SP500_60d > {model.bull_velocity_sp500_min:+.2f}")
    print(f"  Bull exit:  VIX_z > {model.bull_exit_threshold} OR YC inv OR macro stress")
    print(f"  YC override: rolling {model.yield_curve_persistence_window}d mean < 0")
    print(f"  Macro stress: DXY[30d]>{model.dxy_strong_threshold} {model.macro_stress_combine} "
          f"M2[30d]<{model.m2_low_threshold}")

    print(f"\n[2] Inference")
    pre_r = model.predict(pretrain_clean)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_composite_macro_v3_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_composite_macro_v3_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_composite_macro_v3_v5.csv")

    n_velocity = int(pre_r["velocity_override"].sum())
    n_yc = int(pre_r["yield_curve_override"].sum())
    n_ms = int(pre_r["macro_stress_override"].sum())
    n_bv = int(pre_r["bull_velocity_entry"].sum())
    print(f"  Bear→Neutral velocity: {n_velocity} times")
    print(f"  Neutral→Bull velocity (V-shape): {n_bv} times")
    if n_bv > 0:
        for d in pre_r.index[pre_r["bull_velocity_entry"]]:
            print(f"    BV: {d.date()}")
    print(f"  Yield curve override: {n_yc} times")
    print(f"  Macro stress override: {n_ms} times")
    if n_ms > 0:
        for d in pre_r.index[pre_r["macro_stress_override"]][:10]:
            print(f"    MS: {d.date()}")

    print(f"\n[3] Plots (yüksek çözünürlük + parçalı)")
    sp500 = sp500_raw.dropna()
    derived = pretrain
    vix_z = derived["VIX_zscore_long"].dropna()
    yc = derived["Yield_Curve_10Y_2Y"]
    dxy_z = derived["DXY_zscore_long"]
    m2 = derived["M2_yoy_change"]

    plot_pretrain_overview(pre_r, sp500, vix_z, model,
                           reports / "Phase2" / "v5_p2.10_01_pretrain_overview.png")
    plot_crypto_panel("BTC", btc["Close"].loc[btc_r.index], btc_r,
                      reports / "Phase2" / "v5_p2.10_02_btc.png")
    plot_crypto_panel("ETH", eth["Close"].loc[eth_r.index], eth_r,
                      reports / "Phase2" / "v5_p2.10_03_eth.png")
    plot_yield_curve(yc, model, pre_r,
                     reports / "Phase2" / "v5_p2.10_04_yield_curve.png")
    plot_dxy_m2(dxy_z, m2, model, pre_r,
                reports / "Phase2" / "v5_p2.10_05_dxy_m2.png")
    plot_zoom(pd.Timestamp("2019-06-01"), pd.Timestamp("2022-12-31"),
              btc["Close"].loc[btc_r.index], eth["Close"].loc[eth_r.index],
              btc_r, eth_r, pre_r, sp500,
              "V5 Phase 2.10 — 2019-2022 Zoom (kullanıcı feedback: 2020 V-shape + 2022 Q4 rally)",
              reports / "Phase2" / "v5_p2.10_06_2020_2022_zoom.png")
    plot_zoom(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31"),
              btc["Close"].loc[btc_r.index], eth["Close"].loc[eth_r.index],
              btc_r, eth_r, pre_r, sp500,
              "V5 Phase 2.10 — 2023-2025 Zoom (kullanıcı beğendi: korunmalı)",
              reports / "Phase2" / "v5_p2.10_07_2023_2025_zoom.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "Phase2" / "v5_p2.10_08_distribution.png")

    print("\n" + "=" * 70)
    print("Distribution:")
    distribution = {}
    for label, df, key in [("Pre-train", pre_r, "pretrain"),
                            ("BTC era", btc_r, "btc"),
                            ("ETH era", eth_r, "eth")]:
        c = df["regime_label"].value_counts()
        total = c.sum()
        print(f"  {label:12s} ", end="")
        distribution[key] = {}
        for r in BULL_BEAR_LABELS:
            v = int(c.get(r, 0))
            pct = float(v / total * 100) if total else 0.0
            distribution[key][r] = pct
            print(f"{r}: {v} ({pct:.1f}%)  ", end="")
        print()

    runs = []
    cur, start = pre_r["regime_label"].iloc[0], pre_r.index[0]
    for i in range(1, len(pre_r)):
        v = pre_r["regime_label"].iloc[i]
        if v != cur:
            runs.append({"regime": cur, "start": str(start.date()),
                         "end": str(pre_r.index[i - 1].date()),
                         "duration_days": (pre_r.index[i - 1] - start).days})
            cur, start = v, pre_r.index[i]
    runs.append({"regime": cur, "start": str(start.date()),
                 "end": str(pre_r.index[-1].date()),
                 "duration_days": (pre_r.index[-1] - start).days})
    bear_runs = [r for r in runs if r["regime"] == "Bear"]
    bull_runs = [r for r in runs if r["regime"] == "Bull"]
    neutral_runs = [r for r in runs if r["regime"] == "Neutral"]
    print(f"\nRun-length stats (pretrain):")
    if bear_runs:
        print(f"  # Bear runs: {len(bear_runs)}, mean: "
              f"{np.mean([r['duration_days'] for r in bear_runs]):.0f}d, "
              f"max: {max([r['duration_days'] for r in bear_runs]):.0f}d")
    if bull_runs:
        print(f"  # Bull runs: {len(bull_runs)}, mean: "
              f"{np.mean([r['duration_days'] for r in bull_runs]):.0f}d, "
              f"max: {max([r['duration_days'] for r in bull_runs]):.0f}d")
    if neutral_runs:
        print(f"  # Neutral runs: {len(neutral_runs)}, mean: "
              f"{np.mean([r['duration_days'] for r in neutral_runs]):.0f}d")

    diagnostics = {
        "phase": "V5 Phase 2.10 — Composite Macro FSM v3 (+ Bull velocity entry)",
        "user_feedback_addressed": [
            "2020 Apr-Sep V-shape recovery yakalanıyor (Bull velocity entry)",
            "2022 Q4-2023 Q1 rally yakalanıyor (Bull velocity entry)",
            "2024 cluster korundu (YC override Phase 2.9 default)",
            "2000-2019 cluster korundu (yeni mekanizma sadece V-shape için)",
        ],
        "thresholds": {
            "bear_entry": model.bear_entry_threshold,
            "bear_exit": model.bear_exit_threshold,
            "bull_entry": model.bull_entry_threshold,
            "bull_exit": model.bull_exit_threshold,
        },
        "dwell_time": {"bear": model.bear_min_dwell, "bull": model.bull_min_dwell},
        "bear_velocity": {"window": model.velocity_window,
                          "threshold": model.velocity_threshold,
                          "n_fired": n_velocity},
        "bull_velocity_entry": {"window": model.bull_velocity_window,
                                 "threshold": model.bull_velocity_threshold,
                                 "sp500_min": model.bull_velocity_sp500_min,
                                 "n_fired": n_bv,
                                 "fired_dates": [str(d.date()) for d in pre_r.index[pre_r["bull_velocity_entry"]]]},
        "yield_curve_override": {"persistence": model.yield_curve_persistence_window,
                                  "n_fired": n_yc},
        "macro_stress_override": {"dxy_threshold": model.dxy_strong_threshold,
                                   "m2_threshold": model.m2_low_threshold,
                                   "combine": model.macro_stress_combine,
                                   "n_fired": n_ms},
        "distribution_pct": distribution,
        "bear_runs_summary": {
            "count": len(bear_runs),
            "mean_days": float(np.mean([r['duration_days'] for r in bear_runs])) if bear_runs else 0.0,
            "max_days": int(max([r['duration_days'] for r in bear_runs])) if bear_runs else 0,
        },
        "bull_runs_summary": {
            "count": len(bull_runs),
            "mean_days": float(np.mean([r['duration_days'] for r in bull_runs])) if bull_runs else 0.0,
            "max_days": int(max([r['duration_days'] for r in bull_runs])) if bull_runs else 0,
        },
    }
    diag_path = reports / "Phase2" / "v5_p2.10_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.10 complete.")


if __name__ == "__main__":
    main()
