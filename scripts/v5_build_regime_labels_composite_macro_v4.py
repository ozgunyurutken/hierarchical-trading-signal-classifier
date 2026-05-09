"""V5 Phase 2.11 — Composite Macro FSM v4.

Phase 2.10 üzerine 2 ek mekanizma (kullanıcı feedback):
  - velocity_window 5 → 10 (2008 GFC Bear çıkışı erken: 2009 Mar civarı)
  - Bear velocity entry (Rule 6, yeni): Neutral → Bear hızlı geçiş
    ΔVIX_z[5d] > +0.6 AND SP500_5d < -1.5%
    → 2025 Apr Liberation Day, 2018 Feb volmageddon erken Bear

Korunan:
  - Phase 2.10'un 5 mekanizması (hysteresis, dwell, velocity, YC, macro stress, bull velocity)
  - Tüm threshold'lar Phase 2.10 ile aynı (sadece velocity_window 10)

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_composite_macro_v4_v5.csv
  reports/Phase2/v5_p2.11_*.png
  reports/Phase2/v5_p2.11_diagnostics.json
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

plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240,
                     "font.size": 14,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

REGIME_COLORS = {"Bull": "#7ec27e", "Neutral": "#f0c870", "Bear": "#e07e7e"}


def shade_regimes(ax, regime):
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


def plot_decade_panel(ax, ax2, sp500, vix_z, regime, pre_r, start, end, model, title):
    sp_slice = sp500.loc[(sp500.index >= start) & (sp500.index <= end)]
    vix_slice = vix_z.loc[(vix_z.index >= start) & (vix_z.index <= end)]
    reg_slice = regime.loc[(regime.index >= start) & (regime.index <= end)]
    pre_slice = pre_r.loc[(pre_r.index >= start) & (pre_r.index <= end)]

    shade_regimes(ax, reg_slice)
    ax.plot(sp_slice.index, sp_slice.values, color="#1f77b4", lw=1.4, zorder=3)
    ax.set_ylabel("S&P 500", color="#1f77b4", fontsize=14)
    ax.tick_params(axis="y", labelcolor="#1f77b4", labelsize=12)

    ax2.plot(vix_slice.index, vix_slice.values, color="#d62728",
             lw=0.9, alpha=0.85, zorder=4)
    ax2.axhline(model.bear_entry_threshold, color="#d62728", ls="--",
                lw=0.7, alpha=0.5)
    ax2.axhline(model.bull_entry_threshold, color="green", ls="--",
                lw=0.7, alpha=0.5)
    ax2.set_ylabel("VIX z-score", color="#d62728", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=12)
    ax2.grid(False); ax2.spines["top"].set_visible(False)

    sp_max = sp_slice.max() if not sp_slice.empty else 1.0
    bv = pre_slice.index[pre_slice["bull_velocity_entry"]]
    if len(bv):
        ax.scatter(bv, [sp_max * 1.04] * len(bv), marker="^",
                   color="darkgreen", s=70, zorder=5)
    bev = pre_slice.index[pre_slice["bear_velocity_entry"]]
    if len(bev):
        ax.scatter(bev, [sp_max * 1.04] * len(bev), marker="v",
                   color="darkred", s=80, zorder=5)
    vov = pre_slice.index[pre_slice["velocity_override"]]
    if len(vov):
        ax.scatter(vov, [sp_max * 1.10] * len(vov), marker="o",
                   color="orange", s=50, zorder=5)
    yc = pre_slice.index[pre_slice["yield_curve_override"]]
    if len(yc):
        ax.scatter(yc, [sp_max * 1.16] * len(yc), marker="s",
                   color="purple", s=60, zorder=5)
    ms = pre_slice.index[pre_slice["macro_stress_override"]]
    if len(ms):
        ax.scatter(ms, [sp_max * 1.16] * len(ms), marker="D",
                   color="brown", s=60, zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_pretrain_3decade(pre_r, sp500, vix_z, model, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(20, 22))
    decades = [
        (pd.Timestamp("2000-01-01"), pd.Timestamp("2010-12-31"),
         "2000-2010 — Dot-com, GFC, recovery (kullanıcı: 2008 erken çıkış)"),
        (pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31"),
         "2010-2020 — ZIRP, 2018 volmageddon (Bear vel?), COVID"),
        (pd.Timestamp("2020-01-01"), pd.Timestamp("2026-01-31"),
         "2020-2026 — V-shape, 2022 Bear, 2025 Apr tariff (Bear vel?)"),
    ]
    twin_axes = [ax.twinx() for ax in axes]
    for (start, end, title), ax, ax2 in zip(decades, axes, twin_axes):
        plot_decade_panel(ax, ax2, sp500, vix_z, pre_r["regime_label"],
                          pre_r, start, end, model, title)
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    handles += [
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="darkgreen",
                   markersize=10, label="Bull velocity entry (V-shape)"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="darkred",
                   markersize=10, label="Bear velocity entry (rapid)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="orange",
                   markersize=10, label="Bear→Neutral velocity"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="purple",
                   markersize=10, label="Yield curve override"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="brown",
                   markersize=10, label="DXY+M2 macro stress"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11,
               bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("V5 Phase 2.11 — Pre-train Overview by Decade",
                 fontsize=16, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
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

    shade_regimes(axes[0], pre_slice["regime_label"])
    axes[0].plot(sp_slice.index, sp_slice.values, color="black", lw=1.4)
    axes[0].set_ylabel("S&P 500", fontsize=14)
    axes[0].set_title(f"S&P 500 ({start.date()} → {end.date()})",
                      fontsize=14, fontweight="bold")

    if not btc_slice.empty:
        shade_regimes(axes[1], btc_r_slice["regime_label"])
        axes[1].semilogy(btc_slice.index, btc_slice.values, color="black", lw=1.5)
        axes[1].set_ylabel("BTC (log)", fontsize=14)
        axes[1].set_title(f"BTC ({start.date()} → {end.date()})",
                          fontsize=14, fontweight="bold")
    if not eth_slice.empty:
        shade_regimes(axes[2], eth_r_slice["regime_label"])
        axes[2].semilogy(eth_slice.index, eth_slice.values, color="black", lw=1.5)
        axes[2].set_ylabel("ETH (log)", fontsize=14)
        axes[2].set_title(f"ETH ({start.date()} → {end.date()})",
                          fontsize=14, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
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


def main():
    print("V5 Phase 2.11 — Composite Macro FSM v4")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    raw = PROJECT_ROOT / "data" / "raw"
    reports = PROJECT_ROOT / "reports"

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True)
    sp500_raw = pd.read_csv(raw / "v5_macro_risk.csv",
                            index_col=0, parse_dates=True)["SP500"]
    pretrain["SP500_60d_return"] = (sp500_raw / sp500_raw.shift(60) - 1.0).reindex(pretrain.index)
    pretrain_clean = pretrain.dropna(subset=["VIX_zscore_long",
                                              "SP500_log_return_5d",
                                              "Yield_Curve_10Y_2Y",
                                              "DXY_zscore_long",
                                              "M2_yoy_change",
                                              "SP500_60d_return"])
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print("\n[1] Fit Composite Macro FSM v4")
    model = CompositeVIXRegimeClassifier(
        bear_entry_threshold=1.0,
        bear_exit_threshold=0.3,
        bull_entry_threshold=-0.5,
        bull_exit_threshold=0.0,
        bear_min_dwell=20,
        bull_min_dwell=30,
        # Phase 2.11: Bear→Neutral velocity window 5 → 10
        velocity_window=10,
        velocity_threshold=-0.8,
        velocity_sp500_min=0.0,
        initial_regime="Neutral",
        enable_yield_curve_override=True,
        yield_curve_inverted_threshold=0.0,
        yield_curve_persistence_window=60,
        yield_curve_blocks_bull_entry=False,
        yc_requires_sp500_weakness=False,
        enable_macro_stress_override=True,
        dxy_strong_threshold=0.7,
        m2_low_threshold=0.040,
        macro_stress_window=30,
        macro_stress_combine="AND",
        enable_bull_velocity_entry=True,
        bull_velocity_window=30,
        bull_velocity_threshold=-0.6,
        bull_velocity_sp500_min=0.02,
        # Phase 2.11 YENİ: Bear velocity entry (rapid escalation)
        enable_bear_velocity_entry=True,
        bear_velocity_entry_window=5,
        bear_velocity_entry_threshold=0.6,             # 2025 Apr Liberation Day yakalanır
        bear_velocity_entry_sp500_max=-0.015,
        bear_reentry_min_neutral_dwell=10,             # Bear→Neutral→Bear flip-flop önler
    ).fit(pretrain_clean)

    print(f"  Bear→Neutral velocity: ΔVIX_z[{model.velocity_window}d] < {model.velocity_threshold} (window 5→10)")
    print(f"  Bull velocity entry:   ΔVIX_z[{model.bull_velocity_window}d] < {model.bull_velocity_threshold}")
    print(f"  Bear velocity entry:   ΔVIX_z[{model.bear_velocity_entry_window}d] > {model.bear_velocity_entry_threshold}")
    print(f"                          AND SP500_5d < {model.bear_velocity_entry_sp500_max}")

    print(f"\n[2] Inference")
    pre_r = model.predict(pretrain_clean)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_composite_macro_v4_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_composite_macro_v4_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_composite_macro_v4_v5.csv")

    n_velocity = int(pre_r["velocity_override"].sum())
    n_yc = int(pre_r["yield_curve_override"].sum())
    n_ms = int(pre_r["macro_stress_override"].sum())
    n_bv = int(pre_r["bull_velocity_entry"].sum())
    n_bev = int(pre_r["bear_velocity_entry"].sum())
    print(f"  Bear→Neutral velocity: {n_velocity} times")
    print(f"  Bull velocity entry:   {n_bv} times")
    print(f"  Bear velocity entry:   {n_bev} times")
    if n_bev > 0:
        for d in pre_r.index[pre_r["bear_velocity_entry"]]:
            print(f"    BeV: {d.date()}")
    print(f"  Yield curve override:  {n_yc} times")
    print(f"  Macro stress override: {n_ms} times")

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
    print(f"\n[3] 2008 GFC Bear run check:")
    for r in bear_runs:
        if "2008" in r["start"] or "2009" in r["start"]:
            print(f"    {r['regime']}: {r['start']} → {r['end']} ({r['duration_days']}d)")
    print(f"\n2025 Q1-Q2 Bear run check:")
    for r in bear_runs:
        if "2025" in r["start"]:
            print(f"    {r['regime']}: {r['start']} → {r['end']} ({r['duration_days']}d)")

    print(f"\n[4] Plots")
    sp500 = sp500_raw.dropna()
    derived = pretrain
    vix_z = derived["VIX_zscore_long"].dropna()

    plot_pretrain_3decade(pre_r, sp500, vix_z, model,
                          reports / "Phase2" / "v5_p2.11_01_pretrain_overview_decades.png")
    plot_zoom(pd.Timestamp("2007-06-01"), pd.Timestamp("2010-06-30"),
              btc["Close"].loc[btc_r.index], eth["Close"].loc[eth_r.index],
              btc_r, eth_r, pre_r, sp500,
              "V5 Phase 2.11 — 2008 GFC Zoom (kullanıcı: erken çıkış)",
              reports / "Phase2" / "v5_p2.11_02_2008_gfc_zoom.png")
    plot_zoom(pd.Timestamp("2024-06-01"), pd.Timestamp("2025-12-31"),
              btc["Close"].loc[btc_r.index], eth["Close"].loc[eth_r.index],
              btc_r, eth_r, pre_r, sp500,
              "V5 Phase 2.11 — 2025 Q1-Q2 Zoom (kullanıcı: 2025 Apr tarif erken tespit)",
              reports / "Phase2" / "v5_p2.11_03_2025_zoom.png")
    plot_zoom(pd.Timestamp("2019-06-01"), pd.Timestamp("2022-12-31"),
              btc["Close"].loc[btc_r.index], eth["Close"].loc[eth_r.index],
              btc_r, eth_r, pre_r, sp500,
              "V5 Phase 2.11 — 2020-2022 Zoom (V-shape + 2022 rally korunur)",
              reports / "Phase2" / "v5_p2.11_04_2020_2022_zoom.png")

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

    diagnostics = {
        "phase": "V5 Phase 2.11 — Composite Macro FSM v4",
        "user_feedback_addressed": [
            "2008 GFC: Bear→Neutral velocity_window 5→10 (cumulative drop yakalanır)",
            "2025 Apr Liberation Day: Bear velocity entry (rapid escalation override)",
        ],
        "velocity_window": model.velocity_window,
        "bear_velocity_entry": {
            "window": model.bear_velocity_entry_window,
            "threshold": model.bear_velocity_entry_threshold,
            "sp500_max": model.bear_velocity_entry_sp500_max,
            "n_fired": n_bev,
            "fired_dates": [str(d.date()) for d in pre_r.index[pre_r["bear_velocity_entry"]]],
        },
        "n_bull_velocity_entry": n_bv,
        "n_yield_curve_override": n_yc,
        "n_macro_stress_override": n_ms,
        "distribution_pct": distribution,
        "bear_runs": bear_runs,
    }
    diag_path = reports / "Phase2" / "v5_p2.11_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.11 complete.")


if __name__ == "__main__":
    main()
