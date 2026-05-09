"""
V5 Phase 2.7 — Composite VIX-based finite state machine.

Phase 2.6 (saf VIX threshold) iki sorun gösterdi:
  - 2018 Q4: Bear spike kısa, hemen Neutral'a düştü (Bear yeterince sticky değil)
  - 2020 COVID: Spike çok büyük, recovery uzun (Bull'a geri dönüş gecikti)

Phase 2.7 çözümü:
  1) Asymmetric (hysteresis) thresholds:
       Neutral → Bear : VIX_z > +1.0  (giriş kolay)
       Bear → Neutral : VIX_z < +0.3  (çıkış için ortalamaya iyice dön)
       Neutral → Bull : VIX_z < -0.5 AND SP500_5d > 0
       Bull → Neutral : VIX_z > 0.0
  2) Minimum dwell time:
       Bear min 20 gün, Bull min 30 gün (Neutral transition zone)
  3) VIX velocity override:
       Bear iken VIX_z 5d Δ < -0.8 AND SP500_5d > 0 → Bear'dan Neutral'a
       fast-track (dwell time bypass — 2020 fix)

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_composite_rule_v5.csv
  reports/Phase2/v5_p2.7_composite_rule_state_diagram.png
  reports/Phase2/v5_p2.7_composite_rule_timeline.png
  reports/Phase2/v5_p2.7_composite_rule_distribution.png
  reports/Phase2/v5_p2.7_composite_rule_diagnostics.json
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
from matplotlib.patches import Patch, FancyArrowPatch

from src.labels.v5_regime_labels import (
    CompositeVIXRegimeClassifier, BULL_BEAR_LABELS,
)

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
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


def plot_state_diagram(model, out: Path):
    """3-state finite-state-machine diagram with transition labels."""
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = {"Bull": (0.15, 0.5), "Neutral": (0.5, 0.5), "Bear": (0.85, 0.5)}
    for regime, (x, y) in positions.items():
        circle = plt.Circle((x, y), 0.08, color=REGIME_COLORS[regime],
                             ec="black", lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, regime, ha="center", va="center",
                fontsize=13, fontweight="bold", zorder=4)
        ax.text(x, y - 0.18, f"min dwell:\n"
                f"{model.bull_min_dwell if regime == 'Bull' else (model.bear_min_dwell if regime == 'Bear' else 0)} d",
                ha="center", va="center", fontsize=8.5, color="#444")

    def arrow(p1, p2, label, offset=0.08, color="black", style="->"):
        x1, y1 = p1; x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        sx, sy = x1 + ux * 0.085, y1 + uy * 0.085
        ex, ey = x2 - ux * 0.085, y2 - uy * 0.085
        sy += offset; ey += offset
        arr = FancyArrowPatch((sx, sy), (ex, ey),
                               arrowstyle=style, mutation_scale=15,
                               color=color, lw=1.4, zorder=2)
        ax.add_patch(arr)
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx, my + 0.04, label, ha="center", va="bottom",
                fontsize=8.5, color=color,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=0.8))

    arrow(positions["Bull"], positions["Neutral"],
          f"VIX_z > {model.bull_exit_threshold}", offset=0.06)
    arrow(positions["Neutral"], positions["Bull"],
          f"VIX_z < {model.bull_entry_threshold}\n& SP500_5d > 0", offset=-0.06)
    arrow(positions["Neutral"], positions["Bear"],
          f"VIX_z > {model.bear_entry_threshold}", offset=0.06)
    arrow(positions["Bear"], positions["Neutral"],
          f"VIX_z < {model.bear_exit_threshold}", offset=-0.06)
    arrow(positions["Bear"], positions["Neutral"],
          f"velocity override:\nΔVIX_z[5d] < {model.velocity_threshold}\n& SP500_5d > 0",
          offset=-0.18, color="darkred", style="->")

    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 0.95)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("V5 Phase 2.7 — Composite VIX Regime FSM\n"
                 "Hysteresis + Dwell time + Velocity override",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def plot_timeline(btc_close, eth_close, btc_r, eth_r, pre_r, model, out: Path):
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1, 1]})

    raw_dir = PROJECT_ROOT / "data" / "raw"
    proc_dir = PROJECT_ROOT / "data" / "processed"
    sp500 = pd.read_csv(raw_dir / "v5_macro_risk.csv",
                        index_col=0, parse_dates=True)["SP500"].dropna()
    derived = pd.read_csv(proc_dir / "macro_derived_pretrain_v5.csv",
                          index_col=0, parse_dates=True)
    vix_z = derived["VIX_zscore_long"].dropna()

    ax = axes[0]
    _shade_regimes(ax, pd.Series(np.nan, index=pre_r.index),
                   pre_r["regime_label"], log=False, lw=0)
    ax.plot(sp500.index, sp500.values, color="#1f77b4", lw=0.9,
            label="S&P 500 (left, linear)", zorder=3)
    ax.set_ylabel("S&P 500", color="#1f77b4")
    ax.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(vix_z.index, vix_z.values, color="#d62728", lw=0.7, alpha=0.8,
             label="VIX z-score long (right)", zorder=4)
    ax2.axhline(model.bear_entry_threshold, color="#d62728", ls="--", lw=0.5,
                alpha=0.5, label=f"Bear entry (>{model.bear_entry_threshold})")
    ax2.axhline(model.bear_exit_threshold, color="#d62728", ls=":", lw=0.5,
                alpha=0.5, label=f"Bear exit (<{model.bear_exit_threshold})")
    ax2.axhline(model.bull_entry_threshold, color="green", ls="--", lw=0.5,
                alpha=0.5, label=f"Bull entry (<{model.bull_entry_threshold})")
    ax2.set_ylabel("VIX z-score", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False); ax2.spines["top"].set_visible(False)

    if "velocity_override" in pre_r.columns:
        vel_dates = pre_r.index[pre_r["velocity_override"]]
        if len(vel_dates) > 0:
            ax.scatter(vel_dates, [sp500.reindex(vel_dates).max() * 1.05] * len(vel_dates),
                       marker="v", color="darkred", s=40, zorder=5,
                       label="Velocity override fired")

    ax.set_title("Pre-train context — S&P 500 + VIX z-score + Composite regime shading\n"
                 "(velocity overrides marked with red triangles)",
                 fontsize=10, fontweight="bold")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=7.5)

    _shade_regimes(axes[1], btc_close, btc_r["regime_label"], log=True, lw=1.0)
    axes[1].set_ylabel("BTC (log)")
    axes[1].set_title("BTC + Composite regime", fontsize=10, fontweight="bold")

    _shade_regimes(axes[2], eth_close, eth_r["regime_label"], log=True, lw=1.0)
    axes[2].set_ylabel("ETH (log)")
    axes[2].set_title("ETH + Composite regime", fontsize=10, fontweight="bold")

    pre_close = pd.Series(1.0, index=pre_r.index)
    _shade_regimes(axes[3], pre_close, pre_r["regime_label"], log=False, lw=0.2)
    axes[3].set_yticks([]); axes[3].grid(False)
    axes[3].set_title("Pre-train regime band only (compact)",
                      fontsize=10, fontweight="bold")

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    handles = [Patch(facecolor=c, alpha=0.5, label=r) for r, c in REGIME_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("V5 Phase 2.7 — Composite VIX FSM "
                 "(Hysteresis + Dwell + Velocity override)",
                 fontsize=12, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
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
    fig, ax = plt.subplots(figsize=(12, 4.5))
    pivot.plot(kind="barh", stacked=True, ax=ax,
               color=[REGIME_COLORS[r] for r in BULL_BEAR_LABELS],
               edgecolor="black", lw=0.5)
    ax.set_xlabel("% of days")
    ax.set_xlim(0, 100)
    ax.set_title("V5 Phase 2.7 — Composite Rule Regime Distribution",
                 fontsize=12, fontweight="bold")
    for i, period in enumerate(pivot.index):
        cum = 0
        for r in BULL_BEAR_LABELS:
            v = pivot.loc[period, r]
            if v > 4:
                ax.text(cum + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold")
            cum += v
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out.relative_to(PROJECT_ROOT)}")


def main():
    print("V5 Phase 2.7 — Composite VIX Regime FSM")
    print("=" * 70)
    proc = PROJECT_ROOT / "data" / "processed"
    reports = PROJECT_ROOT / "reports"
    (reports / "Phase2").mkdir(parents=True, exist_ok=True)

    pretrain = pd.read_csv(proc / "macro_derived_pretrain_v5.csv",
                           index_col=0, parse_dates=True)
    pretrain = pretrain.dropna(subset=["VIX_zscore_long", "SP500_log_return_5d"])
    btc = pd.read_csv(proc / "btc_aligned_v5.csv", index_col=0, parse_dates=True)
    eth = pd.read_csv(proc / "eth_aligned_v5.csv", index_col=0, parse_dates=True)

    print("\n[1] Fit Composite VIX FSM (hysteresis + dwell + velocity)")
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
    ).fit(pretrain)
    print(f"  Bear entry: VIX_z > {model.bear_entry_threshold}")
    print(f"  Bear exit:  VIX_z < {model.bear_exit_threshold}")
    print(f"  Bull entry: VIX_z < {model.bull_entry_threshold} AND SP500_5d > 0")
    print(f"  Bull exit:  VIX_z > {model.bull_exit_threshold}")
    print(f"  Bear min dwell: {model.bear_min_dwell}d, Bull min dwell: {model.bull_min_dwell}d")
    print(f"  Velocity override: ΔVIX_z[{model.velocity_window}d] < {model.velocity_threshold}"
          f" AND SP500_5d > {model.velocity_sp500_min}")

    plot_state_diagram(model, reports / "Phase2" / "v5_p2.7_composite_rule_state_diagram.png")

    print(f"\n[2] Inference (FSM forward pass)")
    pre_r = model.predict(pretrain)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_composite_rule_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_composite_rule_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_composite_rule_v5.csv")
    n_overrides = int(pre_r["velocity_override"].sum())
    override_dates = pre_r.index[pre_r["velocity_override"]]
    if n_overrides > 0:
        print(f"  Velocity override fired: {n_overrides} times")
        for d in override_dates:
            print(f"    {d.date()}")

    print(f"\n[3] Plots")
    plot_timeline(btc["Close"].loc[btc_r.index],
                  eth["Close"].loc[eth_r.index],
                  btc_r, eth_r, pre_r, model,
                  reports / "Phase2" / "v5_p2.7_composite_rule_timeline.png")
    plot_distribution(btc_r, eth_r, pre_r,
                      reports / "Phase2" / "v5_p2.7_composite_rule_distribution.png")

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
    print(f"  # Bear runs: {len(bear_runs)}, mean duration: "
          f"{np.mean([r['duration_days'] for r in bear_runs]):.0f}d, "
          f"max: {max([r['duration_days'] for r in bear_runs]):.0f}d")
    print(f"  # Bull runs: {len(bull_runs)}, mean duration: "
          f"{np.mean([r['duration_days'] for r in bull_runs]):.0f}d, "
          f"max: {max([r['duration_days'] for r in bull_runs]):.0f}d")
    print(f"  # Neutral runs: {len(neutral_runs)}, mean: "
          f"{np.mean([r['duration_days'] for r in neutral_runs]):.0f}d")

    diagnostics = {
        "phase": "V5 Phase 2.7 — Composite VIX Regime FSM",
        "method": "Hysteresis + dwell time + velocity override",
        "thresholds": {
            "bear_entry": model.bear_entry_threshold,
            "bear_exit": model.bear_exit_threshold,
            "bull_entry": model.bull_entry_threshold,
            "bull_exit": model.bull_exit_threshold,
        },
        "dwell_time": {
            "bear_min_days": model.bear_min_dwell,
            "bull_min_days": model.bull_min_dwell,
        },
        "velocity_override": {
            "window_days": model.velocity_window,
            "threshold": model.velocity_threshold,
            "sp500_min": model.velocity_sp500_min,
            "n_fired_pretrain": n_overrides,
            "fired_dates": [str(d.date()) for d in override_dates],
        },
        "pretrain_vix_z_stats": model.pretrain_vix_z_stats_,
        "distribution_pct": distribution,
        "bear_runs_summary": {
            "count": len(bear_runs),
            "mean_days": float(np.mean([r['duration_days'] for r in bear_runs])),
            "max_days": int(max([r['duration_days'] for r in bear_runs])),
        },
        "bull_runs_summary": {
            "count": len(bull_runs),
            "mean_days": float(np.mean([r['duration_days'] for r in bull_runs])),
            "max_days": int(max([r['duration_days'] for r in bull_runs])),
        },
        "all_runs_pretrain": runs,
    }
    diag_path = reports / "Phase2" / "v5_p2.7_composite_rule_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.7 complete.")


if __name__ == "__main__":
    main()
