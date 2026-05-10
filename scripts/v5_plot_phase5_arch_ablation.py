"""V5 Phase 5.1 — Architecture Ablation visualization.

Reads:
  reports/Phase5.1_arch_ablation/v5_p5_arch_overall.csv
  reports/Phase5.1_arch_ablation/v5_p5_arch_backtest_summary.csv
  reports/Phase5.1_arch_ablation/v5_p5_arch_equity_curves_{btc,eth}.csv

Outputs to reports/Phase5.1_arch_ablation/:
  v5_p5_arch_summary_metrics.png       3-panel Sharpe/Return/MaxDD per arch (best per arch)
  v5_p5_arch_full_heatmap.png          full grid heatmap (arch x rule x model) for Sharpe/Return/MaxDD
  v5_p5_arch_equity_best.png           best equity curve per arch + B&H, BTC and ETH
  v5_p5_arch_f1m_comparison.png        F1 macro per arch (4 model x 4 arch x 2 asset)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ASSETS = ["btc", "eth"]
ASSET_COLORS = {"btc": "#F7931A", "eth": "#627EEA"}
ARCHS = ["flat", "2stage_trend", "2stage_macro", "3stage_full"]
ARCH_LABELS = {
    "flat":         "Flat\n(6 osc)",
    "2stage_trend": "2-Stage Trend\n(+ S1)",
    "2stage_macro": "2-Stage Macro\n(+ S2)",
    "3stage_full":  "3-Stage Full\n(S1+S2+osc)",
}
ARCH_FEATURE_COUNT = {"flat": 6, "2stage_trend": 12, "2stage_macro": 10, "3stage_full": 16}
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
RULES = ["stateful", "defensive", "prob_weighted"]


def _load():
    rep = PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation"
    overall = pd.read_csv(rep / "v5_p5_arch_overall.csv")
    backtest = pd.read_csv(rep / "v5_p5_arch_backtest_summary.csv")
    return rep, overall, backtest


def plot_summary_metrics(out: Path, backtest: pd.DataFrame):
    """3-panel: Sharpe / Return / MaxDD per arch (best per arch+asset)."""
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # B&H reference
    bh = backtest[backtest["arch"] == "buy_and_hold"].set_index("asset")
    bt = backtest[backtest["arch"] != "buy_and_hold"].copy()

    # Best Sharpe per (asset, arch)
    best = bt.loc[bt.groupby(["asset", "arch"])["annualized_sharpe"].idxmax()].copy()

    metrics = [
        ("annualized_sharpe", "Sharpe Ratio (annualized)", "Sharpe"),
        ("total_return",      "Total Return",              "Return"),
        ("max_drawdown",      "Max Drawdown",              "MaxDD"),
    ]

    x = np.arange(len(ARCHS))
    w = 0.36

    for ax, (col, title, ylabel) in zip(axes, metrics):
        for i, asset in enumerate(ASSETS):
            sub = best[best["asset"] == asset].set_index("arch").reindex(ARCHS)
            offset = -w/2 if asset == "btc" else w/2
            bars = ax.bar(x + offset, sub[col], width=w,
                          color=ASSET_COLORS[asset], edgecolor="black", linewidth=0.5,
                          label=asset.upper())
            for bar, v in zip(bars, sub[col]):
                if pd.isna(v):
                    continue
                if col == "total_return":
                    txt = f"{v:+.0%}"
                elif col == "annualized_sharpe":
                    txt = f"{v:+.2f}"
                else:
                    txt = f"{v:.0%}"
                y_off = 0.005 if v >= 0 else -0.02
                ax.text(bar.get_x() + bar.get_width()/2, v + y_off, txt,
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=8, fontweight="bold")
        # B&H horizontal lines per asset
        for asset in ASSETS:
            v = bh.loc[asset, col]
            ax.axhline(v, color=ASSET_COLORS[asset], ls="--", lw=1.0, alpha=0.6,
                       label=f"{asset.upper()} B&H ({v:+.2f})" if col == "annualized_sharpe"
                       else f"{asset.upper()} B&H ({v:+.0%})")

        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS], fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="black", lw=0.5)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Phase 5.1 — Architecture Ablation: best (rule + model) per architecture vs B&H\n"
                 "BTC: hiyerarşik mimari yardımcı (3-stage best). ETH: flat best — küçük dataset overfit eder.",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out / "v5_p5_arch_summary_metrics.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_full_heatmap(out: Path, backtest: pd.DataFrame):
    """Full grid heatmap: 4 arch x 3 rule x 4 model = 48 cells per asset, per metric."""
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    bt = backtest[backtest["arch"] != "buy_and_hold"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    metrics = [
        ("annualized_sharpe", "Sharpe", "RdYlGn",  None,  None),
        ("total_return",      "Return", "RdYlGn",  None,  None),
        ("max_drawdown",      "MaxDD",  "RdYlGn_r", None, None),
    ]

    for r_i, asset in enumerate(ASSETS):
        sub = bt[bt["asset"] == asset]
        for c_i, (col, title, cmap, _, _) in enumerate(metrics):
            ax = axes[r_i, c_i]
            # Pivot: row = (arch, rule), col = model
            sub2 = sub.copy()
            sub2["arch_rule"] = sub2["arch"].astype(str) + "/" + sub2["rule"].astype(str)
            pivot = sub2.pivot_table(index="arch_rule", columns="model",
                                      values=col)
            # Order rows: (arch, rule)
            row_order = [f"{a}/{r}" for a in ARCHS for r in RULES]
            pivot = pivot.reindex(index=row_order, columns=MODELS)
            arr = pivot.values
            v_lim = max(abs(np.nanmin(arr)), abs(np.nanmax(arr)))
            if col == "max_drawdown":
                im = ax.imshow(arr, cmap=cmap, vmin=-v_lim, vmax=0, aspect="auto")
            else:
                im = ax.imshow(arr, cmap=cmap, vmin=-v_lim, vmax=v_lim, aspect="auto")

            ax.set_xticks(range(len(MODELS)))
            ax.set_xticklabels(MODELS, fontsize=8, rotation=15)
            ax.set_yticks(range(len(row_order)))
            ax.set_yticklabels(row_order, fontsize=7)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    if np.isnan(v): continue
                    if col == "annualized_sharpe":
                        txt = f"{v:+.2f}"
                    elif col == "total_return":
                        txt = f"{v:+.0%}"
                    else:
                        txt = f"{v:.0%}"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7, fontweight="bold")
            # Arch group separator lines
            for k in range(1, len(ARCHS)):
                ax.axhline(k * len(RULES) - 0.5, color="black", lw=0.6, alpha=0.5)
            ax.set_title(f"{asset.upper()} — {title}", fontsize=11, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    fig.suptitle("Phase 5.1 — Architecture Ablation FULL grid (4 arch × 3 rule × 4 model = 48 combos)\n"
                 "Lines separate arch groups. B&H reference: BTC Sharpe 0.95 / Return +29.7 / MaxDD -76%, "
                 "ETH Sharpe 0.26 / Return -7% / MaxDD -72%",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout()
    out_path = out / "v5_p5_arch_full_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_equity_best(out: Path, backtest: pd.DataFrame):
    """Best equity curve per arch per asset, vs B&H."""
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(2, 1, figsize=(15, 9))
    rep = PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation"

    arch_colors = {
        "flat":          "#888888",
        "2stage_trend":  "#3a6fb0",
        "2stage_macro":  "#bf6e1d",
        "3stage_full":   "#cc4444",
    }

    bt = backtest[backtest["arch"] != "buy_and_hold"]

    for ax, asset in zip(axes, ASSETS):
        eq = pd.read_csv(rep / f"v5_p5_arch_equity_curves_{asset}.csv",
                         index_col=0, parse_dates=True)

        for arch in ARCHS:
            sub = bt[(bt["asset"] == asset) & (bt["arch"] == arch)]
            best_row = sub.loc[sub["annualized_sharpe"].idxmax()]
            best_col = f"{arch}_{best_row['model']}_{best_row['rule']}"
            if best_col not in eq.columns:
                continue
            ax.semilogy(eq.index, eq[best_col],
                        color=arch_colors[arch], lw=1.5,
                        label=f"{ARCH_LABELS[arch]}: {best_row['model']}/{best_row['rule']}  "
                              f"(S={best_row['annualized_sharpe']:+.2f}, "
                              f"R={best_row['total_return']:+.0%}, "
                              f"DD={best_row['max_drawdown']:.0%})")
        # B&H
        ax.semilogy(eq.index, eq["BUY_AND_HOLD"], color="black", ls="--", lw=1.2,
                    label=f"Buy & Hold")
        ax.axhline(1.0, color="grey", ls=":", lw=0.4, alpha=0.5)
        ax.set_ylabel("Equity (log, $1 start)")
        ax.set_title(f"{asset.upper()} — Best Stage 3 model per architecture vs B&H",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(eq.index.min(), eq.index.max())

    fig.suptitle("Phase 5.1 — Architecture Ablation Equity Curves\n"
                 "Each architecture's best (rule, model) combination",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    out_path = out / "v5_p5_arch_equity_best.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_f1m_comparison(out: Path, overall: pd.DataFrame):
    """F1 macro per arch x model for both assets."""
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    x = np.arange(len(ARCHS))
    w = 0.21

    for ax, asset in zip(axes, ASSETS):
        sub = overall[overall["asset"] == asset]
        for i, model in enumerate(MODELS):
            ys = []
            for arch in ARCHS:
                row = sub[(sub["model"] == model) & (sub["arch"] == arch)]
                ys.append(row["f1_macro"].iloc[0] if len(row) else np.nan)
            ax.bar(x + (i - 1.5) * w, ys, width=w,
                   label=model, edgecolor="black", linewidth=0.4)
        ax.axhline(0.333, color="grey", ls="--", lw=0.8, label="chance (0.33)")
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABELS[a] for a in ARCHS], fontsize=9)
        ax.set_title(f"{asset.upper()} — F1 macro per architecture × model",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("F1 macro")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 0.45)

    fig.suptitle("Phase 5.1 — Architecture × Model F1 macro\n"
                 "Classification metric (frame-level), all walk-forward outer OOF",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out / "v5_p5_arch_f1m_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    rep, overall, backtest = _load()
    plot_summary_metrics(rep, backtest)
    plot_full_heatmap(rep, backtest)
    plot_equity_best(rep, backtest)
    plot_f1m_comparison(rep, overall)
    print(f"\nAll plots in {rep.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
