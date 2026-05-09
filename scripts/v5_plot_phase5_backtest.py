"""V5 Phase 5 — Backtest visualization.

Outputs to reports/Phase5/:
  v5_p5_equity_best_vs_bh.png       Best per-asset model equity vs B&H
  v5_p5_equity_grid.png             4 model x 3 rule equity grid per asset
  v5_p5_summary_heatmap.png         Sharpe heatmap (model x rule, BTC + ETH)
  v5_p5_risk_return_scatter.png     Sharpe vs Return scatter, rule-coded
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
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]
MODEL_COLORS = {
    "xgboost":       "#cc4444",
    "lightgbm":      "#3a6fb0",
    "random_forest": "#2a7a2a",
    "mlp":           "#bf6e1d",
}
RULES = ["stateful", "defensive", "prob_weighted"]
RULE_LINESTYLES = {"stateful": "-", "defensive": "--", "prob_weighted": ":"}


def _load(asset: str):
    rep = PROJECT_ROOT / "reports" / "Phase5"
    summary = pd.read_csv(rep / "v5_p5_backtest_summary.csv")
    summary = summary[summary["asset"] == asset]
    eq = pd.read_csv(rep / f"v5_p5_equity_curves_{asset}.csv",
                     index_col=0, parse_dates=True)
    return summary, eq


def plot_best_vs_bh(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=False)
    for ax, asset in zip(axes, ASSETS):
        summary, eq = _load(asset)
        # Best by Sharpe (excluding B&H)
        ranked = summary[summary["rule"] != "buy_and_hold"].sort_values(
            "annualized_sharpe", ascending=False
        )
        best = ranked.iloc[0]
        best_key = f"{best['model']}_{best['rule']}"

        ax.semilogy(eq.index, eq[best_key],
                    color=ASSET_COLORS[asset], lw=1.6,
                    label=f"BEST: {best['model']}/{best['rule']}  "
                          f"(Sharpe {best['annualized_sharpe']:+.2f}, "
                          f"return {best['total_return']:+.1%}, "
                          f"MaxDD {best['max_drawdown']:.1%})")
        ax.semilogy(eq.index, eq["BUY_AND_HOLD"],
                    color="#888888", lw=1.4, ls="--",
                    label=f"Buy & Hold  "
                          f"(Sharpe {summary[summary['rule']=='buy_and_hold'].iloc[0]['annualized_sharpe']:+.2f}, "
                          f"return {summary[summary['rule']=='buy_and_hold'].iloc[0]['total_return']:+.1%}, "
                          f"MaxDD {summary[summary['rule']=='buy_and_hold'].iloc[0]['max_drawdown']:.1%})")
        ax.axhline(1.0, color="black", ls=":", lw=0.5, alpha=0.5)
        ax.set_ylabel("Equity (log, $1 start)")
        ax.set_title(f"{asset.upper()} — Best Stage 3 model vs Buy & Hold",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(eq.index.min(), eq.index.max())

    fig.suptitle("Phase 5 — Best Stage 3 trading rule vs Buy & Hold benchmark",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    out_path = out / "v5_p5_equity_best_vs_bh.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_equity_grid(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    for asset in ASSETS:
        summary, eq = _load(asset)
        bh_eq = eq["BUY_AND_HOLD"]
        bh_sharpe = summary[summary['rule'] == 'buy_and_hold'].iloc[0]['annualized_sharpe']
        bh_return = summary[summary['rule'] == 'buy_and_hold'].iloc[0]['total_return']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        for ax, model in zip(axes.flat, MODELS):
            for rule in RULES:
                key = f"{model}_{rule}"
                if key not in eq.columns:
                    continue
                row = summary[(summary['model'] == model) & (summary['rule'] == rule)]
                sharpe = row['annualized_sharpe'].iloc[0]
                ax.semilogy(eq.index, eq[key],
                            color=MODEL_COLORS[model], ls=RULE_LINESTYLES[rule], lw=1.4,
                            label=f"{rule:14s}  S={sharpe:+.2f}")
            ax.semilogy(eq.index, bh_eq,
                        color="#888888", ls="-", lw=1.0, alpha=0.7,
                        label=f"B&H            S={bh_sharpe:+.2f}")
            ax.axhline(1.0, color="black", ls=":", lw=0.5, alpha=0.5)
            ax.set_title(f"{model}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8, loc="upper left")
            ax.grid(True, alpha=0.3)
            ax.set_ylabel("Equity (log)")

        fig.suptitle(f"Phase 5 — {asset.upper()} equity curves: 4 models x 3 trading rules + B&H",
                     fontsize=13, fontweight="bold", y=0.995)
        fig.tight_layout()
        out_path = out / f"v5_p5_equity_grid_{asset}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_summary_heatmap(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    metrics = [
        ("annualized_sharpe", "Sharpe Ratio",   "RdYlGn",  None, None),
        ("total_return",      "Total Return",   "RdYlGn",  None, None),
        ("max_drawdown",      "Max Drawdown",   "RdYlGn_r", None, None),  # less negative = better
    ]

    for col_i, asset in enumerate(ASSETS):
        summary, _ = _load(asset)
        # Long-form to pivot
        for row_i, (metric, title, cmap, vmin, vmax) in enumerate(metrics):
            ax = axes[row_i, col_i]
            sub = summary[summary["rule"] != "buy_and_hold"]
            pivot = sub.pivot_table(index="model", columns="rule",
                                    values=metric).reindex(index=MODELS, columns=RULES)
            arr = pivot.values
            bh_v = summary[summary["rule"] == "buy_and_hold"].iloc[0][metric]

            v_lim = max(abs(np.nanmin(arr)), abs(np.nanmax(arr)), abs(bh_v))
            if metric == "max_drawdown":
                vmn, vmx = -v_lim, 0  # MaxDD is always non-positive
                im = ax.imshow(arr, cmap=cmap, vmin=vmn, vmax=vmx, aspect="auto")
            else:
                im = ax.imshow(arr, cmap=cmap, vmin=-v_lim, vmax=v_lim, aspect="auto")

            ax.set_xticks(range(len(RULES)))
            ax.set_xticklabels(RULES, fontsize=9)
            ax.set_yticks(range(len(MODELS)))
            ax.set_yticklabels(MODELS, fontsize=9)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    v = arr[i, j]
                    if metric == "total_return":
                        txt = f"{v:+.1%}"
                    elif metric == "annualized_sharpe":
                        txt = f"{v:+.2f}"
                    else:
                        txt = f"{v:.1%}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                            color="black", fontweight="bold")
            # B&H reference text
            if metric == "total_return":
                bh_txt = f"B&H: {bh_v:+.1%}"
            elif metric == "annualized_sharpe":
                bh_txt = f"B&H: {bh_v:+.2f}"
            else:
                bh_txt = f"B&H: {bh_v:.1%}"
            title_full = f"{asset.upper()} — {title}\n({bh_txt})"
            ax.set_title(title_full, fontsize=11, fontweight="bold")
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)

    fig.suptitle("Phase 5 — Backtest summary heatmap (model x rule)\n"
                 "Sharpe / Return / MaxDD — B&H benchmark in title",
                 fontsize=13, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = out / "v5_p5_summary_heatmap.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_risk_return_scatter(out: Path):
    plt.rcParams.update({"figure.dpi": 240, "savefig.dpi": 240})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, asset in zip(axes, ASSETS):
        summary, _ = _load(asset)
        for rule in RULES:
            sub = summary[summary["rule"] == rule]
            ax.scatter(sub["max_drawdown"], sub["annualized_sharpe"],
                       s=120, alpha=0.75,
                       label=rule)
            for _, row in sub.iterrows():
                ax.annotate(row["model"][:3],
                            (row["max_drawdown"], row["annualized_sharpe"]),
                            fontsize=7, ha="center", va="center",
                            color="white", fontweight="bold")
        bh = summary[summary["rule"] == "buy_and_hold"].iloc[0]
        ax.scatter(bh["max_drawdown"], bh["annualized_sharpe"],
                   marker="*", s=300, color="black", label="B&H", zorder=10)
        ax.set_xlabel("Max Drawdown (less negative = better)")
        ax.set_ylabel("Annualized Sharpe")
        ax.set_title(f"{asset.upper()}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)

    fig.suptitle("Phase 5 — Risk-return trade-off (Sharpe vs MaxDD)\n"
                 "Top-right corner = best (high Sharpe, low drawdown)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out / "v5_p5_risk_return_scatter.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def main():
    out = PROJECT_ROOT / "reports" / "Phase5"
    out.mkdir(parents=True, exist_ok=True)

    plot_best_vs_bh(out)
    plot_equity_grid(out)
    plot_summary_heatmap(out)
    plot_risk_return_scatter(out)

    print(f"\nAll plots in {out.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
