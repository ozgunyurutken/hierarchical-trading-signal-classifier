"""V5 Phase J — Master comparison across Phase 5.1 baseline / Phase 7
(regime-aware) / Phase 8 (calibration) / Phase 9 (MLP oversample).

Builds a single comparison table that the paper-writing teammate can
drop into Section 4.7-4.9. Also produces two diagnostic plots:

  reports/Phase10_summary/v5_p10_master_backtest.csv
  reports/Phase10_summary/v5_p10_master_overall.csv
  reports/Phase10_summary/fig_calibration_ece.png
  reports/Phase10_summary/fig_oversample_f1.png
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    out = PROJECT_ROOT / "reports" / "Phase10_summary"
    out.mkdir(parents=True, exist_ok=True)

    # --- Backtest master ---
    p51 = pd.read_csv(PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation" / "v5_p5_arch_backtest_summary.csv")
    p51 = p51[p51["model"] != "benchmark"].copy()
    p51["variant"] = "baseline"
    p51["phase"]   = "P5.1"

    p7 = pd.read_csv(PROJECT_ROOT / "reports" / "Phase7_regime_aware" / "v5_p7_backtest.csv")
    p7 = p7[p7["arch"] != "buy_and_hold"].copy()
    p7["variant"] = "regime_aware"
    p7["phase"]   = "P7"

    p8 = pd.read_csv(PROJECT_ROOT / "reports" / "Phase8_calibration" / "v5_p8_backtest_compare.csv")
    p8["variant"] = p8["variant"].map({"raw": "p8_raw", "cal": "p8_cal"})
    p8["phase"]   = "P8"

    p9 = pd.read_csv(PROJECT_ROOT / "reports" / "Phase9_mlp_oversample" / "v5_p9_backtest.csv")
    p9["phase"] = "P9"

    keep = ["phase", "variant", "asset", "arch", "model", "rule",
            "annualized_sharpe", "total_return", "max_drawdown",
            "n_trades", "win_rate"]
    master = pd.concat([
        p51.reindex(columns=keep),
        p7.reindex(columns=keep),
        p8.reindex(columns=keep),
        p9.reindex(columns=keep),
    ], ignore_index=True)
    master.to_csv(out / "v5_p10_master_backtest.csv", index=False)
    print(f"master backtest: {len(master)} rows -> {out.name}/v5_p10_master_backtest.csv")

    # --- Best per (asset, arch) per variant ---
    best = master.dropna(subset=["annualized_sharpe"]).copy()
    best = best.loc[best.groupby(["phase", "variant", "asset", "arch", "model"])
                       ["annualized_sharpe"].idxmax()]
    best.to_csv(out / "v5_p10_best_per_combo.csv", index=False)

    # --- Overall F1 master ---
    o51 = pd.read_csv(PROJECT_ROOT / "reports" / "Phase5.1_arch_ablation" / "v5_p5_arch_overall.csv")
    o51["variant"] = "baseline"; o51["phase"] = "P5.1"
    o7  = pd.read_csv(PROJECT_ROOT / "reports" / "Phase7_regime_aware" / "v5_p7_overall.csv")
    o7["variant"] = "regime_aware"; o7["phase"] = "P7"
    o9  = pd.read_csv(PROJECT_ROOT / "reports" / "Phase9_mlp_oversample" / "v5_p9_overall.csv")
    o9["phase"] = "P9"
    keep_o = ["phase", "variant", "asset", "arch", "model",
              "n_oof", "accuracy", "f1_macro", "f1_sell", "f1_hold", "f1_buy"]
    overall = pd.concat([o51.reindex(columns=keep_o),
                         o7.reindex(columns=keep_o),
                         o9.reindex(columns=keep_o)], ignore_index=True)
    overall.to_csv(out / "v5_p10_master_overall.csv", index=False)
    print(f"master overall: {len(overall)} rows")

    # --- Plot: calibration ECE before vs after ---
    cal = pd.read_csv(PROJECT_ROOT / "reports" / "Phase8_calibration" / "v5_p8_calibration_metrics.csv")
    g = cal.groupby(["asset", "arch", "model", "variant"])["ece"].mean().unstack("variant")
    g = g.reset_index()
    g["label"] = g["asset"].str.upper() + "/" + g["arch"] + "/" + g["model"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(g))
    w = 0.4
    ax.bar(x - w/2, g["raw"], width=w, label="Raw", color="#888", alpha=0.85)
    ax.bar(x + w/2, g["cal"], width=w, label="Isotonic", color="#0a9396", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(g["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("ECE (mean across classes, lower = better)")
    ax.set_title("Walk-forward isotonic calibration: ECE before vs. after")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "fig_calibration_ece.png", dpi=140)
    plt.close()
    print(f"plot: fig_calibration_ece.png")

    # --- Plot: MLP oversample F1 (baseline vs oversampled) ---
    base_mlp = o51[o51["model"] == "mlp"][["asset", "arch", "f1_macro", "f1_hold"]].copy()
    base_mlp["variant"] = "baseline"
    over = o9[["asset", "arch", "f1_macro", "f1_hold"]].copy()
    over["variant"] = "oversampled"
    cmp = pd.concat([base_mlp, over], ignore_index=True)
    cmp = cmp.merge(over[["asset", "arch"]], on=["asset", "arch"])  # only over targets
    cmp = cmp.drop_duplicates()
    cmp["label"] = cmp["asset"].str.upper() + "/" + cmp["arch"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(axes[0], "f1_macro", "F1-macro"),
                              (axes[1], "f1_hold",  "Hold-class F1")]:
        piv = cmp.pivot_table(index="label", columns="variant", values=metric).reset_index()
        x = np.arange(len(piv))
        w = 0.35
        ax.bar(x - w/2, piv["baseline"],    width=w, label="Baseline (no resample)", color="#888")
        ax.bar(x + w/2, piv["oversampled"], width=w, label="RandomOverSampler",       color="#bb3e03")
        ax.set_xticks(x); ax.set_xticklabels(piv["label"], rotation=20, ha="right")
        ax.set_title(title); ax.set_ylabel(metric); ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    plt.suptitle("MLP class imbalance fix via oversampling", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "fig_oversample_f1.png", dpi=140, bbox_inches="tight")
    plt.close()
    print(f"plot: fig_oversample_f1.png")

    # --- Headline table: best per asset across ALL variants ---
    bt = master.dropna(subset=["annualized_sharpe"]).copy()
    headline = bt.loc[bt.groupby(["asset"])["annualized_sharpe"].idxmax()]
    print("\n=== Headline: best Sharpe per asset across ALL phases ===")
    print(headline[["phase", "variant", "asset", "arch", "model", "rule",
                    "annualized_sharpe", "total_return", "max_drawdown"]
                   ].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # Per-variant Sharpe lift table (vs Phase 5.1 best-of-class for that asset/arch/model)
    base_idx = ["asset", "arch", "model", "rule"]
    base = p51.set_index(base_idx)["annualized_sharpe"]
    print("\n=== Sharpe lift summary (variant best per asset) ===")
    rows = []
    for variant_label, src in [("regime_aware (P7)", p7),
                                ("calibration_raw (P8)", p8[p8["variant"]=="p8_raw"]),
                                ("calibration_cal (P8)", p8[p8["variant"]=="p8_cal"]),
                                ("mlp_oversample (P9)", p9)]:
        s = src.set_index(base_idx)["annualized_sharpe"]
        joined = s.to_frame("variant").join(base.to_frame("base"), how="left")
        joined["lift"] = joined["variant"] - joined["base"]
        per_asset = joined.reset_index().groupby("asset")["lift"].mean()
        rows.append({"variant": variant_label, **per_asset.to_dict()})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
