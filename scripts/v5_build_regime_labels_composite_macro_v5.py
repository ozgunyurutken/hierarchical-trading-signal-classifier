"""V5 Phase 2.12 — Composite Macro FSM v5 (noise reduction).

Phase 2.11 üzerine 3 ek (kullanıcı feedback: 2.11 noisy):
  - neutral_min_dwell = 10  (yeni; threshold-based çıkışlara dwell şartı,
                              velocity entries bypass eder)
  - bull_min_dwell    30 → 40
  - bear_reentry_min_neutral_dwell  10 → 15

Krizler ve çıkışlar etkilenmez (velocity entries dwell'i bypass eder).

Outputs:
  data/processed/{btc,eth,macro_pretrain}_regime_labels_composite_macro_v5_v5.csv
  reports/Phase2/v5_phase2.12_noise_reduction_timeline_4panel.png
  reports/Phase2/v5_p2.12_diagnostics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.labels.v5_regime_labels import (
    CompositeVIXRegimeClassifier, BULL_BEAR_LABELS,
)


def main():
    print("V5 Phase 2.12 — Composite Macro FSM v5 (noise reduction)")
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

    print("\n[1] Fit Composite Macro FSM v5")
    model = CompositeVIXRegimeClassifier(
        bear_entry_threshold=1.0,
        bear_exit_threshold=0.3,
        bull_entry_threshold=-0.5,
        bull_exit_threshold=0.0,
        bear_min_dwell=20,
        bull_min_dwell=40,                              # 30 → 40 (Plan A)
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
        enable_bear_velocity_entry=True,
        bear_velocity_entry_window=5,
        bear_velocity_entry_threshold=0.6,
        bear_velocity_entry_sp500_max=-0.015,
        bear_reentry_min_neutral_dwell=15,              # 10 → 15 (Plan A)
        neutral_min_dwell=10,                           # YENİ (Plan A)
    ).fit(pretrain_clean)

    print(f"  bear_min_dwell:                    {model.bear_min_dwell}")
    print(f"  bull_min_dwell:                    {model.bull_min_dwell} (30 → 40)")
    print(f"  neutral_min_dwell:                 {model.neutral_min_dwell} (YENİ)")
    print(f"  bear_reentry_min_neutral_dwell:    {model.bear_reentry_min_neutral_dwell} (10 → 15)")
    print(f"  velocity entries bypass dwell:     YES (rapid crisis response)")

    print(f"\n[2] Inference")
    pre_r = model.predict(pretrain_clean)
    btc_r = pre_r.reindex(btc.index, method="ffill")
    eth_r = pre_r.reindex(eth.index, method="ffill")
    pre_r.to_csv(proc / "macro_pretrain_regime_labels_composite_macro_v5_v5.csv")
    btc_r.to_csv(proc / "btc_regime_labels_composite_macro_v5_v5.csv")
    eth_r.to_csv(proc / "eth_regime_labels_composite_macro_v5_v5.csv")

    n_velocity = int(pre_r["velocity_override"].sum())
    n_yc = int(pre_r["yield_curve_override"].sum())
    n_ms = int(pre_r["macro_stress_override"].sum())
    n_bv = int(pre_r["bull_velocity_entry"].sum())
    n_bev = int(pre_r["bear_velocity_entry"].sum())
    print(f"  Bear→Neutral velocity:  {n_velocity}")
    print(f"  Bull velocity entry:    {n_bv}")
    print(f"  Bear velocity entry:    {n_bev}")
    print(f"  YC override:            {n_yc}")
    print(f"  Macro stress override:  {n_ms}")

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
    print(f"\n  Total transitions:      {len(runs) - 1}")
    print(f"  # Bear runs:            {len(bear_runs)}, mean: "
          f"{np.mean([r['duration_days'] for r in bear_runs]):.0f}d")
    print(f"  # Bull runs:            {len(bull_runs)}, mean: "
          f"{np.mean([r['duration_days'] for r in bull_runs]):.0f}d")
    print(f"  # Neutral runs:         {len(neutral_runs)}, mean: "
          f"{np.mean([r['duration_days'] for r in neutral_runs]):.0f}d")

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
        "phase": "V5 Phase 2.12 — Composite Macro FSM v5 (noise reduction)",
        "user_feedback_addressed": [
            "2.11 sık rejim değişimi azaltıldı (Plan A):",
            "  neutral_min_dwell=10 (yeni)",
            "  bull_min_dwell 30→40",
            "  bear_reentry 10→15",
            "Velocity entries dwell'i bypass eder (krizler korunur):",
            "  Bull velocity entry (V-shape recovery)",
            "  Bear velocity entry (Liberation Day, SVB, yen carry)",
        ],
        "transitions_total": len(runs) - 1,
        "bear_runs_count": len(bear_runs),
        "bull_runs_count": len(bull_runs),
        "neutral_runs_count": len(neutral_runs),
        "n_bull_velocity_entry": n_bv,
        "n_bear_velocity_entry": n_bev,
        "bear_velocity_dates": [str(d.date()) for d in pre_r.index[pre_r["bear_velocity_entry"]]],
        "bull_velocity_dates": [str(d.date()) for d in pre_r.index[pre_r["bull_velocity_entry"]]],
        "n_yield_curve_override": n_yc,
        "n_macro_stress_override": n_ms,
        "distribution_pct": distribution,
    }
    diag_path = reports / "Phase2" / "v5_p2.12_diagnostics.json"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\n  saved: {diag_path.relative_to(PROJECT_ROOT)}")
    print("\nV5 Phase 2.12 complete.")


if __name__ == "__main__":
    main()
