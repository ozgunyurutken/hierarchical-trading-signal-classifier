"""V5 Phase 3 — Segment-level evaluation runner for Stage 1 OOF.

Computes segment-level metrics for all (asset, model, variant) combinations:
  variants:
    - baseline                 (data/processed/{asset}_stage1_oof_{model}_v5.csv)
    - 3-fold tuned             (..._v5_tuned_3fold.csv)
    - 5-fold tuned             (..._v5_tuned.csv)
    - 5-fold tuned + smoothed  (5d / 10d / 20d rolling-mode applied to predictions)

Output:
  reports/Phase3.5_after_tune/v5_p3_stage1_segment_metrics.csv

Run:
  python scripts/v5_evaluate_stage1_segments.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.evaluation.v5_segment_metrics import (
    compute_segment_metrics, rolling_mode,
)


ASSETS = ["btc", "eth"]
MODELS = ["xgboost", "lightgbm", "random_forest", "mlp"]

VARIANTS = [
    ("baseline",     "_v5"),
    ("3fold tuned",  "_v5_tuned_3fold"),
    ("5fold tuned",  "_v5_tuned"),
]
SMOOTH_WINDOWS = [5, 10, 20]


def main() -> int:
    proc = PROJECT_ROOT / "data" / "processed"
    out = PROJECT_ROOT / "reports" / "Phase3.5_after_tune"
    out.mkdir(parents=True, exist_ok=True)

    records = []

    for asset in ASSETS:
        for model in MODELS:
            for vname, suffix in VARIANTS:
                p = proc / f"{asset}_stage1_oof_{model}{suffix}.csv"
                if not p.exists():
                    print(f"  ! skip {p.name} (missing)")
                    continue
                oof = pd.read_csv(p, index_col=0, parse_dates=True)
                truth = oof["true_label"]
                pred = oof["pred_label"]

                # Raw (no smoothing)
                m = compute_segment_metrics(
                    truth, pred,
                    asset=asset, model=model, variant=vname,
                    smoothing_window=0,
                )
                records.append(m.to_dict())

                # Smoothed variants — only for 5fold tuned (final pipeline)
                if vname == "5fold tuned":
                    for w in SMOOTH_WINDOWS:
                        pred_s = rolling_mode(pred, window=w)
                        m_s = compute_segment_metrics(
                            truth, pred_s,
                            asset=asset, model=model,
                            variant=f"5fold + smooth{w}d",
                            smoothing_window=w,
                        )
                        records.append(m_s.to_dict())
            # progress per asset+model
            print(f"  done: {asset}/{model}")

    df = pd.DataFrame(records)
    csv_path = out / "v5_p3_stage1_segment_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path.relative_to(PROJECT_ROOT)}\n")

    # Summary tables
    print("=== Mean across all 8 (asset, model) combinations, per variant ===")
    summary = (df.groupby("variant")[
        ["frame_f1_macro", "frame_accuracy", "mean_iou",
         "onset_f1", "majority_vote_consistency", "mean_boundary_error_days"]
    ].mean().round(4))
    print(summary.to_string())
    print()

    print("=== Detail per (asset, model) for 5fold-tuned variants ===")
    sub = df[df["variant"].str.startswith("5fold")][
        ["asset", "model", "variant",
         "frame_f1_macro", "mean_iou", "onset_f1",
         "majority_vote_consistency"]
    ].round(3)
    print(sub.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
