"""V5 Phase 3 — Segment-level evaluation metrics for Stage 1 OOF predictions.

Frame-level F1/accuracy treats every day as an independent classification.
For trend regime detection, segment-level metrics better capture whether the
model identifies macro structure even with daily noise.

Metrics implemented:
  - frame_f1_macro             : standard F1 macro (per-day) — baseline
  - smoothed_f1_macro          : F1 macro after rolling-mode smoothing
  - per_class_iou              : (true_c & pred_c) / (true_c | pred_c) for each class
  - mean_iou                   : mean IoU across 3 classes
  - onset_detection_f1         : transition-time agreement (±tolerance days)
  - majority_vote_consistency  : within each true segment, share of dominant pred class
  - mean_boundary_error_days   : avg gap between matched true/pred onsets

Conceptual frame: distinguishing 'frame-level' (each timestep independent) from
'segment-level' (regime as a whole) evaluation. Common in speech recognition,
activity recognition, and climate regime detection literature.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score

CLASS_NAMES = ["downtrend", "range", "uptrend"]
LABEL_TO_IDX = {"downtrend": 0, "range": 1, "uptrend": 2}


# ---------- Smoothing ----------

def rolling_mode(pred: pd.Series, window: int) -> pd.Series:
    """Replace each prediction with rolling-window most frequent label.

    Centered window. Edge timesteps use available subset.
    """
    arr = pred.to_numpy()
    n = len(arr)
    smoothed = np.empty(n, dtype=object)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = arr[lo:hi]
        # Counter is overkill — use np unique
        vals, counts = np.unique(chunk, return_counts=True)
        smoothed[i] = vals[np.argmax(counts)]
    return pd.Series(smoothed, index=pred.index, name=pred.name)


# ---------- Segment extraction ----------

def extract_segments(labels: pd.Series) -> list[tuple[int, int, str]]:
    """Return list of (start_idx, end_idx_exclusive, label) for contiguous runs."""
    arr = labels.to_numpy()
    if len(arr) == 0:
        return []
    segs = []
    cur = arr[0]
    start = 0
    for i in range(1, len(arr)):
        if arr[i] != cur:
            segs.append((start, i, cur))
            cur = arr[i]
            start = i
    segs.append((start, len(arr), cur))
    return segs


def transition_indices(labels: pd.Series) -> list[int]:
    """Indices where label changes (start indices of new segments, excluding 0)."""
    arr = labels.to_numpy()
    return [i for i in range(1, len(arr)) if arr[i] != arr[i - 1]]


# ---------- Metrics ----------

@dataclass
class SegmentMetrics:
    asset: str
    model: str
    variant: str             # baseline / 3-fold tuned / 5-fold tuned / smoothed
    smoothing_window: int    # 0 = raw
    n: int

    frame_f1_macro: float
    frame_accuracy: float

    iou_downtrend: float
    iou_range: float
    iou_uptrend: float
    mean_iou: float

    onset_f1: float
    onset_precision: float
    onset_recall: float
    n_true_onsets: int
    n_pred_onsets: int
    mean_boundary_error_days: float

    majority_vote_consistency: float

    def to_dict(self) -> dict:
        return asdict(self)


def compute_segment_metrics(truth: pd.Series, pred: pd.Series,
                            *, asset: str, model: str, variant: str,
                            smoothing_window: int = 0,
                            onset_tolerance_days: int = 5) -> SegmentMetrics:
    """All segment metrics in a single call."""
    if len(truth) != len(pred):
        raise ValueError(f"Length mismatch: truth={len(truth)} pred={len(pred)}")

    y_true = truth.map(LABEL_TO_IDX).to_numpy()
    y_pred = pred.map(LABEL_TO_IDX).to_numpy()

    frame_f1 = f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)
    frame_acc = (y_true == y_pred).mean()

    iou_per = jaccard_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    mean_iou = float(np.mean(iou_per))

    onset_p, onset_r, onset_f1, n_true_t, n_pred_t, mean_be = _onset_metrics(
        truth, pred, tolerance=onset_tolerance_days
    )
    mvc = _majority_vote_consistency(truth, pred)

    return SegmentMetrics(
        asset=asset, model=model, variant=variant,
        smoothing_window=smoothing_window,
        n=len(truth),
        frame_f1_macro=float(frame_f1),
        frame_accuracy=float(frame_acc),
        iou_downtrend=float(iou_per[0]),
        iou_range=float(iou_per[1]),
        iou_uptrend=float(iou_per[2]),
        mean_iou=mean_iou,
        onset_f1=float(onset_f1),
        onset_precision=float(onset_p),
        onset_recall=float(onset_r),
        n_true_onsets=int(n_true_t),
        n_pred_onsets=int(n_pred_t),
        mean_boundary_error_days=float(mean_be),
        majority_vote_consistency=float(mvc),
    )


def _onset_metrics(truth: pd.Series, pred: pd.Series, tolerance: int):
    """Greedy 1-to-1 match of true onsets with pred onsets within ±tolerance.

    Returns (precision, recall, f1, n_true, n_pred, mean_boundary_error_days).
    """
    true_t = transition_indices(truth)
    pred_t = transition_indices(pred)
    n_true = len(true_t)
    n_pred = len(pred_t)

    if n_true == 0 and n_pred == 0:
        return 1.0, 1.0, 1.0, 0, 0, 0.0
    if n_true == 0:
        return 0.0, 1.0, 0.0, 0, n_pred, np.nan
    if n_pred == 0:
        return 1.0, 0.0, 0.0, n_true, 0, np.nan

    matched_pred = set()
    errors = []
    matched_true = 0

    for t in true_t:
        # Find nearest unmatched pred within tolerance
        best, best_d = None, None
        for j, p in enumerate(pred_t):
            if j in matched_pred:
                continue
            d = abs(p - t)
            if d <= tolerance and (best_d is None or d < best_d):
                best, best_d = j, d
        if best is not None:
            matched_pred.add(best)
            matched_true += 1
            errors.append(best_d)

    tp = matched_true
    fp = n_pred - len(matched_pred)
    fn = n_true - matched_true
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    mean_be = float(np.mean(errors)) if errors else np.nan
    return precision, recall, f1, n_true, n_pred, mean_be


def _majority_vote_consistency(truth: pd.Series, pred: pd.Series) -> float:
    """Within each true segment, share of the dominant predicted class.

    Length-weighted average across true segments.
    """
    segs = extract_segments(truth)
    if not segs:
        return float("nan")
    total_consistency = 0.0
    total_len = 0
    pred_arr = pred.to_numpy()
    for start, end, _label in segs:
        chunk = pred_arr[start:end]
        if len(chunk) == 0:
            continue
        vals, counts = np.unique(chunk, return_counts=True)
        dom_share = counts.max() / counts.sum()
        total_consistency += dom_share * len(chunk)
        total_len += len(chunk)
    return total_consistency / total_len if total_len else float("nan")
