"""
Evaluation metrics module.
Computes all classification metrics for model comparison.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    classes: list[str] | None = None,
) -> dict:
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities (n_samples, n_classes). Required for ROC-AUC and PR-AUC.
    classes : list, optional
        Class names.

    Returns
    -------
    dict with all metrics.
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Per-class metrics
    metrics["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class breakdown
    metrics["precision_per_class"] = precision_score(y_true, y_pred, average=None, zero_division=0)
    metrics["recall_per_class"] = recall_score(y_true, y_pred, average=None, zero_division=0)
    metrics["f1_per_class"] = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    # Classification report
    metrics["classification_report"] = classification_report(y_true, y_pred, zero_division=0)

    # Probability-based metrics
    if y_proba is not None and classes is not None:
        try:
            y_bin = label_binarize(y_true, classes=classes)
            if y_bin.shape[1] == 1:  # Binary case
                y_bin = np.hstack([1 - y_bin, y_bin])

            metrics["roc_auc_ovr"] = roc_auc_score(
                y_bin, y_proba, multi_class="ovr", average="macro"
            )
            metrics["pr_auc_macro"] = average_precision_score(
                y_bin, y_proba, average="macro"
            )
        except Exception as e:
            logger.warning(f"Could not compute AUC metrics: {e}")

    logger.info(
        f"Metrics: acc={metrics['accuracy']:.4f}, "
        f"bal_acc={metrics['balanced_accuracy']:.4f}, "
        f"f1_macro={metrics['f1_macro']:.4f}, "
        f"mcc={metrics['mcc']:.4f}"
    )

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str] | None = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (8, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot confusion matrix as heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes or sorted(set(y_true)),
        yticklabels=classes or sorted(set(y_true)),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def create_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Create a comparison table from multiple experiment results.

    Parameters
    ----------
    results : dict
        Mapping of experiment name to metrics dict.

    Returns
    -------
    pd.DataFrame
        Comparison table with experiments as rows.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Experiment": name,
            "Accuracy": metrics.get("accuracy", np.nan),
            "Balanced Accuracy": metrics.get("balanced_accuracy", np.nan),
            "F1 Macro": metrics.get("f1_macro", np.nan),
            "MCC": metrics.get("mcc", np.nan),
            "ROC-AUC": metrics.get("roc_auc_ovr", np.nan),
            "PR-AUC": metrics.get("pr_auc_macro", np.nan),
        })

    df = pd.DataFrame(rows).set_index("Experiment")
    return df.round(4)
