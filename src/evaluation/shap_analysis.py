"""
SHAP analysis module.
Generates SHAP summary and dependence plots for model explainability.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> tuple[np.ndarray, shap.Explainer]:
    """
    Compute SHAP values for a trained model.

    Parameters
    ----------
    model : BaseClassifier
        Trained classifier (must have .model attribute).
    X : pd.DataFrame
        Feature matrix.
    max_samples : int
        Maximum samples for SHAP computation (for speed).

    Returns
    -------
    tuple of (shap_values, explainer)
    """
    X_clean = X.fillna(X.median())

    if len(X_clean) > max_samples:
        X_sample = X_clean.sample(max_samples, random_state=42)
    else:
        X_sample = X_clean

    # Use TreeExplainer for tree-based models
    inner_model = model.model if hasattr(model, "model") else model

    try:
        explainer = shap.TreeExplainer(inner_model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        logger.info("TreeExplainer failed, falling back to KernelExplainer")
        background = shap.sample(X_clean, min(100, len(X_clean)))

        def predict_fn(x):
            return model.predict_proba(pd.DataFrame(x, columns=X.columns))

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_sample)

    logger.info(f"SHAP values computed for {len(X_sample)} samples, {X.shape[1]} features")
    return shap_values, explainer


def plot_shap_summary(
    shap_values,
    X: pd.DataFrame,
    class_names: list[str] | None = None,
    title: str = "SHAP Summary",
    max_display: int = 15,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot SHAP summary bar plot."""
    X_clean = X.fillna(X.median())

    if isinstance(shap_values, list):
        # Multiclass: average absolute SHAP
        fig, axes = plt.subplots(1, len(shap_values), figsize=(6 * len(shap_values), 8))
        if len(shap_values) == 1:
            axes = [axes]
        for i, (sv, ax) in enumerate(zip(shap_values, axes)):
            plt.sca(ax)
            class_name = class_names[i] if class_names else f"Class {i}"
            shap.summary_plot(
                sv, X_clean.iloc[:len(sv)],
                plot_type="bar",
                max_display=max_display,
                show=False,
            )
            ax.set_title(f"{title} - {class_name}")
    else:
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_clean.iloc[:len(shap_values)],
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title(title)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_dependence(
    shap_values,
    X: pd.DataFrame,
    feature_names: list[str],
    class_idx: int = 0,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot SHAP dependence plots for specified features."""
    X_clean = X.fillna(X.median())
    sv = shap_values[class_idx] if isinstance(shap_values, list) else shap_values

    n_features = len(feature_names)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 4))
    if n_features == 1:
        axes = [axes]

    for ax, feat in zip(axes, feature_names):
        if feat in X_clean.columns:
            feat_idx = list(X_clean.columns).index(feat)
            plt.sca(ax)
            shap.dependence_plot(
                feat_idx, sv, X_clean.iloc[:len(sv)],
                show=False, ax=ax,
            )
            ax.set_title(f"SHAP Dependence: {feat}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
