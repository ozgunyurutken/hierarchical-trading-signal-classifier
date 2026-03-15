"""
Statistical significance tests.
McNemar's test for comparing classifier predictions.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """
    Perform McNemar's test to compare two classifiers.

    Tests whether two classifiers have the same error rate.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred_a : array-like
        Predictions from classifier A (e.g., flat baseline).
    y_pred_b : array-like
        Predictions from classifier B (e.g., 3-stage pipeline).

    Returns
    -------
    dict with 'statistic', 'p_value', 'significant' keys.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Contingency table
    # b01 = A correct, B wrong
    # b10 = A wrong, B correct
    b01 = np.sum(correct_a & ~correct_b)  # A right, B wrong
    b10 = np.sum(~correct_a & correct_b)  # A wrong, B right

    # McNemar statistic with continuity correction
    if b01 + b10 == 0:
        logger.info("McNemar: both classifiers have identical predictions")
        return {"statistic": 0, "p_value": 1.0, "significant": False}

    statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - chi2.cdf(statistic, df=1)

    significant = p_value < 0.05

    logger.info(
        f"McNemar's test: statistic={statistic:.4f}, p={p_value:.6f}, "
        f"significant={significant}, b01={b01}, b10={b10}"
    )

    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant": significant,
        "b01_a_right_b_wrong": int(b01),
        "b10_a_wrong_b_right": int(b10),
    }


def plot_decision_boundary_pca(
    X: pd.DataFrame,
    y: np.ndarray,
    model,
    title: str = "Decision Boundary (PCA 2D)",
    resolution: int = 200,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize decision boundaries using PCA 2D projection.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : array-like
        Labels.
    model : trained classifier
        Must have predict() method.
    title : str
        Plot title.
    resolution : int
        Grid resolution.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
    """
    # PCA projection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.fillna(X.median()))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_
    logger.info(f"PCA: {explained[0]:.2%} + {explained[1]:.2%} = {sum(explained):.2%} variance explained")

    # Create mesh grid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    # Inverse-transform grid points for prediction
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    grid_original = pca.inverse_transform(grid_pca)
    grid_unscaled = scaler.inverse_transform(grid_original)
    grid_df = pd.DataFrame(grid_unscaled, columns=X.columns)

    # Predict on grid
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    Z = model.predict(grid_df)
    Z_encoded = le.transform(Z)
    Z_grid = Z_encoded.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    cmap_bg = ListedColormap(colors[:len(le.classes_)])
    cmap_pts = ListedColormap(colors[:len(le.classes_)])

    ax.contourf(xx, yy, Z_grid, alpha=0.3, cmap=cmap_bg)

    for i, cls in enumerate(le.classes_):
        mask = y_encoded == i
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[i], label=cls, alpha=0.6, s=20, edgecolor="k", linewidth=0.3,
        )

    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
