"""
Stage 2 — Macro regime label generation.
Labels: Risk-On, Risk-Off, Neutral
Methods: K-Means, GMM, HMM
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src.utils.config import cfg
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def fit_kmeans(
    X: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, KMeans, StandardScaler]:
    """
    Fit K-Means clustering on macro features.

    Returns
    -------
    tuple of (labels_array, fitted_kmeans, fitted_scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    sil_score = silhouette_score(X_scaled, labels)
    logger.info(f"K-Means (k={n_clusters}): silhouette={sil_score:.4f}")

    return labels, kmeans, scaler


def fit_gmm(
    X: pd.DataFrame,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, GaussianMixture, StandardScaler]:
    """
    Fit Gaussian Mixture Model on macro features.

    Returns
    -------
    tuple of (hard_labels, soft_probabilities, fitted_gmm, fitted_scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        covariance_type="full",
    )
    hard_labels = gmm.fit_predict(X_scaled)
    soft_probs = gmm.predict_proba(X_scaled)

    sil_score = silhouette_score(X_scaled, hard_labels)
    logger.info(f"GMM (k={n_components}): silhouette={sil_score:.4f}, BIC={gmm.bic(X_scaled):.2f}")

    return hard_labels, soft_probs, gmm, scaler


def fit_hmm(
    X: pd.DataFrame,
    n_states: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, object, StandardScaler]:
    """
    Fit Hidden Markov Model on macro features.

    Returns
    -------
    tuple of (hard_labels, state_probabilities, fitted_hmm, fitted_scaler)
    """
    from hmmlearn.hmm import GaussianHMM

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    hmm = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=random_state,
    )
    hmm.fit(X_scaled)
    hard_labels = hmm.predict(X_scaled)
    state_probs = hmm.predict_proba(X_scaled)

    logger.info(f"HMM (states={n_states}): converged={hmm.monitor_.converged}, score={hmm.score(X_scaled):.2f}")

    return hard_labels, state_probs, hmm, scaler


def assign_semantic_labels(
    cluster_labels: np.ndarray,
    macro_df: pd.DataFrame,
    vix_col: str | None = None,
    sp500_col: str | None = None,
) -> pd.Series:
    """
    Assign semantic regime names (Risk-On, Risk-Off, Neutral) based on cluster centroids.

    Logic:
        - Risk-On: Lowest mean VIX + highest mean S&P return
        - Risk-Off: Highest mean VIX + most negative S&P return
        - Neutral: Remainder
    """
    # Find VIX and S&P columns
    if vix_col is None:
        vix_candidates = [c for c in macro_df.columns if "vix" in c.lower() or "VIX" in c]
        vix_col = vix_candidates[0] if vix_candidates else None

    if sp500_col is None:
        sp_candidates = [c for c in macro_df.columns if "s&p" in c.lower() or "gspc" in c.lower() or "S&P" in c]
        sp500_col = sp_candidates[0] if sp_candidates else None

    labels_series = pd.Series(cluster_labels, index=macro_df.index)
    unique_clusters = sorted(labels_series.unique())

    if vix_col and vix_col in macro_df.columns:
        # Rank clusters by mean VIX
        cluster_vix_means = {}
        for c in unique_clusters:
            mask = labels_series == c
            cluster_vix_means[c] = macro_df.loc[mask, vix_col].mean()

        sorted_by_vix = sorted(cluster_vix_means, key=cluster_vix_means.get)

        mapping = {
            sorted_by_vix[0]: "Risk-On",    # Lowest VIX
            sorted_by_vix[-1]: "Risk-Off",   # Highest VIX
        }
        for c in unique_clusters:
            if c not in mapping:
                mapping[c] = "Neutral"

        logger.info(f"Regime mapping (by VIX): {mapping}, VIX means: {cluster_vix_means}")
    else:
        # Fallback: just number them
        mapping = {c: ["Risk-On", "Neutral", "Risk-Off"][i % 3] for i, c in enumerate(unique_clusters)}
        logger.warning(f"No VIX column found, using fallback mapping: {mapping}")

    semantic = labels_series.map(mapping)
    semantic.name = "regime_label"
    return semantic


def evaluate_clustering_k(
    X: pd.DataFrame,
    k_range: list[int] | None = None,
) -> pd.DataFrame:
    """
    Evaluate K-Means for multiple k values using elbow and silhouette.

    Returns
    -------
    pd.DataFrame
        Columns: k, inertia, silhouette_score
    """
    config = cfg()
    k_range = k_range or config["labels"]["regime"]["cluster_range"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if k > 1 else 0
        results.append({"k": k, "inertia": km.inertia_, "silhouette_score": sil})
        logger.info(f"  k={k}: inertia={km.inertia_:.2f}, silhouette={sil:.4f}")

    return pd.DataFrame(results)


def compare_methods(
    X: pd.DataFrame,
    macro_df: pd.DataFrame,
    n_clusters: int = 3,
) -> dict[str, pd.Series]:
    """
    Compare K-Means, GMM, and HMM regime labels.

    Returns
    -------
    dict with keys 'kmeans', 'gmm', 'hmm' mapping to label Series.
    """
    results = {}

    # K-Means
    km_labels, _, _ = fit_kmeans(X, n_clusters)
    results["kmeans"] = assign_semantic_labels(km_labels, macro_df)

    # GMM
    gmm_labels, _, _, _ = fit_gmm(X, n_clusters)
    results["gmm"] = assign_semantic_labels(gmm_labels, macro_df)

    # HMM
    try:
        hmm_labels, _, _, _ = fit_hmm(X, n_clusters)
        results["hmm"] = assign_semantic_labels(hmm_labels, macro_df)
    except Exception as e:
        logger.error(f"HMM fitting failed: {e}")

    # Compare with ARI
    if "kmeans" in results and "gmm" in results:
        ari = adjusted_rand_score(results["kmeans"], results["gmm"])
        logger.info(f"ARI (K-Means vs GMM): {ari:.4f}")

    if "kmeans" in results and "hmm" in results:
        ari = adjusted_rand_score(results["kmeans"], results["hmm"])
        logger.info(f"ARI (K-Means vs HMM): {ari:.4f}")

    # Report label stability (number of regime flips)
    for method, labels in results.items():
        flips = (labels != labels.shift()).sum()
        logger.info(f"  {method}: {flips} regime transitions")

    return results
