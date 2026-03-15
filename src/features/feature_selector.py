"""
Feature selection pipeline.
Applies correlation filtering, mutual information ranking, and SHAP-based selection.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


def remove_highly_correlated(
    df: pd.DataFrame,
    target: pd.Series | None = None,
    threshold: float = 0.95,
) -> list[str]:
    """
    Remove features with |correlation| > threshold.
    When two features are highly correlated, keep the one with higher target correlation.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    target : pd.Series, optional
        Target variable for tie-breaking (encoded as integers).
    threshold : float
        Correlation threshold.

    Returns
    -------
    list[str]
        List of features to keep.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        high_corr = upper.index[upper[col] > threshold].tolist()
        if high_corr:
            if target is not None:
                # Keep the feature with higher target correlation
                target_corr_col = abs(df[col].corr(target))
                for hc in high_corr:
                    target_corr_hc = abs(df[hc].corr(target))
                    if target_corr_hc > target_corr_col:
                        to_drop.add(col)
                    else:
                        to_drop.add(hc)
            else:
                to_drop.update(high_corr)

    kept = [c for c in df.columns if c not in to_drop]
    logger.info(f"Correlation filter: {len(df.columns)} -> {len(kept)} features (dropped {len(to_drop)})")
    return kept


def rank_by_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    n_neighbors: int = 5,
) -> pd.Series:
    """
    Rank features by mutual information with the target.

    Returns
    -------
    pd.Series
        MI scores indexed by feature name, sorted descending.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Handle NaN by filling with median
    X_clean = X.fillna(X.median())

    mi_scores = mutual_info_classif(X_clean, y_encoded, n_neighbors=n_neighbors, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns, name="MI_score").sort_values(ascending=False)

    logger.info(f"MI ranking: top 5 = {mi_series.head().to_dict()}")
    return mi_series


def rank_by_shap(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.Series:
    """
    Rank features by SHAP importance using a preliminary XGBoost model.

    Returns
    -------
    pd.Series
        Mean |SHAP| values indexed by feature name, sorted descending.
    """
    import xgboost as xgb
    import shap

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_clean = X.fillna(X.median())

    # Train a quick XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss",
        verbosity=0,
    )
    model.fit(X_clean, y_encoded)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_clean)

    # For multiclass: average absolute SHAP across classes
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_series = pd.Series(
        mean_abs_shap, index=X.columns, name="SHAP_importance"
    ).sort_values(ascending=False)

    logger.info(f"SHAP ranking: top 5 = {shap_series.head().to_dict()}")
    return shap_series


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 15,
    correlation_threshold: float = 0.95,
) -> list[str]:
    """
    Full feature selection pipeline.

    1. Remove highly correlated features
    2. Rank by mutual information
    3. Rank by SHAP importance
    4. Union of top features from both methods

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    top_k : int
        Number of top features to keep per method.
    correlation_threshold : float
        Correlation threshold for filtering.

    Returns
    -------
    list[str]
        Selected feature names.
    """
    le = LabelEncoder()
    y_numeric = pd.Series(le.fit_transform(y), index=y.index, name="target")

    # Step 1: Correlation filtering
    kept_features = remove_highly_correlated(X, target=y_numeric, threshold=correlation_threshold)
    X_filtered = X[kept_features]

    # Step 2: Mutual Information
    mi_ranking = rank_by_mutual_information(X_filtered, y)
    mi_top = set(mi_ranking.head(top_k).index)

    # Step 3: SHAP
    shap_ranking = rank_by_shap(X_filtered, y)
    shap_top = set(shap_ranking.head(top_k).index)

    # Step 4: Union
    selected = sorted(mi_top | shap_top)

    logger.info(
        f"Feature selection complete: {len(X.columns)} -> {len(selected)} features "
        f"(MI top-{top_k}: {len(mi_top)}, SHAP top-{top_k}: {len(shap_top)}, union: {len(selected)})"
    )

    return selected
