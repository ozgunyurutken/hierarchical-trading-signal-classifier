"""
V5 Faz 2.2 — Stage 2 Macro Regime Labels via K-Means + Semantic Relabeling.

Pipeline:
  1. Fit K-Means k=3 on pre-train derived macro features (2000-2025, ~6500g)
     - StandardScaler fit on pre-train only
     - Cluster validation: Elbow + Silhouette + Gap statistic + Calinski-Harabasz
  2. Semantic relabeling by centroid inspection:
     - Lowest VIX_z + Highest SP500_5d → Risk-On
     - Highest VIX_z + Most negative SP500_5d → Risk-Off
     - Remainder → Neutral
  3. Inference on crypto-aligned macro features (BTC 2014+, ETH 2017+)

Literature:
  - Rousseeuw 1987 [N7]: Silhouette
  - Tibshirani et al. 2001 [N8]: Gap statistic
  - Caliński & Harabasz 1974 [N9]: variance ratio
  - Hatzius FCI 2010 [N64]: long-term macro factor analysis
  - Liu & Härdle 2025 [N12]: BTC K-Means + HMM
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


# Stage 2 features (config.yaml stage2_features)
# 2026-05-09 Phase 1 fix: UNRATE_change_180d ÇIKARILDI.
# Sebep: Risk-Off cluster centroid'inde +1.69 dominant, sadece COVID 2020 yakalıyor;
# 2008 GFC, 2002 dot-com, 2022 hike Neutral'a düşüyordu.
# 8 feature kaldı — VIX/SP500/Gold/FFR/CPI/Yield Curve/M2 + DXY z-score.
STAGE2_FEATURES = [
    "VIX_zscore_long",
    "SP500_log_return_5d",
    "DXY_zscore_long",
    "Gold_log_return_20d",
    "FEDFUNDS_change_60d",
    "CPI_yoy_change",
    "Yield_Curve_10Y_2Y",
    "M2_yoy_change",
]

REGIME_LABELS = ["Risk-On", "Risk-Off", "Neutral"]


@dataclass
class ValidationResult:
    k_range: list[int]
    inertia: list[float]            # Elbow
    silhouette: list[float]
    gap: list[float]
    gap_se: list[float]
    calinski_harabasz: list[float]


def gap_statistic(X: np.ndarray, k_range: list[int],
                  n_refs: int = 10, random_state: int = 42) -> tuple[list[float], list[float]]:
    """Tibshirani et al. 2001 gap statistic.
    For each k: gap(k) = E[log(W_k_ref)] - log(W_k_obs)
    Higher gap = better cluster structure relative to uniform reference."""
    rng = np.random.default_rng(random_state)
    n, d = X.shape
    mins, maxs = X.min(axis=0), X.max(axis=0)

    gaps, ses = [], []
    for k in k_range:
        # Observed within-cluster dispersion
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        log_W_obs = np.log(km.inertia_)

        # Reference (uniform) dispersion
        log_W_refs = []
        for b in range(n_refs):
            X_ref = rng.uniform(mins, maxs, size=(n, d))
            km_ref = KMeans(n_clusters=k, random_state=random_state + b, n_init=5).fit(X_ref)
            log_W_refs.append(np.log(km_ref.inertia_))
        log_W_ref_mean = np.mean(log_W_refs)
        log_W_ref_std = np.std(log_W_refs)
        gap = log_W_ref_mean - log_W_obs
        se = log_W_ref_std * np.sqrt(1 + 1 / n_refs)
        gaps.append(gap)
        ses.append(se)
    return gaps, ses


def validate_k(X: np.ndarray, k_range: list[int] = None,
               random_state: int = 42) -> ValidationResult:
    """Run all 4 cluster-count validation methods."""
    if k_range is None:
        k_range = [2, 3, 4, 5, 6, 7, 8]

    inertia, silh, ch = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        inertia.append(km.inertia_)
        labels = km.labels_
        silh.append(silhouette_score(X, labels))
        ch.append(calinski_harabasz_score(X, labels))

    gap, gap_se = gap_statistic(X, k_range, n_refs=10, random_state=random_state)

    return ValidationResult(k_range=k_range, inertia=inertia, silhouette=silh,
                            gap=gap, gap_se=gap_se, calinski_harabasz=ch)


@dataclass
class SemanticKMeans:
    """K-Means + semantic relabel wrapper."""
    n_clusters: int = 3
    random_state: int = 42
    feature_names: list[str] = None

    def fit(self, df_pretrain: pd.DataFrame):
        """Fit scaler + K-Means on pre-train derived features."""
        X = df_pretrain[STAGE2_FEATURES].dropna().values
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state,
                              n_init=20).fit(Xs)
        # Centroid in original (un-scaled) space for interpretability
        centroid_scaled = self.kmeans_.cluster_centers_
        self.centroids_ = self.scaler_.inverse_transform(centroid_scaled)
        # Semantic mapping
        self.cluster_to_regime_ = self._semantic_relabel()
        return self

    def _semantic_relabel(self) -> dict[int, str]:
        """Map cluster idx → Risk-On/Off/Neutral by centroid VIX + SP500."""
        cent = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)
        # Score each cluster: high VIX_z + low SP500_5d = Risk-Off score
        # low VIX_z + high SP500_5d = Risk-On score
        risk_off_score = cent["VIX_zscore_long"] - cent["SP500_log_return_5d"] * 50
        risk_on_score = -cent["VIX_zscore_long"] + cent["SP500_log_return_5d"] * 50

        risk_off_idx = int(risk_off_score.idxmax())
        risk_on_idx = int(risk_on_score.idxmax())
        # Ensure distinct
        if risk_on_idx == risk_off_idx:
            # Tie-breaker: pick second-best for Risk-Off
            risk_off_idx = int(risk_off_score.drop(risk_on_idx).idxmax())
        # Remainder = Neutral
        all_idx = set(range(self.n_clusters))
        neutral_idx = (all_idx - {risk_on_idx, risk_off_idx}).pop()
        return {risk_on_idx: "Risk-On", risk_off_idx: "Risk-Off", neutral_idx: "Neutral"}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for each row + return one-hot + raw label."""
        X = df[STAGE2_FEATURES]
        valid_mask = X.notna().all(axis=1)
        Xs = self.scaler_.transform(X[valid_mask].values)
        cluster_ids = self.kmeans_.predict(Xs)

        out = pd.DataFrame(index=df.index)
        out["regime_cluster"] = pd.NA
        out.loc[valid_mask, "regime_cluster"] = cluster_ids
        out["regime_label"] = out["regime_cluster"].map(
            lambda c: self.cluster_to_regime_.get(c) if pd.notna(c) else None
        )
        # One-hot (proposal §3 said "one-hot encoded")
        for r in REGIME_LABELS:
            out[f"P_{r}"] = (out["regime_label"] == r).astype(float)
        # NaN mask
        out.loc[~valid_mask, [f"P_{r}" for r in REGIME_LABELS]] = np.nan
        return out

    def centroid_summary(self) -> pd.DataFrame:
        """Return centroids in original units, indexed by semantic label."""
        df = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)
        df.index = [self.cluster_to_regime_[i] for i in range(self.n_clusters)]
        return df.loc[REGIME_LABELS]   # ordered: Risk-On, Risk-Off, Neutral


# ============================================================
# Phase 2 (V5 2026-05-09): GMM with soft posterior
# Reference: Li et al. 2024 JMLR [LR6] — rare-event mixture model
# ============================================================

@dataclass
class SemanticGMM:
    """Gaussian Mixture Model + semantic relabel + soft posterior.

    Advantages over SemanticKMeans:
      - Soft posterior P(regime | x) instead of hard label
      - Multi-severity capture: COVID 0.95, GFC 0.7, normal 0.0
      - Natural fit for Stage 3 soft fusion (Ting & Witten 1999 [N2])
    """
    n_components: int = 3
    covariance_type: str = "full"   # full / tied / diag / spherical
    random_state: int = 42

    def fit(self, df_pretrain: pd.DataFrame):
        """Fit scaler + GMM on pre-train derived features."""
        X = df_pretrain[STAGE2_FEATURES].dropna().values
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.gmm_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=20,
            max_iter=300,
            reg_covar=1e-4,
        ).fit(Xs)
        # Centroids in original space (means_ → inverse scaler)
        self.centroids_ = self.scaler_.inverse_transform(self.gmm_.means_)
        # Semantic mapping (same logic as K-Means)
        self.cluster_to_regime_ = self._semantic_relabel()
        return self

    def _semantic_relabel(self) -> dict[int, str]:
        cent = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)
        risk_off_score = cent["VIX_zscore_long"] - cent["SP500_log_return_5d"] * 50
        risk_on_score = -cent["VIX_zscore_long"] + cent["SP500_log_return_5d"] * 50

        risk_off_idx = int(risk_off_score.idxmax())
        risk_on_idx = int(risk_on_score.idxmax())
        if risk_on_idx == risk_off_idx:
            risk_off_idx = int(risk_off_score.drop(risk_on_idx).idxmax())
        all_idx = set(range(self.n_components))
        neutral_idx = (all_idx - {risk_on_idx, risk_off_idx}).pop()
        return {risk_on_idx: "Risk-On", risk_off_idx: "Risk-Off", neutral_idx: "Neutral"}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict regime — returns SOFT posterior P(regime | x) for each row."""
        X = df[STAGE2_FEATURES]
        valid_mask = X.notna().all(axis=1)
        Xs = self.scaler_.transform(X[valid_mask].values)

        # Soft posterior P(component | x)
        proba = self.gmm_.predict_proba(Xs)   # (n_valid, n_components)

        out = pd.DataFrame(index=df.index)
        # P_<regime> in semantic order
        for r in REGIME_LABELS:
            out[f"P_{r}"] = np.nan
        for cluster_idx, regime in self.cluster_to_regime_.items():
            col = f"P_{regime}"
            out.loc[valid_mask, col] = proba[:, cluster_idx]

        # Hard label (argmax) for compatibility — only on valid rows
        out["regime_label"] = None
        valid_proba = out.loc[valid_mask, [f"P_{r}" for r in REGIME_LABELS]]
        out.loc[valid_mask, "regime_label"] = valid_proba.idxmax(axis=1).str.replace("P_", "")
        out["regime_cluster"] = pd.NA
        cluster_ids_pred = self.gmm_.predict(Xs)
        out.loc[valid_mask, "regime_cluster"] = cluster_ids_pred
        return out

    def centroid_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)
        df.index = [self.cluster_to_regime_[i] for i in range(self.n_components)]
        return df.loc[REGIME_LABELS]
