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


# ============================================================
# Phase 3 (V5 2026-05-09): Sparse K-Means with L1 feature weighting
# Reference: Witten & Tibshirani (2010) JASA [LR1]
# Algorithm: §3 KMeans Sparse Clustering
# ============================================================

def _soft_threshold(a: np.ndarray, delta: float) -> np.ndarray:
    """Soft-threshold operator S(a, Δ) = sign(a) * max(|a| - Δ, 0)."""
    return np.sign(a) * np.maximum(np.abs(a) - delta, 0.0)


def _compute_a_j(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Compute BCSS contribution per feature j:
       a_j = sum_k n_k * (mean_kj - mean_j)^2 — TSS_j - WCSS_j
       More precisely (Witten 2010): a_j = TSS_j - WCSS_j per feature."""
    n, p = X.shape
    overall_mean = X.mean(axis=0)
    a = np.zeros(p)
    for c in range(k):
        mask = labels == c
        n_c = mask.sum()
        if n_c == 0:
            continue
        cluster_mean = X[mask].mean(axis=0)
        a += n_c * (cluster_mean - overall_mean) ** 2
    return a   # length p, BCSS per feature


def _solve_w(a: np.ndarray, s: float, tol: float = 1e-6, max_iter: int = 50) -> np.ndarray:
    """Find w that maximizes sum_j w_j * a_j s.t. ||w||_2 ≤ 1, ||w||_1 ≤ s, w ≥ 0.
    Witten & Tibshirani 2010 Lemma 1: w = S(a, Δ)+ / ||S(a, Δ)+||_2
    where Δ binary-searched until ||w||_1 = s."""
    a_pos = np.maximum(a, 0.0)
    if a_pos.sum() == 0:
        return np.ones_like(a) / np.sqrt(len(a))
    # Try Δ=0 first
    w_try = a_pos / np.linalg.norm(a_pos)
    if np.sum(w_try) <= s:
        return w_try
    # Binary search on Δ
    lo, hi = 0.0, np.max(a_pos)
    for _ in range(max_iter):
        delta = (lo + hi) / 2
        w_st = _soft_threshold(a_pos, delta)
        norm = np.linalg.norm(w_st)
        if norm < tol:
            hi = delta
            continue
        w = w_st / norm
        l1 = np.sum(w)
        if abs(l1 - s) < tol:
            return w
        if l1 > s:
            lo = delta
        else:
            hi = delta
    w_st = _soft_threshold(a_pos, (lo + hi) / 2)
    norm = np.linalg.norm(w_st)
    return w_st / max(norm, tol)


@dataclass
class SemanticSparseKMeans:
    """Witten-Tibshirani 2010 Sparse K-Means + semantic relabel.
    Feature weights w_j learned per iteration with L1 sparsity penalty.

    Hyperparameter:
      s ∈ [1, sqrt(p)] — L1 bound. s=1 → most sparse, s=sqrt(p) → uniform.
      Default s = sqrt(p)/1.5 (mild sparsity).
    """
    n_clusters: int = 3
    s: float | None = None
    n_outer_iter: int = 20
    random_state: int = 42

    def fit(self, df_pretrain: pd.DataFrame):
        X = df_pretrain[STAGE2_FEATURES].dropna().values
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        n, p = Xs.shape
        s = self.s if self.s is not None else np.sqrt(p) / 1.5
        self.s_ = s

        # Initialize w uniformly
        w = np.ones(p) / np.sqrt(p)
        prev_obj = -np.inf

        for it in range(self.n_outer_iter):
            # Step 1: fixed w, optimize cluster assignments via weighted K-Means
            X_weighted = Xs * np.sqrt(w)
            km = KMeans(n_clusters=self.n_clusters,
                        random_state=self.random_state,
                        n_init=10).fit(X_weighted)
            labels = km.labels_

            # Step 2: fixed labels, optimize w
            a = _compute_a_j(Xs, labels, self.n_clusters)
            w_new = _solve_w(a, s)

            # Convergence check
            obj = float(np.sum(w_new * a))
            if abs(obj - prev_obj) < 1e-5:
                w = w_new
                break
            prev_obj = obj
            w = w_new

        self.feature_weights_ = w
        # Final K-Means with learned weights
        X_weighted = Xs * np.sqrt(w)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state,
                              n_init=20).fit(X_weighted)
        # Centroid in original (un-weighted, un-scaled) space
        # Recompute by averaging original X within each weighted-K-Means cluster
        centroids_unscaled = np.zeros((self.n_clusters, p))
        for c in range(self.n_clusters):
            mask = self.kmeans_.labels_ == c
            centroids_unscaled[c] = X[mask].mean(axis=0)
        self.centroids_ = centroids_unscaled
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
        all_idx = set(range(self.n_clusters))
        neutral_idx = (all_idx - {risk_on_idx, risk_off_idx}).pop()
        return {risk_on_idx: "Risk-On", risk_off_idx: "Risk-Off", neutral_idx: "Neutral"}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[STAGE2_FEATURES]
        valid_mask = X.notna().all(axis=1)
        Xs = self.scaler_.transform(X[valid_mask].values)
        Xw = Xs * np.sqrt(self.feature_weights_)
        cluster_ids = self.kmeans_.predict(Xw)

        out = pd.DataFrame(index=df.index)
        out["regime_cluster"] = pd.NA
        out.loc[valid_mask, "regime_cluster"] = cluster_ids
        out["regime_label"] = out["regime_cluster"].map(
            lambda c: self.cluster_to_regime_.get(c) if pd.notna(c) else None
        )
        for r in REGIME_LABELS:
            out[f"P_{r}"] = (out["regime_label"] == r).astype(float)
        out.loc[~valid_mask, [f"P_{r}" for r in REGIME_LABELS]] = np.nan
        return out

    def centroid_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)
        df.index = [self.cluster_to_regime_[i] for i in range(self.n_clusters)]
        return df.loc[REGIME_LABELS]

    def feature_weights_summary(self) -> pd.Series:
        return pd.Series(self.feature_weights_, index=STAGE2_FEATURES,
                         name="weight").sort_values(ascending=False)


# ============================================================
# Phase 4 (V5 2026-05-09): Constrained K-Means with crisis date priors
# Reference: Wagstaff, Cardie, Rogers & Schroedl (2001) ICML [LR2]
# Algorithm: COP-KMeans (Constrained K-Means)
# ============================================================

# NBER recession dates + major financial crisis events
# These force the Risk-Off cluster to span multi-event severity
CRISIS_DATE_RANGES = [
    ("2001-03-01", "2001-11-30", "NBER dot-com recession"),
    ("2008-09-01", "2009-06-30", "NBER GFC recession (Lehman aftermath)"),
    ("2011-08-01", "2011-10-15", "US debt ceiling + Eurozone crisis"),
    ("2015-08-15", "2016-02-15", "China devaluation + oil crash"),
    ("2018-10-01", "2018-12-31", "Q4 2018 sell-off"),
    ("2020-02-20", "2020-04-30", "COVID crash"),
    ("2022-02-01", "2022-10-31", "Fed hike + Ukraine war"),
]


def build_must_link_anchors(index: pd.DatetimeIndex) -> list[list[int]]:
    """For each crisis range, return list of row indices that share must-link
    constraint (transitive closure: all anchors of one crisis are same cluster)."""
    anchor_groups = []
    for start, end, _label in CRISIS_DATE_RANGES:
        mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
        idxs = np.where(mask)[0].tolist()
        if len(idxs) >= 2:
            anchor_groups.append(idxs)
    return anchor_groups


def cop_kmeans(X: np.ndarray, k: int, must_link_groups: list[list[int]],
               max_iter: int = 100, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """COP-KMeans (Wagstaff 2001).
    must_link_groups: list of index-lists; all indices in a group must share cluster.
    Returns (labels, centroids)."""
    rng = np.random.default_rng(random_state)
    n, p = X.shape

    # Build must-link membership: each anchor → group_id (transitive closure)
    point_to_group = {}
    for gid, group in enumerate(must_link_groups):
        for idx in group:
            point_to_group[idx] = gid

    # Init centroids via standard K-Means++
    init_km = KMeans(n_clusters=k, random_state=random_state, n_init=5).fit(X)
    centroids = init_km.cluster_centers_.copy()

    for it in range(max_iter):
        # Assign each must-link group as a whole → cluster minimizing aggregate distance
        labels = np.full(n, -1, dtype=int)
        # Process must-link groups first
        for gid, group in enumerate(must_link_groups):
            group_mean = X[group].mean(axis=0)
            distances = np.linalg.norm(centroids - group_mean, axis=1)
            best_c = int(distances.argmin())
            labels[group] = best_c
        # Process remaining points
        for i in range(n):
            if labels[i] != -1:
                continue
            distances = np.linalg.norm(centroids - X[i], axis=1)
            labels[i] = int(distances.argmin())

        # Update centroids
        new_centroids = np.array([
            X[labels == c].mean(axis=0) if (labels == c).any() else centroids[c]
            for c in range(k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return labels, centroids


@dataclass
class SemanticConstrainedKMeans:
    """COP-KMeans (Wagstaff 2001) + semantic relabel.

    Crisis date prior: NBER recessions + major events → must-link Risk-Off.
    The cluster these anchors land in gets Risk-Off label automatically
    (since crisis VIX/FFR/SP500 patterns dominate that centroid).
    """
    n_clusters: int = 3
    random_state: int = 42

    def fit(self, df_pretrain: pd.DataFrame):
        df = df_pretrain.dropna(subset=STAGE2_FEATURES)
        X = df[STAGE2_FEATURES].values
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        # Build must-link anchors from CRISIS_DATE_RANGES
        anchor_groups = build_must_link_anchors(df.index)
        # Note: in COP-KMeans, all anchors across ALL crises must form one
        # combined group (transitive: 2008 ≈ 2020 same cluster) → flatten.
        flat_anchors = [idx for grp in anchor_groups for idx in grp]
        combined_group = [flat_anchors] if flat_anchors else []
        n_anchors = len(flat_anchors)

        labels, centroids_scaled = cop_kmeans(
            Xs, k=self.n_clusters,
            must_link_groups=combined_group,
            random_state=self.random_state,
        )
        self.labels_pretrain_ = labels
        self.kmeans_centroids_scaled_ = centroids_scaled
        # Centroids in original space
        self.centroids_ = self.scaler_.inverse_transform(centroids_scaled)
        # Force a fitted KMeans-like object for predict reuse
        self.cluster_to_regime_ = self._semantic_relabel(labels, anchor_groups)
        self.n_anchors_ = n_anchors
        return self

    def _semantic_relabel(self, labels: np.ndarray,
                          anchor_groups: list[list[int]]) -> dict[int, str]:
        """Find which cluster the crisis anchors landed in → Risk-Off.
        Then standard VIX/SP500 logic for Risk-On / Neutral on remaining."""
        all_anchor_idx = [idx for grp in anchor_groups for idx in grp]
        # Vote by majority among anchors
        anchor_clusters = labels[all_anchor_idx]
        risk_off_idx = int(np.bincount(anchor_clusters,
                                       minlength=self.n_clusters).argmax())
        # Risk-On / Neutral on remaining: lowest VIX_z + highest SP500 → Risk-On
        cent = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)
        remaining = [c for c in range(self.n_clusters) if c != risk_off_idx]
        risk_on_score = (-cent.loc[remaining, "VIX_zscore_long"]
                         + cent.loc[remaining, "SP500_log_return_5d"] * 50)
        risk_on_idx = int(risk_on_score.idxmax())
        neutral_idx = [c for c in remaining if c != risk_on_idx][0]
        return {risk_on_idx: "Risk-On", risk_off_idx: "Risk-Off",
                neutral_idx: "Neutral"}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict via nearest scaled centroid (no constraint enforcement on test)."""
        X = df[STAGE2_FEATURES]
        valid_mask = X.notna().all(axis=1)
        Xs = self.scaler_.transform(X[valid_mask].values)
        # Distance to each scaled centroid
        d = np.linalg.norm(Xs[:, None, :] - self.kmeans_centroids_scaled_[None, :, :], axis=2)
        cluster_ids = d.argmin(axis=1)

        out = pd.DataFrame(index=df.index)
        out["regime_cluster"] = pd.NA
        out.loc[valid_mask, "regime_cluster"] = cluster_ids
        out["regime_label"] = out["regime_cluster"].map(
            lambda c: self.cluster_to_regime_.get(c) if pd.notna(c) else None
        )
        for r in REGIME_LABELS:
            out[f"P_{r}"] = (out["regime_label"] == r).astype(float)
        out.loc[~valid_mask, [f"P_{r}" for r in REGIME_LABELS]] = np.nan
        return out

    def centroid_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)
        df.index = [self.cluster_to_regime_[i] for i in range(self.n_clusters)]
        return df.loc[REGIME_LABELS]
