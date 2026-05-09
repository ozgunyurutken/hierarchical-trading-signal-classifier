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

from dataclasses import dataclass, field

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
    "Oil_log_return_20d",
    "FEDFUNDS_change_60d",
    "CPI_yoy_change",
    "Yield_Curve_10Y_2Y",
    "M2_yoy_change",
]

REGIME_LABELS = ["Risk-On", "Risk-Off", "Neutral"]

# Phase 2.4 / 2.5 — direct semantic terminology (Bull/Neutral/Bear).
# Plot color mapping: Bull = green, Neutral = yellow, Bear = red.
BULL_BEAR_LABELS = ["Bull", "Neutral", "Bear"]

# Minimal feature subset for low-dimensional unsupervised regime detection
# (avoids curse-of-dimensionality + outlier-driven cluster collapse seen with
# 9-feature K-Means/Sparse/GMM in earlier ablations).
MINIMAL_FEATURES = ["VIX_zscore_long", "SP500_log_return_5d"]


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
    ("2025-03-01", "2025-04-15", "2025 Trump tariff crisis"),
]

# Calm bull market periods — force Risk-On cluster to span ZIRP/QE-era rallies
# (post-GFC ortasi 3 yıl ZIRP rally K-Means'in serbest fit'inde Neutral'a düşüyordu)
RISK_ON_DATE_RANGES = [
    ("2003-09-01", "2007-08-31", "Post-dot-com boom (low VIX, S&P new highs)"),
    ("2012-06-01", "2014-12-31", "QE / ZIRP rally (S&P new highs, VIX 13-15)"),
    ("2021-01-01", "2021-12-31", "Post-COVID stimulus rally (kalici S&P ATH)"),
    ("2023-10-01", "2024-12-31", "Post-2022 recovery"),
]

# Transition / uncertain recovery periods — force Neutral cluster
# (Stage 3'e 'rejim değişim yaşıyor / karışık sinyal' bilgisi vermek için)
NEUTRAL_DATE_RANGES = [
    ("2020-05-01", "2020-10-31", "Post-COVID early recovery (V-shaped, hala VIX 25+)"),
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


def build_triple_must_link_anchors(index: pd.DatetimeIndex
                                   ) -> tuple[list[int], list[int], list[int]]:
    """Returns (riskoff_anchors, riskon_anchors, neutral_anchors) as flat lists.

    Each list becomes ONE must-link group in cop_kmeans → 3 groups for k=3
    means each anchor type lands in its own cluster. Triple constraint
    enforces semantic boundaries: Crisis vs calm bull vs transition."""
    def _collect(date_ranges):
        out: list[int] = []
        for start, end, _ in date_ranges:
            mask = (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
            out.extend(np.where(mask)[0].tolist())
        return out
    return _collect(CRISIS_DATE_RANGES), _collect(RISK_ON_DATE_RANGES), \
        _collect(NEUTRAL_DATE_RANGES)


def build_dual_must_link_anchors(index: pd.DatetimeIndex
                                 ) -> tuple[list[int], list[int]]:
    """Backwards-compatible 2-anchor variant (Risk-Off + Risk-On only).
    For new code use build_triple_must_link_anchors which adds Neutral."""
    riskoff, riskon, _ = build_triple_must_link_anchors(index)
    return riskoff, riskon


def cop_kmeans(X: np.ndarray, k: int, must_link_groups: list[list[int]],
               max_iter: int = 100, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """COP-KMeans (Wagstaff 2001) + implicit cannot-link via Hungarian
    assignment when num_groups <= k.

    must_link_groups: list of index-lists; all indices in a group must share
    the same cluster, AND distinct groups land in distinct clusters
    (Hungarian-optimal). Returns (labels, centroids)."""
    from scipy.optimize import linear_sum_assignment
    n, p = X.shape

    # Init centroids via standard K-Means++
    init_km = KMeans(n_clusters=k, random_state=random_state, n_init=5).fit(X)
    centroids = init_km.cluster_centers_.copy()

    n_groups = len(must_link_groups)
    use_hungarian = 0 < n_groups <= k

    for it in range(max_iter):
        labels = np.full(n, -1, dtype=int)

        if use_hungarian:
            # Hungarian: cost[g, c] = sum of distances of group g points to centroid c
            cost = np.zeros((n_groups, k))
            for g, group in enumerate(must_link_groups):
                Xg = X[group]
                for c in range(k):
                    cost[g, c] = np.linalg.norm(Xg - centroids[c], axis=1).sum()
            row_idx, col_idx = linear_sum_assignment(cost)
            for g, c in zip(row_idx, col_idx):
                labels[must_link_groups[g]] = int(c)
        else:
            # Fallback: greedy nearest-centroid for each group (no cannot-link)
            for g, group in enumerate(must_link_groups):
                group_mean = X[group].mean(axis=0)
                distances = np.linalg.norm(centroids - group_mean, axis=1)
                labels[group] = int(distances.argmin())

        # Free points → nearest centroid
        unassigned = np.where(labels == -1)[0]
        if len(unassigned) > 0:
            d = np.linalg.norm(X[unassigned, None, :] - centroids[None, :, :], axis=2)
            labels[unassigned] = d.argmin(axis=1)

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
        # Triple must-link: Risk-Off (crisis) + Risk-On (calm bull) +
        # Neutral (transition) anchors. Three separate groups → land in
        # three distinct clusters (k=3) because their feature distributions
        # are markedly different.
        riskoff_anchors, riskon_anchors, neutral_anchors = (
            build_triple_must_link_anchors(df.index)
        )
        must_link_groups: list[list[int]] = []
        if riskoff_anchors:
            must_link_groups.append(riskoff_anchors)
        if riskon_anchors:
            must_link_groups.append(riskon_anchors)
        if neutral_anchors:
            must_link_groups.append(neutral_anchors)

        labels, centroids_scaled = cop_kmeans(
            Xs, k=self.n_clusters,
            must_link_groups=must_link_groups,
            random_state=self.random_state,
        )
        self.labels_pretrain_ = labels
        self.kmeans_centroids_scaled_ = centroids_scaled
        self.centroids_ = self.scaler_.inverse_transform(centroids_scaled)
        self.cluster_to_regime_ = self._semantic_relabel(
            labels, riskoff_anchors, riskon_anchors, neutral_anchors
        )
        self.n_riskoff_anchors_ = len(riskoff_anchors)
        self.n_riskon_anchors_ = len(riskon_anchors)
        self.n_neutral_anchors_ = len(neutral_anchors)
        # Backwards-compatible — total anchor count
        self.n_anchors_ = (len(riskoff_anchors) + len(riskon_anchors)
                           + len(neutral_anchors))
        return self

    def _semantic_relabel(self, labels: np.ndarray,
                          riskoff_anchors: list[int],
                          riskon_anchors: list[int],
                          neutral_anchors: list[int]) -> dict[int, str]:
        """Risk-Off / Risk-On / Neutral cluster id'lerini her anchor list'in
        majority vote'una göre belirle. Anchor list boşsa centroid VIX/SP500
        fallback'i."""
        cent = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)

        # Risk-Off cluster from crisis anchors
        if riskoff_anchors:
            anchor_clusters = labels[riskoff_anchors]
            risk_off_idx = int(np.bincount(anchor_clusters,
                                           minlength=self.n_clusters).argmax())
        else:
            risk_off_idx = int(cent["VIX_zscore_long"].idxmax())

        # Risk-On cluster from calm-bull anchors (exclude risk_off_idx)
        if riskon_anchors:
            counts = np.bincount(labels[riskon_anchors],
                                 minlength=self.n_clusters).copy()
            counts[risk_off_idx] = 0
            risk_on_idx = int(counts.argmax())
        else:
            remaining = [c for c in range(self.n_clusters) if c != risk_off_idx]
            risk_on_score = (-cent.loc[remaining, "VIX_zscore_long"]
                             + cent.loc[remaining, "SP500_log_return_5d"] * 50)
            risk_on_idx = int(risk_on_score.idxmax())

        # Neutral cluster: prefer anchor majority (excluding both above)
        if neutral_anchors:
            counts = np.bincount(labels[neutral_anchors],
                                 minlength=self.n_clusters).copy()
            counts[risk_off_idx] = 0
            counts[risk_on_idx] = 0
            neutral_idx = int(counts.argmax())
            # Fallback if anchor majority overlaps with already-assigned
            if neutral_idx in (risk_off_idx, risk_on_idx):
                neutral_idx = [c for c in range(self.n_clusters)
                               if c not in (risk_off_idx, risk_on_idx)][0]
        else:
            neutral_idx = [c for c in range(self.n_clusters)
                           if c not in (risk_off_idx, risk_on_idx)][0]
        return {risk_on_idx: "Risk-On", risk_off_idx: "Risk-Off",
                neutral_idx: "Neutral"}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict via nearest scaled centroid (no constraint enforcement on test).

        WARNING: For pretrain DataFrame the model was fit on, use
        `get_pretrain_labels()` instead — predict() bypasses must-link
        constraints and causes anchor leakage on training data."""
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

    def get_pretrain_labels(self, df_pretrain: pd.DataFrame) -> pd.DataFrame:
        """Return training-time forced labels (constraint enforced) for the
        pretrain DataFrame the model was fit on. Use INSTEAD of predict()
        for pretrain — predict() doesn't enforce must-link constraints and
        causes anchor leakage."""
        df_valid = df_pretrain.dropna(subset=STAGE2_FEATURES)
        out = pd.DataFrame(index=df_pretrain.index)
        out["regime_cluster"] = pd.NA
        out.loc[df_valid.index, "regime_cluster"] = self.labels_pretrain_
        out["regime_label"] = out["regime_cluster"].map(
            lambda c: self.cluster_to_regime_.get(c) if pd.notna(c) else None
        )
        for r in REGIME_LABELS:
            out[f"P_{r}"] = (out["regime_label"] == r).astype(float)
        out.loc[out["regime_cluster"].isna(),
                [f"P_{r}" for r in REGIME_LABELS]] = np.nan
        return out

    def centroid_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.centroids_, columns=STAGE2_FEATURES)
        df.index = [self.cluster_to_regime_[i] for i in range(self.n_clusters)]
        return df.loc[REGIME_LABELS]


# ============================================================
# Phase 2.4 / 2.5 (V5 2026-05-09): Minimal-feature unsupervised regime
# detection. Manuel anchor mekanizmasi tamamen kaldirildi (kullanici
# 'kümelemenin anlami kalmiyor' uyarisi) — 9-feature high-dim cluster geometry
# yerine 2-feature low-dim semantic separation.
#
# Feature subset MINIMAL_FEATURES = [VIX_zscore_long, SP500_log_return_5d]
# Klasik volatility-return regime detection (Hamilton-Susmel 1994 SWARCH §3).
#
# Output labels: Bull / Neutral / Bear (direct semantic, not Risk-On/Off).
# ============================================================


def _bull_bear_relabel(centroids: np.ndarray, feature_names: list[str],
                       n_clusters: int) -> dict[int, str]:
    """Map cluster idx → Bull / Neutral / Bear by VIX centroid ranking.
    Highest VIX → Bear, lowest → Bull, remainder → Neutral."""
    cent = pd.DataFrame(centroids, columns=feature_names)
    if "VIX_zscore_long" not in cent.columns:
        raise ValueError("Bull/Bear relabel requires VIX_zscore_long in features")
    bear_idx = int(cent["VIX_zscore_long"].idxmax())
    bull_idx = int(cent["VIX_zscore_long"].idxmin())
    if bear_idx == bull_idx:
        sp_score = cent["SP500_log_return_5d"]
        bear_idx = int(sp_score.idxmin())
        bull_idx = int(sp_score.idxmax())
    neutral_idx = [c for c in range(n_clusters)
                   if c not in (bear_idx, bull_idx)][0]
    return {bull_idx: "Bull", neutral_idx: "Neutral", bear_idx: "Bear"}


@dataclass
class MinimalKMeans:
    """Az-feature K-Means k=3 (default 2 features: VIX + SP500 5d return).

    Pure unsupervised — no anchors, no constraints. Cluster geometry is
    natural in low-dim space, semantic labeling via VIX centroid ranking.
    """
    n_clusters: int = 3
    feature_subset: list[str] = field(default_factory=lambda: list(MINIMAL_FEATURES))
    random_state: int = 42

    def fit(self, df_pretrain: pd.DataFrame):
        X = df_pretrain[self.feature_subset].dropna().values
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters,
                              random_state=self.random_state,
                              n_init=20).fit(Xs)
        self.centroids_ = self.scaler_.inverse_transform(self.kmeans_.cluster_centers_)
        self.cluster_to_regime_ = _bull_bear_relabel(
            self.centroids_, self.feature_subset, self.n_clusters)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_subset]
        valid_mask = X.notna().all(axis=1)
        Xs = self.scaler_.transform(X[valid_mask].values)
        cluster_ids = self.kmeans_.predict(Xs)

        out = pd.DataFrame(index=df.index)
        out["regime_cluster"] = pd.NA
        out.loc[valid_mask, "regime_cluster"] = cluster_ids
        out["regime_label"] = out["regime_cluster"].map(
            lambda c: self.cluster_to_regime_.get(c) if pd.notna(c) else None
        )
        for r in BULL_BEAR_LABELS:
            out[f"P_{r}"] = (out["regime_label"] == r).astype(float)
        out.loc[~valid_mask, [f"P_{r}" for r in BULL_BEAR_LABELS]] = np.nan
        return out

    def centroid_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.centroids_, columns=self.feature_subset)
        df.index = [self.cluster_to_regime_[i] for i in range(self.n_clusters)]
        return df.loc[BULL_BEAR_LABELS]


@dataclass
class MinimalGaussianHMM:
    """3-state Gaussian HMM (Hamilton 1989 [N6]) on minimal feature subset.

    Markov-chain regime transitions with built-in persistence (smooth
    state changes). Returns soft posterior P(regime | x) via forward-
    backward — ideal for Stage 3 soft fusion.
    """
    n_states: int = 3
    feature_subset: list[str] = field(default_factory=lambda: list(MINIMAL_FEATURES))
    covariance_type: str = "full"
    n_iter: int = 100
    random_state: int = 42

    def fit(self, df_pretrain: pd.DataFrame):
        from hmmlearn.hmm import GaussianHMM
        X = df_pretrain[self.feature_subset].dropna().values
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.hmm_ = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        ).fit(Xs)
        self.centroids_ = self.scaler_.inverse_transform(self.hmm_.means_)
        self.transmat_ = self.hmm_.transmat_
        self.state_to_regime_ = _bull_bear_relabel(
            self.centroids_, self.feature_subset, self.n_states)
        self.cluster_to_regime_ = self.state_to_regime_
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Viterbi (hard label) + posterior (soft P(regime | x))."""
        X = df[self.feature_subset]
        valid_mask = X.notna().all(axis=1)
        Xs = self.scaler_.transform(X[valid_mask].values)
        states_hard = self.hmm_.predict(Xs)
        posterior = self.hmm_.predict_proba(Xs)

        out = pd.DataFrame(index=df.index)
        out["regime_state"] = pd.NA
        out.loc[valid_mask, "regime_state"] = states_hard
        out["regime_label"] = out["regime_state"].map(
            lambda s: self.state_to_regime_.get(s) if pd.notna(s) else None
        )
        for r in BULL_BEAR_LABELS:
            out[f"P_{r}"] = np.nan
        for state_idx, regime in self.state_to_regime_.items():
            out.loc[valid_mask, f"P_{regime}"] = posterior[:, state_idx]
        return out

    def centroid_summary(self) -> pd.DataFrame:
        df = pd.DataFrame(self.centroids_, columns=self.feature_subset)
        df.index = [self.state_to_regime_[i] for i in range(self.n_states)]
        return df.loc[BULL_BEAR_LABELS]

    def transition_summary(self) -> pd.DataFrame:
        labels = [self.state_to_regime_[i] for i in range(self.n_states)]
        df = pd.DataFrame(self.transmat_, index=labels, columns=labels)
        return df.loc[BULL_BEAR_LABELS, BULL_BEAR_LABELS]


# ============================================================
# Phase 2.6 (V5 2026-05-09): Rule-based VIX threshold classifier
# (Hamilton-Susmel 1994 SWARCH-benchmark single-feature volatility regime).
# Pragmatic fallback after K-Means/HMM ablations failed clean semantic
# separation. Önceki proposal projesinde benzer rule-based VIX yaklaşımı
# çalışmıştı; deterministic + interpretable.
# ============================================================


@dataclass
class RuleBasedVIXClassifier:
    """VIX z-score threshold classifier with persistence filter.

    Default thresholds (calibrated to 25-year VIX baseline):
      - VIX_z > +1.0 → Bear  (~ VIX > 27, panic threshold)
      - VIX_z < -0.5 → Bull  (~ VIX < 15, calm threshold)
      - between      → Neutral (transition / elevated-but-not-panic zone)

    Persistence filter: runs shorter than `persistence_days` merged into
    the preceding run's label (smooths daily VIX flicker).
    """
    bear_threshold: float = 1.0
    bull_threshold: float = -0.5
    persistence_days: int = 7
    feature_name: str = "VIX_zscore_long"

    def fit(self, df_pretrain: pd.DataFrame):
        # No parameters to learn — included for API parity with other models.
        # Compute distribution preview on pretrain for diagnostics.
        vix_z = df_pretrain[self.feature_name].dropna()
        self.pretrain_vix_z_stats_ = {
            "mean": float(vix_z.mean()),
            "std": float(vix_z.std()),
            "p10": float(vix_z.quantile(0.10)),
            "p50": float(vix_z.quantile(0.50)),
            "p90": float(vix_z.quantile(0.90)),
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        vix_z = df[self.feature_name]
        valid_mask = vix_z.notna()
        raw_label = pd.Series(index=df.index, dtype=object)
        raw_label[vix_z > self.bear_threshold] = "Bear"
        raw_label[vix_z < self.bull_threshold] = "Bull"
        between = ((vix_z >= self.bull_threshold)
                   & (vix_z <= self.bear_threshold))
        raw_label[between] = "Neutral"
        raw_label[~valid_mask] = None

        if self.persistence_days > 1:
            valid_idx = raw_label.index[valid_mask]
            smoothed = self._persistence_filter(
                raw_label.loc[valid_idx], self.persistence_days)
            raw_label.loc[valid_idx] = smoothed.values

        out = pd.DataFrame(index=df.index)
        out["regime_label"] = raw_label
        for r in BULL_BEAR_LABELS:
            out[f"P_{r}"] = (raw_label == r).astype(float)
        out.loc[~valid_mask, [f"P_{r}" for r in BULL_BEAR_LABELS]] = np.nan
        return out

    @staticmethod
    def _persistence_filter(labels: pd.Series, min_days: int) -> pd.Series:
        """Merge runs shorter than min_days into preceding run's label.
        Two-pass to handle cascading short runs (e.g. AAA-B-A-CCC after first
        pass becomes AAAAA-A-CCC = AAAAAA-CCC after second pass)."""
        if len(labels) == 0:
            return labels
        out = labels.values.copy()
        for _pass in range(2):
            runs = []
            cur, start = out[0], 0
            for i in range(1, len(out)):
                if out[i] != cur:
                    runs.append((start, i, cur))
                    cur = out[i]
                    start = i
            runs.append((start, len(out), cur))
            for j, (s, e, lbl) in enumerate(runs):
                if e - s < min_days:
                    if j > 0:
                        out[s:e] = runs[j - 1][2]
                    elif j < len(runs) - 1:
                        out[s:e] = runs[j + 1][2]
        return pd.Series(out, index=labels.index)


# ============================================================
# Phase 2.7 (V5 2026-05-09): Composite VIX-based finite state machine.
# Phase 2.6 (saf VIX threshold) iki sorun gösterdi:
#   - 2018 Q4: Bear spike kısa, hemen Neutral'a düştü (Bear yeterince stickyhear değil)
#   - 2020 COVID: Spike çok büyük, recovery uzun (Bull'a geri dönüş gecikti)
#
# Çözüm:
#   1) Asymmetric (hysteresis) thresholds — Bear giriş kolay, çıkış zor
#   2) Minimum dwell time — Bear'a girince min 20 gün, Bull'a girince min 30 gün
#   3) VIX velocity override — VIX hızla düşerken (5d Δ < -0.8) ve SP500
#      pozitifse, Bear'dan Neutral'a fast-track (dwell time bypass)
# ============================================================


@dataclass
class CompositeVIXRegimeClassifier:
    """Phase 2.7 — Hysteresis + Dwell + VIX Velocity composite state machine.

    State transitions (asymmetric, single-hop):
      Bull    → Neutral : VIX_z > bull_exit_threshold (and dwell ≥ bull_min_dwell)
      Neutral → Bear    : VIX_z > bear_entry_threshold
      Neutral → Bull    : VIX_z < bull_entry_threshold AND SP500_5d > 0
      Bear    → Neutral : VIX_z < bear_exit_threshold (and dwell ≥ bear_min_dwell)
                          OR velocity override

    Velocity override (Bear → Neutral fast-track):
      VIX_z[t] - VIX_z[t-velocity_window] < velocity_threshold
      AND SP500_5d > velocity_sp500_min

    Direct Bull↔Bear transitions disallowed (must pass through Neutral).
    """
    bear_entry_threshold: float = 1.0
    bear_exit_threshold: float = 0.3
    bull_entry_threshold: float = -0.5
    bull_exit_threshold: float = 0.0
    bear_min_dwell: int = 20
    bull_min_dwell: int = 30
    velocity_window: int = 5
    velocity_threshold: float = -0.8
    velocity_sp500_min: float = 0.0
    feature_vix: str = "VIX_zscore_long"
    feature_sp500: str = "SP500_log_return_5d"
    initial_regime: str = "Neutral"

    # Phase 2.8 extension fields (defaults disable behavior to keep
    # Phase 2.7 backward-compatible).
    feature_yield_curve: str = "Yield_Curve_10Y_2Y"
    enable_yield_curve_override: bool = False
    yield_curve_inverted_threshold: float = 0.0
    yield_curve_persistence_window: int = 30
    yield_curve_blocks_bull_entry: bool = False  # Phase 2.8 had True; 2.9 default False

    # Phase 2.10: YC override conditional on S&P 500 momentum weakness
    # If True, YC override only fires when SP500 long-window return < threshold
    # (i.e., yield curve recession signal corroborated by actual market weakness).
    # This prevents 2024-style false positives where YC inverted but S&P at ATH.
    yc_requires_sp500_weakness: bool = False
    feature_sp500_long_return: str = "SP500_60d_return"
    sp500_weakness_threshold: float = 0.0

    # Phase 2.10: Bull entry velocity override (V-shape recovery detection).
    # Standard Bull entry needs VIX_z < bull_entry_threshold (e.g., -0.5).
    # In V-shape recoveries (2020 May, 2022 Q4) VIX z-score is still positive
    # but rapidly falling AND SP500 has recovered. This branch lets us enter
    # Bull early in those cases without changing other regions.
    enable_bull_velocity_entry: bool = False
    bull_velocity_window: int = 30
    bull_velocity_threshold: float = -0.6        # ΔVIX_z[30d] < -0.6
    bull_velocity_sp500_min: float = 0.05        # SP500_60d_return > +5%

    # Phase 2.11: Bear velocity entry override (rapid escalation).
    # Standard Bear entry needs VIX_z > bear_entry_threshold (e.g., 1.0).
    # In rapid shock events (2025 Apr Liberation Day, 2018 Feb volmageddon),
    # VIX spikes quickly while SP500 sells off — but VIX_z may not yet exceed
    # 1.0. This branch catches Bear earlier.
    enable_bear_velocity_entry: bool = False
    bear_velocity_entry_window: int = 5
    bear_velocity_entry_threshold: float = 0.6   # ΔVIX_z[5d] > +0.6
    bear_velocity_entry_sp500_max: float = -0.015 # SP500_5d < -1.5%

    # Phase 2.11b: minimum Neutral dwell before allowing Bear re-entry.
    # Prevents flip-flop after Bear→Neutral transition (e.g., 2020 Apr-Aug noise).
    bear_reentry_min_neutral_dwell: int = 0

    # Phase 2.9 extension: DXY + M2 macro stress composite (Rule 5)
    feature_dxy: str = "DXY_zscore_long"
    feature_m2: str = "M2_yoy_change"
    enable_macro_stress_override: bool = False
    dxy_strong_threshold: float = 1.0
    m2_low_threshold: float = 0.023
    macro_stress_window: int = 30
    macro_stress_combine: str = "AND"

    def fit(self, df_pretrain: pd.DataFrame):
        vix_z = df_pretrain[self.feature_vix].dropna()
        self.pretrain_vix_z_stats_ = {
            "mean": float(vix_z.mean()),
            "std": float(vix_z.std()),
            "p10": float(vix_z.quantile(0.10)),
            "p50": float(vix_z.quantile(0.50)),
            "p90": float(vix_z.quantile(0.90)),
        }
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df_sorted = df.sort_index()
        vix_z = df_sorted[self.feature_vix]
        sp_5d = df_sorted[self.feature_sp500]
        vix_velocity = vix_z - vix_z.shift(self.velocity_window)
        vix_velocity_bull = vix_z - vix_z.shift(self.bull_velocity_window)
        vix_velocity_bear_entry = vix_z - vix_z.shift(self.bear_velocity_entry_window)

        # Phase 2.8: persistent yield curve inversion (rolling-window mean < threshold)
        if self.enable_yield_curve_override and self.feature_yield_curve in df_sorted.columns:
            yc = df_sorted[self.feature_yield_curve]
            yc_persistent = (yc.rolling(self.yield_curve_persistence_window,
                                         min_periods=self.yield_curve_persistence_window).mean()
                             < self.yield_curve_inverted_threshold)
            yc_arr = yc_persistent.values
        else:
            yc_arr = np.zeros(len(df_sorted), dtype=bool)

        # Phase 2.10: SP500 long-window weakness gate for YC override
        if (self.yc_requires_sp500_weakness
                and self.feature_sp500_long_return in df_sorted.columns):
            sp_long = df_sorted[self.feature_sp500_long_return]
            sp_weak = (sp_long < self.sp500_weakness_threshold).fillna(False)
            sp_weak_arr = sp_weak.values
        else:
            sp_weak_arr = np.ones(len(df_sorted), dtype=bool)  # default: always true (no gate)

        # Phase 2.10: Bull velocity entry — long SP500 return for V-shape recovery
        if (self.enable_bull_velocity_entry
                and self.feature_sp500_long_return in df_sorted.columns):
            sp_long_series = df_sorted[self.feature_sp500_long_return]
            sp_long_arr = sp_long_series.values.astype(float)
        else:
            sp_long_arr = np.full(len(df_sorted), np.nan)
        vix_velocity_bull_arr = vix_velocity_bull.values
        vix_velocity_bear_entry_arr = vix_velocity_bear_entry.values

        # Phase 2.9: DXY + M2 macro stress composite
        if (self.enable_macro_stress_override
                and self.feature_dxy in df_sorted.columns
                and self.feature_m2 in df_sorted.columns):
            dxy_z = df_sorted[self.feature_dxy]
            m2_yoy = df_sorted[self.feature_m2]
            dxy_strong = (dxy_z.rolling(self.macro_stress_window,
                                         min_periods=self.macro_stress_window).mean()
                          > self.dxy_strong_threshold)
            m2_low = (m2_yoy.rolling(self.macro_stress_window,
                                      min_periods=self.macro_stress_window).mean()
                      < self.m2_low_threshold)
            if self.macro_stress_combine == "AND":
                macro_stress = dxy_strong & m2_low
            else:
                macro_stress = dxy_strong | m2_low
            macro_stress_arr = macro_stress.values
        else:
            macro_stress_arr = np.zeros(len(df_sorted), dtype=bool)

        labels_arr = np.empty(len(df_sorted), dtype=object)
        velocity_triggered = np.zeros(len(df_sorted), dtype=bool)
        yc_triggered = np.zeros(len(df_sorted), dtype=bool)
        macro_stress_triggered = np.zeros(len(df_sorted), dtype=bool)
        bull_velocity_triggered = np.zeros(len(df_sorted), dtype=bool)
        bear_velocity_triggered = np.zeros(len(df_sorted), dtype=bool)
        current_regime = self.initial_regime
        days_in_regime = 0
        last_neutral_from_bear = False  # Phase 2.11b: track Bear→Neutral transitions

        v_arr = vix_z.values
        sp_arr = sp_5d.values
        vel_arr = vix_velocity.values

        for i in range(len(df_sorted)):
            v = v_arr[i]
            sp = sp_arr[i]
            vel = vel_arr[i]
            yc_inverted = bool(yc_arr[i])
            sp_weak = bool(sp_weak_arr[i])
            macro_stress = bool(macro_stress_arr[i])

            if pd.isna(v):
                labels_arr[i] = None
                continue

            new_regime = current_regime
            override = False
            yc_force = False
            ms_force = False
            bull_vel_force = False
            bear_vel_force = False

            if current_regime == "Bull":
                # Phase 2.10: yield curve persistent inversion AND SP500 weakness → defensive force
                if (yc_inverted and sp_weak
                        and days_in_regime >= self.bull_min_dwell):
                    new_regime = "Neutral"
                    yc_force = True
                # Phase 2.9: macro stress composite (DXY strong + M2 low) → defensive force
                elif macro_stress and days_in_regime >= self.bull_min_dwell:
                    new_regime = "Neutral"
                    ms_force = True
                elif (v > self.bull_exit_threshold
                        and days_in_regime >= self.bull_min_dwell):
                    new_regime = "Neutral"
            elif current_regime == "Neutral":
                # Phase 2.11: Bear velocity entry (rapid escalation, e.g., 2025 Apr)
                bear_velocity_entry = False
                if self.enable_bear_velocity_entry:
                    vbe = vix_velocity_bear_entry_arr[i]
                    bear_velocity_entry = (
                        pd.notna(vbe)
                        and vbe > self.bear_velocity_entry_threshold
                        and pd.notna(sp)
                        and sp < self.bear_velocity_entry_sp500_max
                    )
                # Phase 2.11b: re-entry dwell guard — only blocks if Neutral
                # was entered from Bear (prevents Bear→Neutral→Bear flip-flop).
                # Allows immediate Bear entry if Neutral came from Bull.
                bear_reentry_ok = (
                    not last_neutral_from_bear
                    or days_in_regime >= self.bear_reentry_min_neutral_dwell
                )
                if v > self.bear_entry_threshold and bear_reentry_ok:
                    new_regime = "Bear"
                elif bear_velocity_entry and bear_reentry_ok:
                    new_regime = "Bear"
                    bear_vel_force = True
                else:
                    # Bull entry: VIX low + SP500 positive
                    bull_blocked = (
                        self.yield_curve_blocks_bull_entry and yc_inverted
                    )
                    standard_bull_entry = (
                        v < self.bull_entry_threshold
                        and pd.notna(sp) and sp > 0
                    )
                    # Phase 2.10: V-shape recovery — VIX falling fast + SP500 recovered
                    velocity_bull_entry = False
                    if self.enable_bull_velocity_entry:
                        vbv = vix_velocity_bull_arr[i]
                        spl = sp_long_arr[i]
                        velocity_bull_entry = (
                            pd.notna(vbv)
                            and vbv < self.bull_velocity_threshold
                            and pd.notna(spl)
                            and spl > self.bull_velocity_sp500_min
                        )
                    if (standard_bull_entry or velocity_bull_entry) and not bull_blocked:
                        new_regime = "Bull"
                        if velocity_bull_entry and not standard_bull_entry:
                            bull_vel_force = True
            elif current_regime == "Bear":
                vel_override = (pd.notna(vel) and vel < self.velocity_threshold
                                and pd.notna(sp)
                                and sp > self.velocity_sp500_min)
                if vel_override:
                    new_regime = "Neutral"
                    override = True
                elif (v < self.bear_exit_threshold
                      and days_in_regime >= self.bear_min_dwell):
                    new_regime = "Neutral"

            if new_regime != current_regime:
                # Phase 2.11b: track if Neutral was entered from Bear
                if new_regime == "Neutral":
                    last_neutral_from_bear = (current_regime == "Bear")
                current_regime = new_regime
                days_in_regime = 1
                velocity_triggered[i] = override
                yc_triggered[i] = yc_force
                macro_stress_triggered[i] = ms_force
                bull_velocity_triggered[i] = bull_vel_force
                bear_velocity_triggered[i] = bear_vel_force
            else:
                days_in_regime += 1

            labels_arr[i] = current_regime

        labels = pd.Series(labels_arr, index=df_sorted.index)
        out = pd.DataFrame(index=df.index)
        out["regime_label"] = labels.reindex(df.index)
        out["velocity_override"] = pd.Series(
            velocity_triggered, index=df_sorted.index
        ).reindex(df.index, fill_value=False)
        out["yield_curve_override"] = pd.Series(
            yc_triggered, index=df_sorted.index
        ).reindex(df.index, fill_value=False)
        out["macro_stress_override"] = pd.Series(
            macro_stress_triggered, index=df_sorted.index
        ).reindex(df.index, fill_value=False)
        out["bull_velocity_entry"] = pd.Series(
            bull_velocity_triggered, index=df_sorted.index
        ).reindex(df.index, fill_value=False)
        out["bear_velocity_entry"] = pd.Series(
            bear_velocity_triggered, index=df_sorted.index
        ).reindex(df.index, fill_value=False)
        for r in BULL_BEAR_LABELS:
            out[f"P_{r}"] = (out["regime_label"] == r).astype(float)
        valid = vix_z.notna() & sp_5d.notna()
        valid_idx = valid.reindex(df.index, fill_value=False)
        out.loc[~valid_idx, [f"P_{r}" for r in BULL_BEAR_LABELS]] = np.nan
        return out
