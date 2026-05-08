"""
Full pipeline orchestration.
Three pipeline configurations: Flat baseline, 2-Stage, 3-Stage.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from src.models.classifiers import get_classifier, BaseClassifier
from src.utils.config import cfg, get_project_root
from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class FullPipeline:
    """
    3-Stage hierarchical pipeline.
    Stage 1 (Trend) + Stage 2 (Macro Regime) -> Stage 3 (Signal)
    """

    def __init__(
        self,
        stage1_model: BaseClassifier,
        stage2_model: BaseClassifier,
        stage3_model: BaseClassifier,
    ):
        self.stage1 = stage1_model
        self.stage2 = stage2_model
        self.stage3 = stage3_model

    def predict(
        self,
        X_trend: pd.DataFrame,
        X_macro: pd.DataFrame,
        X_oscillator: pd.DataFrame,
    ) -> np.ndarray:
        """Predict signal labels through all 3 stages."""
        proba = self.predict_proba(X_trend, X_macro, X_oscillator)
        pred_idx = np.argmax(proba["signal_probs"], axis=1)
        return self.stage3._decode_labels(pred_idx)

    def predict_proba(
        self,
        X_trend: pd.DataFrame,
        X_macro: pd.DataFrame,
        X_oscillator: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """
        Get probability predictions from all stages.

        Returns
        -------
        dict with keys:
            'trend_probs': (n, 3) array
            'regime_probs': (n, 3) array
            'signal_probs': (n, 3) array
        """
        # Stage 1
        X_trend_clean = X_trend.fillna(X_trend.median())
        trend_probs = self.stage1.predict_proba(X_trend_clean)

        # Stage 2
        X_macro_clean = X_macro.fillna(X_macro.median())
        regime_probs = self.stage2.predict_proba(X_macro_clean)

        # Stage 3 input
        trend_df = pd.DataFrame(
            trend_probs,
            index=X_oscillator.index,
            columns=[f"trend_prob_{c}" for c in self.stage1.classes_],
        )
        regime_df = pd.DataFrame(
            regime_probs,
            index=X_oscillator.index,
            columns=[f"regime_prob_{c}" for c in self.stage2.classes_],
        )
        X_combined = pd.concat([X_oscillator, trend_df, regime_df], axis=1)
        X_combined_clean = X_combined.fillna(X_combined.median())

        signal_probs = self.stage3.predict_proba(X_combined_clean)

        return {
            "trend_probs": trend_probs,
            "regime_probs": regime_probs,
            "signal_probs": signal_probs,
        }

    def save(self, directory: str | Path) -> None:
        """Save all 3 stage models."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.stage1, directory / "stage1_model.joblib")
        joblib.dump(self.stage2, directory / "stage2_model.joblib")
        joblib.dump(self.stage3, directory / "stage3_model.joblib")
        logger.info(f"Pipeline saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "FullPipeline":
        """Load pipeline from saved models."""
        directory = Path(directory)
        stage1 = joblib.load(directory / "stage1_model.joblib")
        stage2 = joblib.load(directory / "stage2_model.joblib")
        stage3 = joblib.load(directory / "stage3_model.joblib")
        return cls(stage1, stage2, stage3)


class HierarchicalSoftPipeline:
    """
    3-Stage hierarchical pipeline using a SOFT POSTERIOR Stage 2 (no supervised classifier).

    Differences from `FullPipeline`:
        - Stage 2 is a clustering artifact (GaussianMixture / KMeans / GaussianHMM)
          plus its scaler and method tag, not a `BaseClassifier`.
        - Stage 2 produces `regime_prob_0..k-1` columns (numeric cluster IDs), not
          semantic names — the Stage 3 model learns the mapping implicitly.

    This is the MVP architecture per the 2026-05-07 design decisions.
    """

    def __init__(
        self,
        stage1_model: BaseClassifier,
        stage2_artifact: dict,  # {"model": ..., "scaler": ..., "method": "gmm"|"kmeans"|"hmm", "n_clusters": int}
        stage3_model: BaseClassifier,
    ):
        self.stage1 = stage1_model
        self.stage2_artifact = stage2_artifact
        self.stage3 = stage3_model
        self.n_regime_clusters = stage2_artifact.get("n_clusters", 3)

    def _predict_regime_posterior(self, X_macro: pd.DataFrame) -> np.ndarray:
        """Predict Stage 2 regime posterior probabilities using the cluster artifact."""
        method = self.stage2_artifact["method"]
        scaler = self.stage2_artifact["scaler"]
        model = self.stage2_artifact["model"]

        X_clean = X_macro.fillna(X_macro.median())
        X_scaled = scaler.transform(X_clean)

        if method in ("gmm", "hmm"):
            return model.predict_proba(X_scaled)
        if method == "kmeans":
            dists = model.transform(X_scaled)
            neg = -dists
            exp = np.exp(neg - neg.max(axis=1, keepdims=True))
            return exp / exp.sum(axis=1, keepdims=True)
        raise ValueError(f"Unknown Stage 2 method: {method}")

    def predict_proba(
        self,
        X_trend: pd.DataFrame,
        X_macro: pd.DataFrame,
        X_oscillator: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        """Get probability predictions from all stages."""
        # Stage 1
        X_trend_clean = X_trend.fillna(X_trend.median())
        trend_probs = self.stage1.predict_proba(X_trend_clean)

        # Stage 2 (cluster posterior)
        regime_probs = self._predict_regime_posterior(X_macro)

        # Stage 3
        trend_df = pd.DataFrame(
            trend_probs,
            index=X_oscillator.index,
            columns=[f"trend_prob_{c}" for c in self.stage1.classes_],
        )
        regime_df = pd.DataFrame(
            regime_probs,
            index=X_oscillator.index,
            columns=[f"regime_prob_{i}" for i in range(self.n_regime_clusters)],
        )
        X_combined = pd.concat([X_oscillator, trend_df, regime_df], axis=1)
        X_combined_clean = X_combined.fillna(X_combined.median())

        signal_probs = self.stage3.predict_proba(X_combined_clean)
        return {
            "trend_probs": trend_probs,
            "regime_probs": regime_probs,
            "signal_probs": signal_probs,
        }

    def predict(
        self,
        X_trend: pd.DataFrame,
        X_macro: pd.DataFrame,
        X_oscillator: pd.DataFrame,
    ) -> np.ndarray:
        proba = self.predict_proba(X_trend, X_macro, X_oscillator)
        pred_idx = np.argmax(proba["signal_probs"], axis=1)
        return self.stage3._decode_labels(pred_idx)

    def save(self, directory: str | Path) -> None:
        """Save all artifacts (stage 1 model, stage 2 cluster artifact, stage 3 model)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.stage1, directory / "stage1_model.joblib")
        joblib.dump(self.stage2_artifact, directory / "stage2_cluster_artifact.joblib")
        joblib.dump(self.stage3, directory / "stage3_model.joblib")
        logger.info(f"HierarchicalSoftPipeline saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "HierarchicalSoftPipeline":
        directory = Path(directory)
        stage1 = joblib.load(directory / "stage1_model.joblib")
        stage2_artifact = joblib.load(directory / "stage2_cluster_artifact.joblib")
        stage3 = joblib.load(directory / "stage3_model.joblib")
        return cls(stage1, stage2_artifact, stage3)


class TwoStagePipeline:
    """
    2-Stage pipeline (ablation Config B).
    Stage 1 (Trend) -> Stage 3 (Signal). No macro regime.
    """

    def __init__(
        self,
        stage1_model: BaseClassifier,
        stage3_model: BaseClassifier,
    ):
        self.stage1 = stage1_model
        self.stage3 = stage3_model

    def predict(self, X_trend: pd.DataFrame, X_oscillator: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X_trend, X_oscillator)
        pred_idx = np.argmax(proba["signal_probs"], axis=1)
        return self.stage3._decode_labels(pred_idx)

    def predict_proba(
        self,
        X_trend: pd.DataFrame,
        X_oscillator: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        X_trend_clean = X_trend.fillna(X_trend.median())
        trend_probs = self.stage1.predict_proba(X_trend_clean)

        trend_df = pd.DataFrame(
            trend_probs,
            index=X_oscillator.index,
            columns=[f"trend_prob_{c}" for c in self.stage1.classes_],
        )
        X_combined = pd.concat([X_oscillator, trend_df], axis=1)
        X_combined_clean = X_combined.fillna(X_combined.median())

        signal_probs = self.stage3.predict_proba(X_combined_clean)

        return {
            "trend_probs": trend_probs,
            "signal_probs": signal_probs,
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.stage1, directory / "stage1_model.joblib")
        joblib.dump(self.stage3, directory / "stage3_model_2stage.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "TwoStagePipeline":
        directory = Path(directory)
        stage1 = joblib.load(directory / "stage1_model.joblib")
        stage3 = joblib.load(directory / "stage3_model_2stage.joblib")
        return cls(stage1, stage3)


class FlatBaseline:
    """
    Flat baseline (ablation Config A).
    Single model using ALL features -> Buy/Sell/Hold directly.
    """

    def __init__(self, model: BaseClassifier):
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.fillna(X.median())
        return self.model.predict(X_clean)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_clean = X.fillna(X.median())
        return self.model.predict_proba(X_clean)

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "flat_baseline_model.joblib")

    @classmethod
    def load(cls, directory: str | Path) -> "FlatBaseline":
        directory = Path(directory)
        model = joblib.load(directory / "flat_baseline_model.joblib")
        return cls(model)
