"""
Classifier wrappers for all models.
Provides a unified interface for XGBoost, LightGBM, Random Forest, and MLP.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.helpers import setup_logger

logger = setup_logger(__name__)


class BaseClassifier(ABC):
    """Abstract base class for all classifiers."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def predict(self, X) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        pass

    def get_feature_importance(self) -> np.ndarray | None:
        """Override in subclasses that support feature importance."""
        return None

    def _encode_labels(self, y):
        if not self.is_fitted:
            return self.label_encoder.fit_transform(y)
        return self.label_encoder.transform(y)

    def _decode_labels(self, y_encoded):
        return self.label_encoder.inverse_transform(y_encoded)


class XGBoostClassifier(BaseClassifier):
    """XGBoost classifier wrapper."""

    def __init__(self, **kwargs):
        super().__init__("XGBoost", **kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import xgboost as xgb

        y_enc = self._encode_labels(y_train)
        n_classes = len(np.unique(y_enc))

        default_params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
            "random_state": 42,
            "verbosity": 0,
            "use_label_encoder": False,
            "n_jobs": 1,                # macOS libomp segfault workaround
            "tree_method": "hist",      # faster, single-threaded safe
        }
        default_params.update(self.params)

        self.model = xgb.XGBClassifier(**default_params)

        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_enc = self.label_encoder.transform(y_val)
            eval_set = [(X_val, y_val_enc)]

        self.model.fit(
            X_train, y_enc,
            eval_set=eval_set,
            verbose=False,
        )
        self.classes_ = self.label_encoder.classes_
        self.is_fitted = True
        return self

    def predict(self, X):
        return self._decode_labels(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        if self.model is not None:
            return self.model.feature_importances_
        return None


class LightGBMClassifier(BaseClassifier):
    """LightGBM classifier wrapper."""

    def __init__(self, **kwargs):
        super().__init__("LightGBM", **kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import lightgbm as lgb

        y_enc = self._encode_labels(y_train)

        default_params = {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "objective": "multiclass",
            "random_state": 42,
            "verbose": -1,
            "n_jobs": 1,                # macOS libomp segfault workaround
            "force_col_wise": True,
        }
        default_params.update(self.params)

        self.model = lgb.LGBMClassifier(**default_params)

        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_enc = self.label_encoder.transform(y_val)
            eval_set = [(X_val, y_val_enc)]

        self.model.fit(
            X_train, y_enc,
            eval_set=eval_set,
        )
        self.classes_ = self.label_encoder.classes_
        self.is_fitted = True
        return self

    def predict(self, X):
        return self._decode_labels(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        if self.model is not None:
            return self.model.feature_importances_
        return None


class RandomForestClassifierWrapper(BaseClassifier):
    """Random Forest classifier wrapper."""

    def __init__(self, **kwargs):
        super().__init__("RandomForest", **kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.ensemble import RandomForestClassifier as RFC

        y_enc = self._encode_labels(y_train)

        default_params = {
            "n_estimators": 300,
            "max_depth": 10,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": 1,                # match XGB/LGBM behavior on macOS
        }
        default_params.update(self.params)

        self.model = RFC(**default_params)
        self.model.fit(X_train, y_enc)
        self.classes_ = self.label_encoder.classes_
        self.is_fitted = True
        return self

    def predict(self, X):
        return self._decode_labels(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importance(self):
        if self.model is not None:
            return self.model.feature_importances_
        return None


class MLPClassifierWrapper(BaseClassifier):
    """PyTorch MLP classifier wrapper."""

    def __init__(self, **kwargs):
        super().__init__("MLP", **kwargs)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        y_enc = self._encode_labels(y_train)
        n_classes = len(np.unique(y_enc))
        n_features = X_train.shape[1]

        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)

        # Default params
        hidden_layers = self.params.get("hidden_layers", [128, 64])
        lr = self.params.get("learning_rate", 0.001)
        batch_size = self.params.get("batch_size", 64)
        patience = self.params.get("patience", 10)
        max_epochs = self.params.get("max_epochs", 200)

        # Build model
        layers = []
        in_dim = n_features
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_classes))
        self.model = nn.Sequential(*layers)

        # Training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_tensor = torch.FloatTensor(X_scaled).to(device)
        y_tensor = torch.LongTensor(y_enc).to(device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Validation setup
        X_val_tensor = None
        if X_val is not None and y_val is not None:
            y_val_enc = self.label_encoder.transform(y_val)
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            y_val_tensor = torch.LongTensor(y_val_enc).to(device)

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation
            if X_val_tensor is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val_tensor)
                    val_loss = criterion(val_out, y_val_tensor).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"MLP early stopping at epoch {epoch+1}")
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        self._device = device
        self.classes_ = self.label_encoder.classes_
        self.is_fitted = True
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_encoded = np.argmax(proba, axis=1)
        return self._decode_labels(pred_encoded)

    def predict_proba(self, X):
        import torch

        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self._device)

        with torch.no_grad():
            logits = self.model(X_tensor)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        return proba


class LDAClassifier(BaseClassifier):
    """
    Linear Discriminant Analysis wrapper (classical Pattern Recognition baseline).

    Generative model with two assumptions:
        1. Class-conditional densities are Gaussian: p(x|y=k) = N(x; μ_k, Σ)
        2. All classes share the SAME covariance matrix Σ (vs QDA which allows Σ_k)

    Discriminant function (linear in x):
        g_k(x) = x^T Σ^{-1} μ_k - 0.5 μ_k^T Σ^{-1} μ_k + log π_k

    The decision rule assigns x to class argmax_k g_k(x). The posterior probability
    is obtained via softmax over discriminants.
    """

    def __init__(self, **kwargs):
        super().__init__("LDA", **kwargs)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        y_enc = self._encode_labels(y_train)
        X_scaled = self.scaler.fit_transform(X_train)

        default_params = {"solver": "svd", "store_covariance": True}
        default_params.update(self.params)

        # 'shrinkage' requires lsqr or eigen solver
        if default_params.get("shrinkage") is not None and default_params.get("solver") == "svd":
            default_params["solver"] = "lsqr"

        self.model = LinearDiscriminantAnalysis(**default_params)
        self.model.fit(X_scaled, y_enc)
        self.classes_ = self.label_encoder.classes_
        self.is_fitted = True
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self._decode_labels(self.model.predict(X_scaled))

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self):
        if self.model is None:
            return None
        coefs = np.abs(self.model.coef_)
        if coefs.ndim == 2:
            return coefs.mean(axis=0)
        return coefs


def get_classifier(name: str, **kwargs) -> BaseClassifier:
    """Factory function to create classifier by name."""
    classifiers = {
        "xgboost": XGBoostClassifier,
        "lightgbm": LightGBMClassifier,
        "random_forest": RandomForestClassifierWrapper,
        "mlp": MLPClassifierWrapper,
        "lda": LDAClassifier,
    }
    if name.lower() not in classifiers:
        raise ValueError(f"Unknown classifier: {name}. Available: {list(classifiers.keys())}")
    return classifiers[name.lower()](**kwargs)


def get_all_classifiers(**kwargs) -> dict[str, BaseClassifier]:
    """Create instances of all classifiers."""
    return {
        "xgboost": XGBoostClassifier(**kwargs),
        "lightgbm": LightGBMClassifier(**kwargs),
        "random_forest": RandomForestClassifierWrapper(**kwargs),
        "mlp": MLPClassifierWrapper(**kwargs),
        "lda": LDAClassifier(**kwargs),
    }
