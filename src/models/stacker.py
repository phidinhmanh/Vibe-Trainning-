"""
Ensemble stacking for CAFA 6.

Combines predictions from multiple base models using a meta-learner.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class Stacker:
    """
    Stacking ensemble for multi-label prediction.

    Level 0: Base models (LR, KNN, XGBoost, etc.)
    Level 1: Meta-learner trained on out-of-fold predictions

    Critical: Use out-of-fold predictions from same GroupKFold splits
    to prevent information leakage.
    """

    def __init__(
        self,
        meta_learner: str = "ridge",
        normalize_base: bool = True,
        per_class_meta: bool = True,
        n_jobs: int = -1,
    ):
        """
        Args:
            meta_learner: Type of meta-learner ('ridge', 'logistic', 'avg').
            normalize_base: Whether to normalize base model predictions.
            per_class_meta: Train separate meta-learner per class.
            n_jobs: Number of parallel jobs.
        """
        self.meta_learner = meta_learner
        self.normalize_base = normalize_base
        self.per_class_meta = per_class_meta
        self.n_jobs = n_jobs

        self.scalers_: List[StandardScaler] = []
        self.meta_models_: List = []
        self.n_classes_: int = 0
        self.n_base_models_: int = 0

    def fit(self, oof_predictions: List[np.ndarray], y_true: np.ndarray) -> "Stacker":
        """
        Fit meta-learner on out-of-fold predictions.

        Args:
            oof_predictions: List of OOF predictions from base models.
                            Each element: shape (N, C).
            y_true: Ground truth labels, shape (N, C).

        Returns:
            self
        """
        self.n_base_models_ = len(oof_predictions)
        self.n_classes_ = y_true.shape[1]

        # Stack base predictions: (N, n_base, C) -> process per class
        base_stack = np.stack(oof_predictions, axis=1)  # (N, n_base, C)

        self.scalers_ = []
        self.meta_models_ = []

        if self.per_class_meta:
            # Train one meta-learner per class
            for c in tqdm(range(self.n_classes_), desc="Training meta-learners"):
                X_meta = base_stack[:, :, c]  # (N, n_base)
                y_meta = y_true[:, c]  # (N,)

                # Normalize
                if self.normalize_base:
                    scaler = StandardScaler()
                    X_meta = scaler.fit_transform(X_meta)
                    self.scalers_.append(scaler)
                else:
                    self.scalers_.append(None)

                # Fit meta-learner
                if self.meta_learner == "ridge":
                    model = RidgeCV(alphas=[0.1, 1.0, 10.0])
                    model.fit(X_meta, y_meta)
                elif self.meta_learner == "logistic":
                    model = LogisticRegression(max_iter=500, random_state=42)
                    model.fit(X_meta, y_meta)
                else:
                    model = None  # Simple average

                self.meta_models_.append(model)
        else:
            # Single meta-learner for all classes (flatten)
            # X_meta: (N * C, n_base), y_meta: (N * C,)
            X_meta = base_stack.reshape(-1, self.n_base_models_)
            y_meta = y_true.flatten()

            if self.normalize_base:
                scaler = StandardScaler()
                X_meta = scaler.fit_transform(X_meta)
                self.scalers_.append(scaler)
            else:
                self.scalers_.append(None)

            if self.meta_learner == "ridge":
                model = RidgeCV(alphas=[0.1, 1.0, 10.0])
                model.fit(X_meta, y_meta)
            elif self.meta_learner == "logistic":
                model = LogisticRegression(max_iter=500, random_state=42)
                model.fit(X_meta, y_meta)
            else:
                model = None

            self.meta_models_.append(model)

        return self

    def predict_proba(self, base_predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combine base model predictions using meta-learner.

        Args:
            base_predictions: List of predictions from base models.
                             Each element: shape (M, C).

        Returns:
            Combined predictions, shape (M, C).
        """
        if not self.meta_models_:
            raise ValueError("Model not fitted. Call fit() first.")

        n_samples = base_predictions[0].shape[0]
        base_stack = np.stack(base_predictions, axis=1)  # (M, n_base, C)

        if self.meta_learner == "avg":
            # Simple average (no learned weights)
            return base_stack.mean(axis=1)

        combined = np.zeros((n_samples, self.n_classes_), dtype=np.float32)

        if self.per_class_meta:
            for c in range(self.n_classes_):
                X_meta = base_stack[:, :, c]  # (M, n_base)

                if self.scalers_[c] is not None:
                    X_meta = self.scalers_[c].transform(X_meta)

                model = self.meta_models_[c]

                if model is None:
                    combined[:, c] = X_meta.mean(axis=1)
                elif hasattr(model, "predict_proba"):
                    combined[:, c] = model.predict_proba(X_meta)[:, 1]
                else:
                    # Ridge: clip to [0, 1]
                    pred = model.predict(X_meta)
                    combined[:, c] = np.clip(pred, 0, 1)
        else:
            X_meta = base_stack.reshape(-1, self.n_base_models_)

            if self.scalers_[0] is not None:
                X_meta = self.scalers_[0].transform(X_meta)

            model = self.meta_models_[0]

            if model is None:
                pred = X_meta.mean(axis=1)
            elif hasattr(model, "predict_proba"):
                pred = model.predict_proba(X_meta)[:, 1]
            else:
                pred = np.clip(model.predict(X_meta), 0, 1)

            combined = pred.reshape(n_samples, self.n_classes_)

        return combined

    def get_weights(self) -> Optional[np.ndarray]:
        """
        Get meta-learner weights (for Ridge).

        Returns:
            Weights array of shape (n_classes, n_base_models) or None.
        """
        if not self.meta_models_ or self.meta_learner not in ["ridge", "logistic"]:
            return None

        if self.per_class_meta:
            weights = []
            for model in self.meta_models_:
                if model is not None and hasattr(model, "coef_"):
                    weights.append(model.coef_.flatten())
                else:
                    weights.append(np.ones(self.n_base_models_) / self.n_base_models_)
            return np.array(weights)
        else:
            if hasattr(self.meta_models_[0], "coef_"):
                return self.meta_models_[0].coef_
            return None

    def save(self, path: str) -> None:
        """Save stacker to disk."""
        import joblib

        data = {
            "config": {
                "meta_learner": self.meta_learner,
                "normalize_base": self.normalize_base,
                "per_class_meta": self.per_class_meta,
                "n_classes": self.n_classes_,
                "n_base_models": self.n_base_models_,
            },
            "scalers": self.scalers_,
            "meta_models": self.meta_models_,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "Stacker":
        """Load stacker from disk."""
        import joblib

        data = joblib.load(path)
        config = data["config"]

        stacker = cls(
            meta_learner=config["meta_learner"],
            normalize_base=config["normalize_base"],
            per_class_meta=config["per_class_meta"],
        )
        stacker.n_classes_ = config["n_classes"]
        stacker.n_base_models_ = config["n_base_models"]
        stacker.scalers_ = data["scalers"]
        stacker.meta_models_ = data["meta_models"]

        return stacker


def generate_oof_predictions(
    model_class,
    model_kwargs: Dict,
    X: np.ndarray,
    y: np.ndarray,
    fold_assignments: List[Tuple[List[str], List[str]]],
    protein_id_to_idx: Dict[str, int],
) -> np.ndarray:
    """
    Generate out-of-fold predictions for stacking.

    Args:
        model_class: Model class to instantiate.
        model_kwargs: Keyword arguments for model init.
        X: Full feature matrix.
        y: Full label matrix.
        fold_assignments: List of (train_ids, val_ids) per fold.
        protein_id_to_idx: Mapping from protein ID to index.

    Returns:
        OOF predictions, shape (N, C).
    """
    n_samples, n_classes = y.shape
    oof_pred = np.zeros_like(y, dtype=np.float32)

    for fold_idx, (train_ids, val_ids) in enumerate(fold_assignments):
        print(f"Fold {fold_idx + 1}/{len(fold_assignments)}")

        train_idx = [
            protein_id_to_idx[pid] for pid in train_ids if pid in protein_id_to_idx
        ]
        val_idx = [
            protein_id_to_idx[pid] for pid in val_ids if pid in protein_id_to_idx
        ]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        # Train model
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)

        # Predict on validation
        val_pred = model.predict_proba(X_val)

        for i, idx in enumerate(val_idx):
            oof_pred[idx] = val_pred[i]

    return oof_pred


class SimpleWeightedAverage:
    """
    Simple weighted average ensemble (no meta-learner training).

    Weights can be set manually or tuned via Optuna.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        """
        Args:
            weights: Optional list of weights per base model.
        """
        self.weights = weights

    def predict_proba(self, base_predictions: List[np.ndarray]) -> np.ndarray:
        """
        Combine predictions via weighted average.

        Args:
            base_predictions: List of predictions from base models.

        Returns:
            Combined predictions.
        """
        if self.weights is None:
            # Uniform weights
            return np.mean(base_predictions, axis=0)

        weights = np.array(self.weights)
        weights = weights / weights.sum()  # Normalize

        combined = np.zeros_like(base_predictions[0])
        for w, pred in zip(weights, base_predictions):
            combined += w * pred

        return combined
