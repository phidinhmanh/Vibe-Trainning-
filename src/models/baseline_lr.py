"""
Baseline Logistic Regression model for CAFA 6.

Uses TF-IDF k-mer features with OneVsRest logistic regression.
Supports calibration and hierarchical propagation.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier


class BaselineLR:
    """
    Baseline Logistic Regression classifier for multi-label protein function prediction.

    Features:
    - OneVsRest wrapper for multi-label
    - Probability calibration (Platt/isotonic)
    - Model persistence
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        n_jobs: int = -1,
        calibrate: bool = True,
        calibration_method: str = "isotonic",
        random_state: int = 42,
    ):
        """
        Args:
            C: Inverse regularization strength.
            max_iter: Maximum iterations for solver.
            solver: Optimization solver ('lbfgs', 'saga', 'liblinear').
            n_jobs: Number of parallel jobs (-1 for all cores).
            calibrate: Whether to calibrate probabilities.
            calibration_method: 'sigmoid' (Platt) or 'isotonic'.
            random_state: Random seed.
        """
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.n_jobs = n_jobs
        self.calibrate = calibrate
        self.calibration_method = calibration_method
        self.random_state = random_state

        self.model_: Optional[OneVsRestClassifier] = None
        self.calibrator_: Optional[CalibratedClassifierCV] = None
        self.n_classes_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ) -> "BaselineLR":
        """
        Fit the model.

        Args:
            X: Training features, shape (N, D).
            y: Training labels, shape (N, C).
            X_cal: Optional calibration set features.
            y_cal: Optional calibration set labels.

        Returns:
            self
        """
        self.n_classes_ = y.shape[1]

        # Base classifier
        base_clf = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
            n_jobs=1,  # Parallelism handled by OneVsRest
        )

        self.model_ = OneVsRestClassifier(base_clf, n_jobs=self.n_jobs)
        self.model_.fit(X, y)

        # Calibration
        if self.calibrate and X_cal is not None and y_cal is not None:
            self.calibrator_ = CalibratedClassifierCV(
                self.model_, method=self.calibration_method, cv="prefit"
            )
            self.calibrator_.fit(X_cal, y_cal)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Input features, shape (N, D).

        Returns:
            Probabilities, shape (N, C).
        """
        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if self.calibrator_ is not None:
            return self.calibrator_.predict_proba(X)

        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary labels.

        Args:
            X: Input features, shape (N, D).
            threshold: Decision threshold.

        Returns:
            Binary predictions, shape (N, C).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(np.int32)

    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib

        data = {
            "config": {
                "C": self.C,
                "max_iter": self.max_iter,
                "solver": self.solver,
                "calibrate": self.calibrate,
                "calibration_method": self.calibration_method,
                "random_state": self.random_state,
                "n_classes": self.n_classes_,
            },
            "model": self.model_,
            "calibrator": self.calibrator_,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "BaselineLR":
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        config = data["config"]

        model = cls(
            C=config["C"],
            max_iter=config["max_iter"],
            solver=config["solver"],
            calibrate=config["calibrate"],
            calibration_method=config["calibration_method"],
            random_state=config["random_state"],
        )
        model.model_ = data["model"]
        model.calibrator_ = data["calibrator"]
        model.n_classes_ = config["n_classes"]

        return model


def train_baseline_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float = 1.0,
    calibrate: bool = True,
) -> Tuple[BaselineLR, Dict]:
    """
    Train baseline LR model and evaluate.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        C: Regularization parameter.
        calibrate: Whether to calibrate.

    Returns:
        Tuple of (trained model, evaluation metrics).
    """
    from ..score.scorer import CafaScorer

    model = BaselineLR(C=C, calibrate=calibrate)

    if calibrate:
        # Split training for calibration
        n_train = int(0.8 * len(X_train))
        model.fit(
            X_train[:n_train], y_train[:n_train], X_train[n_train:], y_train[n_train:]
        )
    else:
        model.fit(X_train, y_train)

    # Evaluate
    proba = model.predict_proba(X_val)

    scorer = CafaScorer()
    result = scorer.score(y_val, proba)

    metrics = {
        "f_max": result.f1,
        "best_threshold": result.tau,
        "weighted_precision": result.weighted_precision,
        "weighted_recall": result.weighted_recall,
    }

    return model, metrics
