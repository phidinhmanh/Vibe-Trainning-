"""
Threshold optimization strategies for CAFA.

Includes global threshold, per-label thresholds, and learned threshold calibration.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    weights: Optional[np.ndarray] = None,
    n_thresholds: int = 100,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Find global optimal threshold for maximum weighted F1.

    Args:
        y_true: Binary ground truth, shape (N, C).
        y_prob: Predicted probabilities, shape (N, C).
        weights: Optional per-term weights (default: uniform).
        n_thresholds: Number of thresholds to try.
        metric: Optimization metric ('f1', 'precision', 'recall').

    Returns:
        Tuple of (best_threshold, best_score).
    """
    if weights is None:
        weights = np.ones(y_true.shape[1], dtype=np.float32)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_tau = 0.5
    best_score = 0.0

    for tau in thresholds:
        pred = (y_prob >= tau).astype(np.float32)

        tp = (pred * y_true).sum(axis=0).astype(np.float64)
        fp = (pred * (1 - y_true)).sum(axis=0).astype(np.float64)
        fn = ((1 - pred) * y_true).sum(axis=0).astype(np.float64)

        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

        weight_sum = np.sum(weights)
        wprec = np.sum(weights * prec) / weight_sum
        wrec = np.sum(weights * rec) / weight_sum

        if metric == "f1":
            score = 2 * wprec * wrec / (wprec + wrec) if (wprec + wrec) > 0 else 0.0
        elif metric == "precision":
            score = wprec
        elif metric == "recall":
            score = wrec
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_tau = tau

    return best_tau, best_score


class ThresholdOptimizer:
    """
    Advanced threshold optimization with per-label thresholds.

    Strategies:
    - Global: single threshold for all terms
    - Per-label: optimize threshold per term (for top-K terms)
    - Hybrid: global + per-label adjustments
    """

    def __init__(
        self,
        strategy: str = "global",
        n_thresholds: int = 50,
        top_k_per_label: int = 100,
        min_support: int = 10,
    ):
        """
        Args:
            strategy: 'global', 'per_label', or 'hybrid'.
            n_thresholds: Number of thresholds per sweep.
            top_k_per_label: Optimize thresholds for top K frequent labels.
            min_support: Minimum positive samples for per-label optimization.
        """
        self.strategy = strategy
        self.n_thresholds = n_thresholds
        self.top_k_per_label = top_k_per_label
        self.min_support = min_support

        self.global_threshold_: float = 0.5
        self.per_label_thresholds_: Optional[np.ndarray] = None

    def fit(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "ThresholdOptimizer":
        """
        Optimize thresholds on validation data.

        Args:
            y_true: Binary ground truth, shape (N, C).
            y_prob: Predicted probabilities, shape (N, C).
            weights: Optional per-term weights.

        Returns:
            self
        """
        n_samples, n_terms = y_true.shape

        if weights is None:
            weights = np.ones(n_terms, dtype=np.float32)

        # Always compute global threshold
        self.global_threshold_, _ = find_optimal_threshold(
            y_true, y_prob, weights, self.n_thresholds
        )

        # Per-label thresholds
        if self.strategy in ["per_label", "hybrid"]:
            self.per_label_thresholds_ = self._optimize_per_label(
                y_true, y_prob, weights
            )

        return self

    def _optimize_per_label(
        self, y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Optimize threshold for each label independently."""
        n_terms = y_true.shape[1]
        thresholds = np.full(n_terms, self.global_threshold_)

        # Get term support (number of positives)
        term_support = y_true.sum(axis=0)

        # Sort by support, optimize top-K
        top_terms = np.argsort(-term_support)[: self.top_k_per_label]

        threshold_candidates = np.linspace(0.01, 0.99, self.n_thresholds)

        for term_idx in tqdm(top_terms, desc="Optimizing per-label thresholds"):
            if term_support[term_idx] < self.min_support:
                continue

            y_term = y_true[:, term_idx]
            p_term = y_prob[:, term_idx]

            best_tau = self.global_threshold_
            best_f1 = 0.0

            for tau in threshold_candidates:
                pred = (p_term >= tau).astype(np.float32)

                tp = (pred * y_term).sum()
                fp = (pred * (1 - y_term)).sum()
                fn = ((1 - pred) * y_term).sum()

                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = tau

            thresholds[term_idx] = best_tau

        return thresholds

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Apply optimized thresholds to get binary predictions.

        Args:
            y_prob: Predicted probabilities, shape (N, C).

        Returns:
            Binary predictions, shape (N, C).
        """
        if self.strategy == "global" or self.per_label_thresholds_ is None:
            return (y_prob >= self.global_threshold_).astype(np.float32)

        # Per-label thresholds
        return (y_prob >= self.per_label_thresholds_[None, :]).astype(np.float32)

    def get_thresholds(self) -> Dict[str, np.ndarray]:
        """Get all optimized thresholds."""
        result = {"global": self.global_threshold_}

        if self.per_label_thresholds_ is not None:
            result["per_label"] = self.per_label_thresholds_

        return result

    def save(self, path: str) -> None:
        """Save thresholds to file."""
        import json

        data = {
            "strategy": self.strategy,
            "global_threshold": float(self.global_threshold_),
        }

        if self.per_label_thresholds_ is not None:
            data["per_label_thresholds"] = self.per_label_thresholds_.tolist()

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ThresholdOptimizer":
        """Load thresholds from file."""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        optimizer = cls(strategy=data["strategy"])
        optimizer.global_threshold_ = data["global_threshold"]

        if "per_label_thresholds" in data:
            optimizer.per_label_thresholds_ = np.array(
                data["per_label_thresholds"], dtype=np.float32
            )

        return optimizer


def ensemble_threshold_search(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
    weights: Optional[np.ndarray] = None,
    ensemble_weights: Optional[List[float]] = None,
    n_thresholds: int = 50,
) -> Tuple[float, List[float], float]:
    """
    Joint optimization of ensemble weights and threshold.

    Args:
        predictions: List of probability matrices from different models.
        y_true: Binary ground truth.
        weights: Per-term IA weights.
        ensemble_weights: Initial ensemble weights (default: uniform).
        n_thresholds: Number of threshold candidates.

    Returns:
        Tuple of (best_threshold, best_ensemble_weights, best_score).
    """
    n_models = len(predictions)

    if ensemble_weights is None:
        ensemble_weights = [1.0 / n_models] * n_models

    if weights is None:
        weights = np.ones(y_true.shape[1], dtype=np.float32)

    # Simple grid search (for production, use Optuna)
    best_threshold = 0.5
    best_weights = ensemble_weights
    best_score = 0.0

    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    # For now, just optimize threshold with fixed weights
    combined = sum(w * p for w, p in zip(ensemble_weights, predictions))

    for tau in thresholds:
        pred = (combined >= tau).astype(np.float32)

        tp = (pred * y_true).sum(axis=0).astype(np.float64)
        fp = (pred * (1 - y_true)).sum(axis=0).astype(np.float64)
        fn = ((1 - pred) * y_true).sum(axis=0).astype(np.float64)

        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

        weight_sum = np.sum(weights)
        wprec = np.sum(weights * prec) / weight_sum
        wrec = np.sum(weights * rec) / weight_sum

        wf1 = 2 * wprec * wrec / (wprec + wrec) if (wprec + wrec) > 0 else 0.0

        if wf1 > best_score:
            best_score = wf1
            best_threshold = tau

    return best_threshold, best_weights, best_score
