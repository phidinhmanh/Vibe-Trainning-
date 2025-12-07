"""
CAFA scoring module.

Implements weighted precision/recall/F1 computation with Information Accretion (IA)
weights and threshold sweeping to find F-Max.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm


@dataclass
class ScoreResult:
    """Result of scoring at a specific threshold."""

    tau: float
    f1: float
    weighted_precision: float
    weighted_recall: float

    def __repr__(self) -> str:
        return (
            f"ScoreResult(tau={self.tau:.4f}, f1={self.f1:.4f}, "
            f"wprec={self.weighted_precision:.4f}, wrec={self.weighted_recall:.4f})"
        )


def weighted_precision_recall_f1(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ia_weights: np.ndarray,
    thresholds: np.ndarray,
) -> Dict[str, Union[float, None]]:
    """
    Compute weighted precision, recall, and F1 across multiple thresholds.

    Returns the best threshold configuration (F-Max).

    Args:
        y_true: Binary ground truth labels, shape (N, C).
        y_prob: Predicted probabilities, shape (N, C).
        ia_weights: Information Accretion weights per class, shape (C,).
        thresholds: Array of thresholds to sweep.

    Returns:
        Dictionary with best tau, f1, weighted precision/recall.
    """
    best = {"tau": None, "f1": -1.0, "wprec": 0.0, "wrec": 0.0}

    for tau in thresholds:
        # Binarize predictions
        pred = (y_prob >= tau).astype(np.float32)

        # Per-class TP, FP, FN
        tp = (pred * y_true).sum(axis=0).astype(np.float64)
        fp = (pred * (1 - y_true)).sum(axis=0).astype(np.float64)
        fn = ((1 - pred) * y_true).sum(axis=0).astype(np.float64)

        # Per-class precision and recall (avoid division by zero)
        prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

        # Weighted averages by IA weights
        weight_sum = np.sum(ia_weights)
        wprec = np.sum(ia_weights * prec) / weight_sum
        wrec = np.sum(ia_weights * rec) / weight_sum

        # Weighted F1
        if wprec + wrec == 0:
            wf1 = 0.0
        else:
            wf1 = 2 * wprec * wrec / (wprec + wrec)

        if wf1 > best["f1"]:
            best = {"tau": tau, "f1": wf1, "wprec": wprec, "wrec": wrec}

    return best


class CafaScorer:
    """
    CAFA-style scorer with threshold optimization.

    Supports:
    - Global threshold sweep for F-Max
    - Per-term threshold optimization (for top-K terms)
    - Detailed per-term performance breakdown
    """

    def __init__(
        self,
        ia_weights: Optional[np.ndarray] = None,
        n_thresholds: int = 100,
        threshold_range: Tuple[float, float] = (0.01, 0.99),
        use_log_scale: bool = True,
    ):
        """
        Args:
            ia_weights: Information Accretion weights per term.
            n_thresholds: Number of thresholds to try.
            threshold_range: (min, max) threshold values.
            use_log_scale: Use log-scale sampling for thresholds.
        """
        self.ia_weights = ia_weights
        self.n_thresholds = n_thresholds
        self.threshold_range = threshold_range
        self.use_log_scale = use_log_scale

        self.thresholds_ = self._generate_thresholds()

    def _generate_thresholds(self) -> np.ndarray:
        """Generate threshold values for sweep."""
        min_t, max_t = self.threshold_range

        if self.use_log_scale:
            # Log-scale for better coverage of low thresholds
            log_thresholds = np.geomspace(min_t, max_t, self.n_thresholds // 2)
            # Add linear grid near middle
            lin_thresholds = np.linspace(0.1, 0.9, self.n_thresholds // 2)
            thresholds = np.unique(np.concatenate([log_thresholds, lin_thresholds]))
        else:
            thresholds = np.linspace(min_t, max_t, self.n_thresholds)

        return np.sort(thresholds)

    def score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ia_weights: Optional[np.ndarray] = None,
    ) -> ScoreResult:
        """
        Compute F-Max by sweeping thresholds.

        Args:
            y_true: Binary ground truth, shape (N, C).
            y_prob: Predicted probabilities, shape (N, C).
            ia_weights: Optional override for IA weights.

        Returns:
            ScoreResult with best threshold and metrics.
        """
        weights = ia_weights if ia_weights is not None else self.ia_weights

        if weights is None:
            # Default: uniform weights
            weights = np.ones(y_true.shape[1], dtype=np.float32)

        result = weighted_precision_recall_f1(y_true, y_prob, weights, self.thresholds_)

        return ScoreResult(
            tau=result["tau"],
            f1=result["f1"],
            weighted_precision=result["wprec"],
            weighted_recall=result["wrec"],
        )

    def score_full_sweep(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        ia_weights: Optional[np.ndarray] = None,
    ) -> List[ScoreResult]:
        """
        Return scores for all thresholds (useful for plotting PR curve).

        Args:
            y_true: Binary ground truth, shape (N, C).
            y_prob: Predicted probabilities, shape (N, C).
            ia_weights: Optional override for IA weights.

        Returns:
            List of ScoreResult for each threshold.
        """
        weights = ia_weights if ia_weights is not None else self.ia_weights

        if weights is None:
            weights = np.ones(y_true.shape[1], dtype=np.float32)

        results = []
        for tau in self.thresholds_:
            pred = (y_prob >= tau).astype(np.float32)

            tp = (pred * y_true).sum(axis=0).astype(np.float64)
            fp = (pred * (1 - y_true)).sum(axis=0).astype(np.float64)
            fn = ((1 - pred) * y_true).sum(axis=0).astype(np.float64)

            prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
            rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)

            weight_sum = np.sum(weights)
            wprec = np.sum(weights * prec) / weight_sum
            wrec = np.sum(weights * rec) / weight_sum

            wf1 = 2 * wprec * wrec / (wprec + wrec) if (wprec + wrec) > 0 else 0.0

            results.append(ScoreResult(tau, wf1, wprec, wrec))

        return results

    def per_term_scores(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
        term_names: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Compute per-term precision, recall, F1 at a specific threshold.

        Args:
            y_true: Binary ground truth, shape (N, C).
            y_prob: Predicted probabilities, shape (N, C).
            threshold: Decision threshold.
            term_names: Optional list of term names.

        Returns:
            List of dicts with term-level metrics.
        """
        pred = (y_prob >= threshold).astype(np.float32)
        n_terms = y_true.shape[1]

        results = []
        for i in range(n_terms):
            tp = float((pred[:, i] * y_true[:, i]).sum())
            fp = float((pred[:, i] * (1 - y_true[:, i])).sum())
            fn = float(((1 - pred[:, i]) * y_true[:, i]).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            result = {
                "term_idx": i,
                "term": term_names[i] if term_names else f"term_{i}",
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": int(y_true[:, i].sum()),
                "predictions": int(pred[:, i].sum()),
            }
            results.append(result)

        return results


def generate_threshold_grid(
    n_points: int = 100,
    min_val: float = 0.01,
    max_val: float = 0.99,
    log_scale: bool = True,
) -> np.ndarray:
    """
    Generate threshold grid for sweep.

    Args:
        n_points: Number of threshold points.
        min_val: Minimum threshold.
        max_val: Maximum threshold.
        log_scale: Use logarithmic sampling.

    Returns:
        Sorted array of threshold values.
    """
    if log_scale:
        # Combine log-scale (for low values) and linear (for mid-range)
        log_part = np.geomspace(min_val, 0.5, n_points // 2)
        lin_part = np.linspace(0.1, max_val, n_points // 2)
        thresholds = np.unique(np.concatenate([log_part, lin_part]))
    else:
        thresholds = np.linspace(min_val, max_val, n_points)

    return np.sort(thresholds)


# Convenience function for quick testing
def quick_score(
    y_true: np.ndarray, y_prob: np.ndarray, n_thresholds: int = 50
) -> ScoreResult:
    """
    Quick scoring with uniform weights.

    Args:
        y_true: Binary ground truth.
        y_prob: Predicted probabilities.
        n_thresholds: Number of thresholds.

    Returns:
        Best ScoreResult.
    """
    scorer = CafaScorer(n_thresholds=n_thresholds)
    return scorer.score(y_true, y_prob)
