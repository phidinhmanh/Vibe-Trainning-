"""
Tests for CAFA 6 scorer module.
"""

import numpy as np
import pytest

from src.score.scorer import (
    weighted_precision_recall_f1,
    CafaScorer,
    ScoreResult,
    quick_score,
)


class TestWeightedPrecisionRecallF1:
    """Tests for the core scoring function."""

    def test_perfect_predictions(self):
        """Perfect predictions should give F1 = 1.0."""
        y_true = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.float32)

        y_prob = np.array(
            [[0.9, 0.1, 0.9], [0.1, 0.9, 0.9], [0.9, 0.9, 0.1]], dtype=np.float32
        )

        ia_weights = np.ones(3, dtype=np.float32)
        thresholds = np.array([0.5])

        result = weighted_precision_recall_f1(y_true, y_prob, ia_weights, thresholds)

        assert result["f1"] == pytest.approx(1.0, abs=1e-5)
        assert result["wprec"] == pytest.approx(1.0, abs=1e-5)
        assert result["wrec"] == pytest.approx(1.0, abs=1e-5)

    def test_no_predictions(self):
        """High threshold leading to no predictions."""
        y_true = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)

        y_prob = np.array([[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]], dtype=np.float32)

        ia_weights = np.ones(3, dtype=np.float32)
        thresholds = np.array([0.99])  # Very high threshold

        result = weighted_precision_recall_f1(y_true, y_prob, ia_weights, thresholds)

        # No predictions -> precision undefined (0), recall 0
        assert result["wrec"] == pytest.approx(0.0, abs=1e-5)

    def test_threshold_sweep_finds_best(self):
        """Threshold sweep should find the best F1."""
        y_true = np.array(
            [[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32
        )

        y_prob = np.array(
            [[0.8, 0.2, 0.7], [0.3, 0.9, 0.2], [0.6, 0.1, 0.8], [0.2, 0.7, 0.3]],
            dtype=np.float32,
        )

        ia_weights = np.ones(3, dtype=np.float32)
        thresholds = np.linspace(0.1, 0.9, 9)

        result = weighted_precision_recall_f1(y_true, y_prob, ia_weights, thresholds)

        assert result["tau"] is not None
        assert 0.1 <= result["tau"] <= 0.9
        assert result["f1"] >= 0.0

    def test_weighted_by_ia(self):
        """IA weights should affect the final score."""
        y_true = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.float32)

        # Model only predicts class 0 and 1 correctly
        y_prob = np.array([[0.9, 0.9, 0.1], [0.9, 0.9, 0.1]], dtype=np.float32)

        # Give high weight to class 2 (which we miss)
        ia_weights_low_2 = np.array([1.0, 1.0, 0.1], dtype=np.float32)
        ia_weights_high_2 = np.array([1.0, 1.0, 10.0], dtype=np.float32)

        thresholds = np.array([0.5])

        result_low = weighted_precision_recall_f1(
            y_true, y_prob, ia_weights_low_2, thresholds
        )
        result_high = weighted_precision_recall_f1(
            y_true, y_prob, ia_weights_high_2, thresholds
        )

        # Higher weight on missed class should lower score
        assert result_low["f1"] > result_high["f1"]


class TestCafaScorer:
    """Tests for CafaScorer class."""

    def test_score_returns_scoreresult(self):
        """score() should return a ScoreResult dataclass."""
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)

        scorer = CafaScorer(n_thresholds=10)
        result = scorer.score(y_true, y_prob)

        assert isinstance(result, ScoreResult)
        assert hasattr(result, "tau")
        assert hasattr(result, "f1")
        assert hasattr(result, "weighted_precision")
        assert hasattr(result, "weighted_recall")

    def test_score_full_sweep(self):
        """score_full_sweep() should return results for all thresholds."""
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)

        scorer = CafaScorer(n_thresholds=20)
        results = scorer.score_full_sweep(y_true, y_prob)

        assert len(results) == len(scorer.thresholds_)
        assert all(isinstance(r, ScoreResult) for r in results)

    def test_per_term_scores(self):
        """per_term_scores() should return metrics for each term."""
        y_true = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2, 0.7], [0.3, 0.9, 0.6]], dtype=np.float32)

        scorer = CafaScorer()
        results = scorer.per_term_scores(y_true, y_prob, threshold=0.5)

        assert len(results) == 3
        assert all("precision" in r for r in results)
        assert all("recall" in r for r in results)
        assert all("f1" in r for r in results)
        assert all("support" in r for r in results)

    def test_custom_ia_weights(self):
        """Scorer should use custom IA weights."""
        y_true = np.array([[1, 1]], dtype=np.float32)

        # Model predicts class 0 correctly but completely misses class 1 (prob=0)
        y_prob = np.array([[0.9, 0.0]], dtype=np.float32)

        # Weight class 1 heavily (which we completely miss)
        ia_weights = np.array([1.0, 10.0], dtype=np.float32)

        scorer = CafaScorer(ia_weights=ia_weights)
        result = scorer.score(y_true, y_prob)

        # Score should be low because we miss the heavily weighted class
        assert result.f1 < 0.5


class TestQuickScore:
    """Tests for the quick_score convenience function."""

    def test_basic_usage(self):
        """quick_score should work with minimal inputs."""
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32)

        result = quick_score(y_true, y_prob)

        assert isinstance(result, ScoreResult)
        assert 0.0 <= result.f1 <= 1.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_predictions(self):
        """Handle case with no positive predictions."""
        y_true = np.array([[1, 1]], dtype=np.float32)
        y_prob = np.array([[0.0, 0.0]], dtype=np.float32)

        scorer = CafaScorer()
        result = scorer.score(y_true, y_prob)

        # Should not raise, F1 should be 0
        assert result.f1 == 0.0

    def test_all_positive_predictions(self):
        """Handle case where everything is predicted positive."""
        y_true = np.array([[1, 0], [0, 1]], dtype=np.float32)
        y_prob = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        scorer = CafaScorer()
        result = scorer.score(y_true, y_prob)

        # Recall should be 1.0, precision should be ~0.5
        assert result.weighted_recall == pytest.approx(1.0, abs=1e-5)

    def test_single_sample(self):
        """Handle single sample input."""
        y_true = np.array([[1, 0, 1]], dtype=np.float32)
        y_prob = np.array([[0.9, 0.1, 0.8]], dtype=np.float32)

        scorer = CafaScorer()
        result = scorer.score(y_true, y_prob)

        assert 0.0 <= result.f1 <= 1.0

    def test_single_class(self):
        """Handle single class input."""
        y_true = np.array([[1], [0], [1]], dtype=np.float32)
        y_prob = np.array([[0.9], [0.2], [0.7]], dtype=np.float32)

        scorer = CafaScorer()
        result = scorer.score(y_true, y_prob)

        assert 0.0 <= result.f1 <= 1.0
