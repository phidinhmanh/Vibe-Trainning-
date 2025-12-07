"""
GO hierarchy enforcement for CAFA 6.

Implements True-Path Rule for GO term predictions:
- If a child term is predicted, all parent terms must also be predicted.
- Propagation is done AFTER calibration but BEFORE thresholding.
"""

import json
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm


class GOHierarchy:
    """
    Gene Ontology hierarchy manager.

    Handles parent-child relationships and score propagation.
    """

    def __init__(self, obo_path: Optional[str] = None):
        """
        Args:
            obo_path: Path to GO .obo file (optional, can load from JSON).
        """
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.topo_order: List[str] = []

        if obo_path:
            self._load_obo(obo_path)

    def _load_obo(self, obo_path: str) -> None:
        """Parse GO .obo file to extract is_a relationships."""
        current_term = None

        with open(obo_path, "r") as f:
            for line in f:
                line = line.strip()

                if line == "[Term]":
                    current_term = None
                elif line.startswith("id: GO:"):
                    current_term = line[4:]  # Extract GO:XXXXXX
                elif line.startswith("is_a: GO:") and current_term:
                    parent = line.split()[1]  # GO:XXXXXX
                    self.parents[current_term].add(parent)
                    self.children[parent].add(current_term)

        self._compute_topo_order()

    def _compute_topo_order(self) -> None:
        """Compute topological order (roots first, leaves last)."""
        # Find all terms
        all_terms = set(self.parents.keys()) | set(self.children.keys())

        # Kahn's algorithm
        in_degree = defaultdict(int)
        for term in all_terms:
            in_degree[term] = len(self.parents.get(term, set()))

        # Start with roots (no parents)
        queue = [t for t in all_terms if in_degree[t] == 0]
        self.topo_order = []

        while queue:
            term = queue.pop(0)
            self.topo_order.append(term)

            for child in self.children.get(term, set()):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    def get_ancestors(self, term: str) -> Set[str]:
        """Get all ancestors of a term (recursive)."""
        ancestors = set()
        to_visit = list(self.parents.get(term, set()))

        while to_visit:
            parent = to_visit.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                to_visit.extend(self.parents.get(parent, set()))

        return ancestors

    def get_descendants(self, term: str) -> Set[str]:
        """Get all descendants of a term (recursive)."""
        descendants = set()
        to_visit = list(self.children.get(term, set()))

        while to_visit:
            child = to_visit.pop()
            if child not in descendants:
                descendants.add(child)
                to_visit.extend(self.children.get(child, set()))

        return descendants

    def propagate_max(
        self, scores: np.ndarray, term_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """
        Propagate scores upward using max rule.

        For each parent term: score = max(own_score, max(children_scores))

        Args:
            scores: Score matrix, shape (N, C).
            term_to_idx: Mapping from GO term to column index.

        Returns:
            Propagated score matrix.
        """
        idx_to_term = {v: k for k, v in term_to_idx.items()}
        propagated = scores.copy()

        # Process in reverse topological order (leaves to roots)
        for term in reversed(self.topo_order):
            if term not in term_to_idx:
                continue

            term_idx = term_to_idx[term]

            # Get children scores
            child_indices = [
                term_to_idx[child]
                for child in self.children.get(term, set())
                if child in term_to_idx
            ]

            if child_indices:
                max_child_score = propagated[:, child_indices].max(axis=1)
                propagated[:, term_idx] = np.maximum(
                    propagated[:, term_idx], max_child_score
                )

        return propagated

    def propagate_avg(
        self, scores: np.ndarray, term_to_idx: Dict[str, int], alpha: float = 0.5
    ) -> np.ndarray:
        """
        Propagate scores upward using weighted average.

        parent_score = alpha * own_score + (1-alpha) * mean(children_scores)

        Args:
            scores: Score matrix, shape (N, C).
            term_to_idx: Mapping from GO term to column index.
            alpha: Weight for own score vs children.

        Returns:
            Propagated score matrix.
        """
        propagated = scores.copy()

        for term in reversed(self.topo_order):
            if term not in term_to_idx:
                continue

            term_idx = term_to_idx[term]

            child_indices = [
                term_to_idx[child]
                for child in self.children.get(term, set())
                if child in term_to_idx
            ]

            if child_indices:
                avg_child_score = propagated[:, child_indices].mean(axis=1)
                propagated[:, term_idx] = alpha * propagated[:, term_idx] + (
                    1 - alpha
                ) * np.maximum(propagated[:, term_idx], avg_child_score)

        return propagated

    def enforce_true_path(
        self, predictions: np.ndarray, term_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """
        Enforce True-Path Rule on binary predictions.

        If any child is predicted, all ancestors must be predicted.

        Args:
            predictions: Binary prediction matrix, shape (N, C).
            term_to_idx: Mapping from GO term to column index.

        Returns:
            Corrected binary predictions.
        """
        corrected = predictions.copy()

        for term, idx in term_to_idx.items():
            # Find samples where this term is predicted
            predicted_mask = corrected[:, idx] > 0

            if not predicted_mask.any():
                continue

            # Ensure all ancestors are also predicted
            for ancestor in self.get_ancestors(term):
                if ancestor in term_to_idx:
                    anc_idx = term_to_idx[ancestor]
                    corrected[predicted_mask, anc_idx] = 1

        return corrected

    def save(self, path: str) -> None:
        """Save hierarchy to JSON."""
        data = {
            "parents": {k: list(v) for k, v in self.parents.items()},
            "children": {k: list(v) for k, v in self.children.items()},
            "topo_order": self.topo_order,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GOHierarchy":
        """Load hierarchy from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        hierarchy = cls()
        hierarchy.parents = {k: set(v) for k, v in data["parents"].items()}
        hierarchy.children = {k: set(v) for k, v in data["children"].items()}
        hierarchy.topo_order = data["topo_order"]

        return hierarchy


def apply_hierarchy_propagation(
    scores: np.ndarray,
    term_to_idx: Dict[str, int],
    hierarchy: GOHierarchy,
    method: str = "max",
) -> np.ndarray:
    """
    Apply hierarchy-aware score propagation.

    This should be called AFTER calibration but BEFORE thresholding.

    Args:
        scores: Calibrated probability matrix, shape (N, C).
        term_to_idx: GO term to index mapping.
        hierarchy: GOHierarchy instance.
        method: 'max' or 'avg'.

    Returns:
        Propagated scores.
    """
    if method == "max":
        return hierarchy.propagate_max(scores, term_to_idx)
    elif method == "avg":
        return hierarchy.propagate_avg(scores, term_to_idx)
    else:
        raise ValueError(f"Unknown propagation method: {method}")
