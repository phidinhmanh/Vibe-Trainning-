"""
K-Nearest Neighbors label transfer for CAFA 6.

Implements:
- Feature-based KNN (physicochemical features)
- Embedding-based KNN (ESM/ProtBERT embeddings with FAISS)
- BLAST-like label transfer (optional)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class KNNTransfer:
    """
    K-Nearest Neighbors label transfer.

    Given a query protein, finds K most similar proteins and aggregates
    their labels using distance-weighted voting.
    """

    def __init__(
        self,
        n_neighbors: int = 7,
        metric: str = "cosine",
        weights: str = "distance",
        n_jobs: int = -1,
    ):
        """
        Args:
            n_neighbors: Number of neighbors to use.
            metric: Distance metric ('cosine', 'euclidean', 'manhattan').
            weights: Weighting scheme ('uniform', 'distance').
            n_jobs: Number of parallel jobs.
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.n_jobs = n_jobs

        self.nn_model_: Optional[NearestNeighbors] = None
        self.train_labels_: Optional[np.ndarray] = None
        self.train_ids_: Optional[List[str]] = None

    def fit(
        self, X: np.ndarray, y: np.ndarray, protein_ids: Optional[List[str]] = None
    ) -> "KNNTransfer":
        """
        Fit the KNN model on training data.

        Args:
            X: Training features, shape (N, D).
            y: Training labels, shape (N, C).
            protein_ids: Optional list of protein IDs.

        Returns:
            self
        """
        self.nn_model_ = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self.metric, n_jobs=self.n_jobs
        )
        self.nn_model_.fit(X)

        self.train_labels_ = y
        self.train_ids_ = protein_ids

        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """
        Predict label probabilities using neighbor voting.

        Args:
            X: Query features, shape (M, D).
            batch_size: Batch size for processing.

        Returns:
            Predicted probabilities, shape (M, C).
        """
        if self.nn_model_ is None or self.train_labels_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_queries = X.shape[0]
        n_classes = self.train_labels_.shape[1]
        predictions = np.zeros((n_queries, n_classes), dtype=np.float32)

        for start_idx in tqdm(range(0, n_queries, batch_size), desc="KNN prediction"):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_X = X[start_idx:end_idx]

            distances, indices = self.nn_model_.kneighbors(batch_X)

            if self.weights == "distance":
                # Inverse distance weighting (avoid div by zero)
                weights = 1.0 / (distances + 1e-8)
                weights /= weights.sum(axis=1, keepdims=True)
            else:
                # Uniform weights
                weights = np.ones_like(distances) / self.n_neighbors

            # Weighted label aggregation
            batch_pred = np.zeros((end_idx - start_idx, n_classes), dtype=np.float32)

            for i in range(end_idx - start_idx):
                neighbor_labels = self.train_labels_[indices[i]]  # (K, C)
                neighbor_weights = weights[i]  # (K,)

                # Weighted sum of neighbor labels
                batch_pred[i] = (neighbor_weights[:, None] * neighbor_labels).sum(
                    axis=0
                )

            predictions[start_idx:end_idx] = batch_pred

        return predictions

    def get_neighbors(
        self, X: np.ndarray, return_distances: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get neighbor indices (and optionally distances).

        Args:
            X: Query features.
            return_distances: Whether to return distances.

        Returns:
            Tuple of (indices, distances) or just indices.
        """
        if self.nn_model_ is None:
            raise ValueError("Model not fitted.")

        if return_distances:
            distances, indices = self.nn_model_.kneighbors(X)
            return indices, distances
        else:
            return self.nn_model_.kneighbors(X, return_distance=False), None

    def save(self, path: str) -> None:
        """Save model to disk."""
        import joblib

        data = {
            "config": {
                "n_neighbors": self.n_neighbors,
                "metric": self.metric,
                "weights": self.weights,
            },
            "nn_model": self.nn_model_,
            "train_labels": self.train_labels_,
            "train_ids": self.train_ids_,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "KNNTransfer":
        """Load model from disk."""
        import joblib

        data = joblib.load(path)
        config = data["config"]

        model = cls(
            n_neighbors=config["n_neighbors"],
            metric=config["metric"],
            weights=config["weights"],
        )
        model.nn_model_ = data["nn_model"]
        model.train_labels_ = data["train_labels"]
        model.train_ids_ = data["train_ids"]

        return model


class FAISSKNNTransfer:
    """
    FAISS-accelerated KNN for large-scale embedding search.

    Recommended for embedding-based (ESM/ProtBERT) transfer.
    """

    def __init__(self, n_neighbors: int = 10, use_gpu: bool = False):
        """
        Args:
            n_neighbors: Number of neighbors.
            use_gpu: Whether to use GPU acceleration.
        """
        self.n_neighbors = n_neighbors
        self.use_gpu = use_gpu

        self.index_ = None
        self.train_labels_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FAISSKNNTransfer":
        """
        Build FAISS index on training embeddings.

        Args:
            X: Training embeddings, shape (N, D).
            y: Training labels, shape (N, C).

        Returns:
            self
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("FAISS not installed. Run: pip install faiss-cpu")

        d = X.shape[1]

        # Normalize for cosine similarity
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        X_norm = X_norm.astype(np.float32)

        # Build index
        self.index_ = faiss.IndexFlatIP(d)  # Inner product = cosine for normalized

        if self.use_gpu:
            res = faiss.StandardGpuResources()
            self.index_ = faiss.index_cpu_to_gpu(res, 0, self.index_)

        self.index_.add(X_norm)
        self.train_labels_ = y

        return self

    def predict_proba(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """
        Predict using FAISS search.

        Args:
            X: Query embeddings, shape (M, D).
            batch_size: Batch size.

        Returns:
            Predicted probabilities, shape (M, C).
        """
        if self.index_ is None or self.train_labels_ is None:
            raise ValueError("Model not fitted.")

        # Normalize queries
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        X_norm = X_norm.astype(np.float32)

        n_queries = X.shape[0]
        n_classes = self.train_labels_.shape[1]
        predictions = np.zeros((n_queries, n_classes), dtype=np.float32)

        for start_idx in tqdm(range(0, n_queries, batch_size), desc="FAISS search"):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_X = X_norm[start_idx:end_idx]

            # Search
            similarities, indices = self.index_.search(batch_X, self.n_neighbors)

            # Convert similarities to weights (softmax-like)
            weights = np.exp(similarities - similarities.max(axis=1, keepdims=True))
            weights /= weights.sum(axis=1, keepdims=True)

            # Weighted aggregation
            for i in range(end_idx - start_idx):
                neighbor_labels = self.train_labels_[indices[i]]
                neighbor_weights = weights[i]
                predictions[start_idx + i] = (
                    neighbor_weights[:, None] * neighbor_labels
                ).sum(axis=0)

        return predictions


def blast_label_transfer(
    blast_hits: Dict[str, List[Tuple[str, float]]],
    label_matrix: np.ndarray,
    protein_id_to_idx: Dict[str, int],
    top_k: int = 10,
    identity_weight_power: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Transfer labels from BLAST hits.

    Args:
        blast_hits: Dict mapping query -> list of (hit_id, identity) tuples.
        label_matrix: Training label matrix, shape (N, C).
        protein_id_to_idx: Mapping from protein ID to label matrix index.
        top_k: Maximum hits to consider per query.
        identity_weight_power: Power to raise identity weights to.

    Returns:
        Dict mapping query ID to predicted probability vector.
    """
    predictions = {}
    n_classes = label_matrix.shape[1]

    for query_id, hits in tqdm(blast_hits.items(), desc="BLAST transfer"):
        # Take top K hits
        hits = hits[:top_k]

        if not hits:
            predictions[query_id] = np.zeros(n_classes, dtype=np.float32)
            continue

        # Weight by identity
        aggregated = np.zeros(n_classes, dtype=np.float32)
        weight_sum = 0.0

        for hit_id, identity in hits:
            if hit_id not in protein_id_to_idx:
                continue

            idx = protein_id_to_idx[hit_id]
            weight = identity**identity_weight_power

            aggregated += weight * label_matrix[idx]
            weight_sum += weight

        if weight_sum > 0:
            aggregated /= weight_sum

        predictions[query_id] = aggregated

    return predictions
