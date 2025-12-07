"""
TF-IDF k-mer feature extraction for protein sequences.

Converts sequences to sparse TF-IDF vectors based on k-mer (substring) frequencies.
"""

from typing import List, Optional, Union

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfKmerFeaturizer:
    """
    Convert protein sequences to TF-IDF k-mer feature vectors.

    Treats each sequence as a document and each k-mer as a word.
    """

    def __init__(
        self,
        k: int = 3,
        max_features: int = 200000,
        ngram_range: Optional[tuple] = None,
        min_df: int = 2,
        max_df: float = 0.95,
        sublinear_tf: bool = True,
    ):
        """
        Args:
            k: K-mer size.
            max_features: Maximum number of features (vocabulary size).
            ngram_range: Optional (min_n, max_n) for character n-grams.
                        If None, uses (k, k).
            min_df: Minimum document frequency for a term.
            max_df: Maximum document frequency (ignore too common terms).
            sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).
        """
        self.k = k
        self.max_features = max_features
        self.ngram_range = ngram_range or (k, k)
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        self.vectorizer_ = None

    def _sequence_to_kmers(self, seq: str) -> str:
        """Convert sequence to space-separated k-mers."""
        kmers = []
        for i in range(len(seq) - self.k + 1):
            kmers.append(seq[i : i + self.k])
        return " ".join(kmers)

    def fit(self, sequences: List[str]) -> "TfidfKmerFeaturizer":
        """
        Fit the TF-IDF vectorizer on sequences.

        Args:
            sequences: List of protein sequences.

        Returns:
            self
        """
        # Option 1: Use analyzer='char' with ngram_range (sklearn built-in)
        # Option 2: Pre-tokenize k-mers and use analyzer='word'
        # Using Option 1 for efficiency with large datasets

        self.vectorizer_ = TfidfVectorizer(
            analyzer="char",
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=False,  # Amino acids are case-sensitive
        )

        self.vectorizer_.fit(sequences)
        return self

    def transform(self, sequences: List[str]) -> sp.csr_matrix:
        """
        Transform sequences to TF-IDF feature matrix.

        Args:
            sequences: List of protein sequences.

        Returns:
            Sparse matrix of shape (n_sequences, n_features).
        """
        if self.vectorizer_ is None:
            raise ValueError("Call fit() first.")

        return self.vectorizer_.transform(sequences)

    def fit_transform(self, sequences: List[str]) -> sp.csr_matrix:
        """Fit and transform in one step."""
        self.fit(sequences)
        return self.transform(sequences)

    @property
    def n_features(self) -> int:
        """Get number of features after fitting."""
        if self.vectorizer_ is None:
            return 0
        return len(self.vectorizer_.vocabulary_)

    @property
    def feature_names(self) -> List[str]:
        """Get feature (k-mer) names."""
        if self.vectorizer_ is None:
            return []
        return self.vectorizer_.get_feature_names_out().tolist()

    def save(self, path: str) -> None:
        """Save fitted vectorizer to disk."""
        import joblib

        joblib.dump(self.vectorizer_, path)

    def load(self, path: str) -> "TfidfKmerFeaturizer":
        """Load fitted vectorizer from disk."""
        import joblib

        self.vectorizer_ = joblib.load(path)
        return self


def extract_kmer_features(
    sequences: List[str], k: int = 3, max_features: int = 200000
) -> sp.csr_matrix:
    """
    Quick helper to extract TF-IDF k-mer features.

    Args:
        sequences: List of protein sequences.
        k: K-mer size.
        max_features: Maximum vocabulary size.

    Returns:
        Sparse TF-IDF matrix.
    """
    featurizer = TfidfKmerFeaturizer(k=k, max_features=max_features)
    return featurizer.fit_transform(sequences)
