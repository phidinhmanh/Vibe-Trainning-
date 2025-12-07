"""
Physicochemical feature extraction for protein sequences.

Extracts amino acid composition, dipeptide/tripeptide frequencies,
and physical property statistics.
"""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


# Amino acid molecular weights
AA_WEIGHTS = {
    "A": 89,
    "C": 121,
    "D": 133,
    "E": 147,
    "F": 165,
    "G": 75,
    "H": 155,
    "I": 131,
    "K": 146,
    "L": 131,
    "M": 149,
    "N": 132,
    "P": 115,
    "Q": 146,
    "R": 174,
    "S": 105,
    "T": 119,
    "V": 117,
    "W": 204,
    "Y": 181,
}

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Hydrophobic amino acids
HYDROPHOBIC = set("AILMFWYV")

# Charged amino acids
CHARGED = set("DEKR")

# Most common dipeptides
TOP_DIPEPTIDES = [
    "AL",
    "LA",
    "LE",
    "EA",
    "AA",
    "AS",
    "SA",
    "EL",
    "LL",
    "AE",
    "SE",
    "ES",
    "GA",
    "AG",
    "VA",
    "AV",
    "LV",
    "VL",
    "LS",
    "SL",
]

# Most common tripeptides
TOP_TRIPEPTIDES = [
    "ALA",
    "LEA",
    "EAL",
    "LAL",
    "AAA",
    "LLE",
    "ELE",
    "ALE",
    "GAL",
    "ASA",
    "VLA",
    "LAV",
    "SLS",
    "LSL",
    "GLA",
    "LAG",
    "AVL",
    "VLA",
    "SLE",
    "LES",
]


class PhysChemFeaturizer:
    """
    Extract physicochemical features from protein sequences.

    Default feature set (85 dimensions):
    - Amino acid frequencies (20)
    - Physical properties (4): log length, hydrophobic ratio, charged ratio, log MW
    - Dipeptide frequencies (20)
    - Tripeptide frequencies (20)
    - Additional composition features (21)
    """

    def __init__(
        self,
        include_aa_freq: bool = True,
        include_phys_props: bool = True,
        include_dipeptides: bool = True,
        include_tripeptides: bool = True,
        top_dipeptides: Optional[List[str]] = None,
        top_tripeptides: Optional[List[str]] = None,
        n_features: int = 85,
    ):
        """
        Args:
            include_aa_freq: Include amino acid frequency features.
            include_phys_props: Include physical property features.
            include_dipeptides: Include dipeptide frequency features.
            include_tripeptides: Include tripeptide frequency features.
            top_dipeptides: Custom list of dipeptides to track.
            top_tripeptides: Custom list of tripeptides to track.
            n_features: Target feature dimension (pad/truncate to match).
        """
        self.include_aa_freq = include_aa_freq
        self.include_phys_props = include_phys_props
        self.include_dipeptides = include_dipeptides
        self.include_tripeptides = include_tripeptides
        self.top_dipeptides = top_dipeptides or TOP_DIPEPTIDES
        self.top_tripeptides = top_tripeptides or TOP_TRIPEPTIDES
        self.n_features = n_features

    def transform(self, sequences: List[str]) -> np.ndarray:
        """
        Extract features from sequences.

        Args:
            sequences: List of protein sequences.

        Returns:
            Feature matrix of shape (n_sequences, n_features).
        """
        features = []

        for seq in tqdm(sequences, desc="Extracting physchem features"):
            vec = self._extract_single(seq)
            features.append(vec)

        return np.array(features, dtype=np.float32)

    def _extract_single(self, seq: str) -> np.ndarray:
        """Extract features from a single sequence."""
        # Handle empty or very short sequences
        if not seq or len(seq) < 3:
            return np.zeros(self.n_features, dtype=np.float32)

        n = len(seq)
        c = Counter(seq)
        feature_parts = []

        # 1. Amino acid frequencies (20 features)
        if self.include_aa_freq:
            aa_freq = [c.get(a, 0) / n for a in AMINO_ACIDS]
            feature_parts.extend(aa_freq)

        # 2. Physical properties (4 features)
        if self.include_phys_props:
            phys = [
                np.log1p(n),  # Log of sequence length
                sum(c.get(a, 0) for a in HYDROPHOBIC) / n,  # Hydrophobic ratio
                sum(c.get(a, 0) for a in CHARGED) / n,  # Charged ratio
                np.log1p(sum(c.get(a, 0) * AA_WEIGHTS.get(a, 0) for a in c)),  # Log MW
            ]
            feature_parts.extend(phys)

        # 3. Dipeptide frequencies (20 features)
        if self.include_dipeptides:
            di_counter = Counter(seq[i : i + 2] for i in range(n - 1))
            di_freq = [
                di_counter.get(d, 0) / max(1, n - 1) for d in self.top_dipeptides
            ]
            feature_parts.extend(di_freq)

        # 4. Tripeptide frequencies (20 features)
        if self.include_tripeptides:
            tri_counter = Counter(seq[i : i + 3] for i in range(n - 2))
            tri_freq = [
                tri_counter.get(t, 0) / max(1, n - 2) for t in self.top_tripeptides
            ]
            feature_parts.extend(tri_freq)

        # Convert to array
        vec = np.array(feature_parts, dtype=np.float32)

        # Pad or truncate to target dimension
        if len(vec) >= self.n_features:
            return vec[: self.n_features]
        else:
            return np.pad(vec, (0, self.n_features - len(vec)))

    def fit_transform(self, sequences: List[str]) -> np.ndarray:
        """Fit and transform (stateless, so same as transform)."""
        return self.transform(sequences)


def extract_features(sequences: List[str], n_features: int = 85) -> np.ndarray:
    """
    Quick helper to extract physicochemical features.

    This is the function used in the sample notebook.

    Args:
        sequences: List of protein sequences.
        n_features: Target feature dimension.

    Returns:
        Feature matrix of shape (n_sequences, n_features).
    """
    featurizer = PhysChemFeaturizer(n_features=n_features)
    return featurizer.transform(sequences)


def compute_sequence_stats(sequences: List[str]) -> Dict[str, float]:
    """
    Compute summary statistics for a set of sequences.

    Useful for data exploration.
    """
    lengths = [len(s) for s in sequences]

    return {
        "n_sequences": len(sequences),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "mean_length": np.mean(lengths) if lengths else 0,
        "median_length": np.median(lengths) if lengths else 0,
        "std_length": np.std(lengths) if lengths else 0,
    }
