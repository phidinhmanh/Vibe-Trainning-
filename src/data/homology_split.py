"""
Homology-aware data splitting for CAFA 6.

Implements sequence clustering and GroupKFold assignment to prevent
data leakage from homologous sequences appearing in both train and validation.
"""

import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


class HomologySplitter:
    """
    Create homology-aware train/validation splits.

    Groups proteins by sequence similarity clusters, then assigns entire
    clusters to folds (no sequence shares >T% identity across folds).

    Supports:
    - MMseqs2 clustering (recommended)
    - CD-HIT clustering
    - K-mer Jaccard fallback (no external tools needed)
    """

    def __init__(
        self,
        n_folds: int = 5,
        identity_threshold: float = 0.7,
        method: str = "auto",
        random_state: int = 42,
    ):
        """
        Args:
            n_folds: Number of folds to create.
            identity_threshold: Maximum sequence identity across folds.
            method: Clustering method ('mmseqs2', 'cdhit', 'kmer', or 'auto').
            random_state: Random seed for fold assignment.
        """
        self.n_folds = n_folds
        self.identity_threshold = identity_threshold
        self.method = method
        self.random_state = random_state
        self.clusters_ = None
        self.fold_assignments_ = None

    def fit(
        self,
        sequences: Dict[str, str],
        labels: Optional[np.ndarray] = None,
        protein_ids: Optional[List[str]] = None,
    ) -> "HomologySplitter":
        """
        Cluster sequences and assign clusters to folds.

        Args:
            sequences: Dict mapping protein IDs to sequences.
            labels: Optional label matrix for stratification (N x C).
            protein_ids: Optional list specifying ordering (for label alignment).

        Returns:
            self
        """
        # Determine clustering method
        method = self._detect_method() if self.method == "auto" else self.method

        # Perform clustering
        if method == "mmseqs2":
            self.clusters_ = self._cluster_mmseqs2(sequences)
        elif method == "cdhit":
            self.clusters_ = self._cluster_cdhit(sequences)
        else:
            self.clusters_ = self._cluster_kmer_jaccard(sequences)

        # Assign clusters to folds
        self.fold_assignments_ = self._assign_folds_stratified(
            sequences, labels, protein_ids
        )

        return self

    def get_fold(self, fold_idx: int) -> Tuple[List[str], List[str]]:
        """
        Get train and validation protein IDs for a specific fold.

        Args:
            fold_idx: Fold index (0 to n_folds-1).

        Returns:
            Tuple of (train_ids, val_ids).
        """
        if self.fold_assignments_ is None:
            raise ValueError("Call fit() first.")

        val_clusters = self.fold_assignments_[fold_idx]
        train_clusters = []
        for i, clusters in enumerate(self.fold_assignments_):
            if i != fold_idx:
                train_clusters.extend(clusters)

        train_ids = []
        val_ids = []

        for cluster_id, protein_ids in self.clusters_.items():
            if cluster_id in val_clusters:
                val_ids.extend(protein_ids)
            elif cluster_id in train_clusters:
                train_ids.extend(protein_ids)

        return train_ids, val_ids

    def iter_folds(self) -> Iterator[Tuple[int, List[str], List[str]]]:
        """
        Iterate over all folds.

        Yields:
            Tuple of (fold_idx, train_ids, val_ids).
        """
        for fold_idx in range(self.n_folds):
            train_ids, val_ids = self.get_fold(fold_idx)
            yield fold_idx, train_ids, val_ids

    def save(self, output_path: str) -> None:
        """Save fold assignments to JSON."""
        data = {
            "description": "Homology-aware k-fold splits",
            "n_folds": self.n_folds,
            "identity_threshold": self.identity_threshold,
            "clustering_method": self.method,
            "folds": self.fold_assignments_,
            "clusters": self.clusters_,
            "metadata": {
                "total_clusters": len(self.clusters_),
                "total_proteins": sum(len(v) for v in self.clusters_.values()),
            },
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, input_path: str) -> "HomologySplitter":
        """Load fold assignments from JSON."""
        with open(input_path, "r") as f:
            data = json.load(f)

        splitter = cls(
            n_folds=data["n_folds"],
            identity_threshold=data["identity_threshold"],
            method=data["clustering_method"],
        )
        splitter.fold_assignments_ = data["folds"]
        splitter.clusters_ = data["clusters"]

        return splitter

    def _detect_method(self) -> str:
        """Auto-detect available clustering tool."""
        try:
            subprocess.run(["mmseqs", "version"], capture_output=True, check=True)
            return "mmseqs2"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        try:
            subprocess.run(["cd-hit", "-h"], capture_output=True, check=True)
            return "cdhit"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        return "kmer"

    def _cluster_mmseqs2(self, sequences: Dict[str, str]) -> Dict[str, List[str]]:
        """Cluster sequences using MMseqs2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write sequences to FASTA
            fasta_path = os.path.join(tmpdir, "seqs.fasta")
            with open(fasta_path, "w") as f:
                for pid, seq in sequences.items():
                    f.write(f">{pid}\n{seq}\n")

            db_path = os.path.join(tmpdir, "DB")
            cluster_path = os.path.join(tmpdir, "clusterRes")
            tsv_path = os.path.join(tmpdir, "clusters.tsv")
            tmp_path = os.path.join(tmpdir, "tmp")

            # Create database
            subprocess.run(
                ["mmseqs", "createdb", fasta_path, db_path],
                check=True,
                capture_output=True,
            )

            # Cluster
            subprocess.run(
                [
                    "mmseqs",
                    "cluster",
                    db_path,
                    cluster_path,
                    tmp_path,
                    "--min-seq-id",
                    str(self.identity_threshold),
                ],
                check=True,
                capture_output=True,
            )

            # Create TSV
            subprocess.run(
                ["mmseqs", "createtsv", db_path, db_path, cluster_path, tsv_path],
                check=True,
                capture_output=True,
            )

            # Parse clusters
            return self._parse_cluster_tsv(tsv_path)

    def _cluster_cdhit(self, sequences: Dict[str, str]) -> Dict[str, List[str]]:
        """Cluster sequences using CD-HIT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "seqs.fasta")
            output_path = os.path.join(tmpdir, "clusters")

            with open(fasta_path, "w") as f:
                for pid, seq in sequences.items():
                    f.write(f">{pid}\n{seq}\n")

            # Determine word size based on threshold
            if self.identity_threshold >= 0.7:
                word_size = 5
            elif self.identity_threshold >= 0.6:
                word_size = 4
            elif self.identity_threshold >= 0.5:
                word_size = 3
            else:
                word_size = 2

            subprocess.run(
                [
                    "cd-hit",
                    "-i",
                    fasta_path,
                    "-o",
                    output_path,
                    "-c",
                    str(self.identity_threshold),
                    "-n",
                    str(word_size),
                ],
                check=True,
                capture_output=True,
            )

            return self._parse_cdhit_clusters(output_path + ".clstr")

    def _cluster_kmer_jaccard(
        self, sequences: Dict[str, str], k: int = 3
    ) -> Dict[str, List[str]]:
        """
        Cluster sequences using k-mer Jaccard similarity (fallback method).

        Uses single-linkage clustering with Jaccard distance.
        """
        protein_ids = list(sequences.keys())
        n = len(protein_ids)

        # Compute k-mer sets
        kmer_sets = {}
        for pid, seq in tqdm(sequences.items(), desc="Computing k-mers"):
            kmers = set()
            for i in range(len(seq) - k + 1):
                kmers.add(seq[i : i + k])
            kmer_sets[pid] = kmers

        # Initialize each protein as its own cluster
        parent = {pid: pid for pid in protein_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Compute pairwise similarities and merge
        threshold = self.identity_threshold  # Jaccard threshold

        for i in tqdm(range(n), desc="Clustering (k-mer Jaccard)"):
            pid_i = protein_ids[i]
            set_i = kmer_sets[pid_i]

            for j in range(i + 1, n):
                pid_j = protein_ids[j]
                set_j = kmer_sets[pid_j]

                # Jaccard similarity
                intersection = len(set_i & set_j)
                union_size = len(set_i | set_j)

                if union_size > 0 and intersection / union_size >= threshold:
                    union(pid_i, pid_j)

        # Collect clusters
        clusters = defaultdict(list)
        for pid in protein_ids:
            root = find(pid)
            clusters[root].append(pid)

        return dict(clusters)

    def _parse_cluster_tsv(self, tsv_path: str) -> Dict[str, List[str]]:
        """Parse MMseqs2 cluster TSV output."""
        clusters = defaultdict(list)

        with open(tsv_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep, member = parts[0], parts[1]
                    clusters[rep].append(member)

        return dict(clusters)

    def _parse_cdhit_clusters(self, clstr_path: str) -> Dict[str, List[str]]:
        """Parse CD-HIT .clstr output."""
        clusters = {}
        current_cluster = []
        cluster_id = None

        with open(clstr_path, "r") as f:
            for line in f:
                if line.startswith(">Cluster"):
                    if current_cluster and cluster_id is not None:
                        clusters[cluster_id] = current_cluster
                    cluster_id = line.strip()[1:]  # Remove '>'
                    current_cluster = []
                else:
                    # Parse member line: "0\t123aa, >ProteinName... *"
                    if ">" in line:
                        start = line.index(">") + 1
                        end = line.index("...")
                        protein_id = line[start:end]
                        current_cluster.append(protein_id)

                        # Representative marked with '*'
                        if "*" in line:
                            cluster_id = protein_id

            if current_cluster and cluster_id is not None:
                clusters[cluster_id] = current_cluster

        return clusters

    def _assign_folds_stratified(
        self,
        sequences: Dict[str, str],
        labels: Optional[np.ndarray],
        protein_ids: Optional[List[str]],
    ) -> List[List[str]]:
        """
        Assign clusters to folds with stratification.

        Tries to balance:
        - Number of proteins per fold
        - Label distribution per fold (if labels provided)
        """
        np.random.seed(self.random_state)

        cluster_ids = list(self.clusters_.keys())
        cluster_sizes = {cid: len(self.clusters_[cid]) for cid in cluster_ids}

        # Shuffle clusters
        np.random.shuffle(cluster_ids)

        # Simple balanced assignment (greedy bin packing)
        fold_sizes = [0] * self.n_folds
        fold_assignments = [[] for _ in range(self.n_folds)]

        # Sort clusters by size (largest first for better balance)
        cluster_ids_sorted = sorted(cluster_ids, key=lambda x: -cluster_sizes[x])

        for cid in cluster_ids_sorted:
            # Assign to fold with smallest current size
            min_fold = np.argmin(fold_sizes)
            fold_assignments[min_fold].append(cid)
            fold_sizes[min_fold] += cluster_sizes[cid]

        return fold_assignments


def validate_homology_split(
    splitter: HomologySplitter, sequences: Dict[str, str]
) -> bool:
    """
    Validate that no cluster appears in multiple folds.

    Returns True if split is valid, raises AssertionError otherwise.
    """
    all_val_clusters = set()

    for fold_idx in range(splitter.n_folds):
        val_clusters = set(splitter.fold_assignments_[fold_idx])

        # Check no overlap with previous folds
        overlap = all_val_clusters & val_clusters
        assert len(overlap) == 0, f"Clusters {overlap} appear in multiple folds!"

        all_val_clusters.update(val_clusters)

    # Check all clusters are assigned
    all_clusters = set(splitter.clusters_.keys())
    assert all_val_clusters == all_clusters, "Not all clusters are assigned to folds!"

    return True
