"""
Fast clustering utilities for large-scale sequence datasets.

Uses MinHash LSH for approximate Jaccard similarity - O(n) instead of O(n²).
"""

import hashlib
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm


class MinHashLSH:
    """
    MinHash Locality-Sensitive Hashing for fast approximate clustering.

    Much faster than pairwise comparison: O(n) vs O(n²).
    """

    def __init__(
        self, num_hashes: int = 128, bands: int = 16, k: int = 3, seed: int = 42
    ):
        """
        Args:
            num_hashes: Number of hash functions (higher = more accurate).
            bands: Number of LSH bands (higher = more candidates, slower).
            k: K-mer size.
            seed: Random seed.
        """
        self.num_hashes = num_hashes
        self.bands = bands
        self.rows_per_band = num_hashes // bands
        self.k = k
        self.seed = seed

        # Generate random hash parameters
        np.random.seed(seed)
        self.hash_a = np.random.randint(1, 2**31, size=num_hashes)
        self.hash_b = np.random.randint(0, 2**31, size=num_hashes)
        self.prime = 2**31 - 1

    def _get_kmers(self, seq: str) -> Set[int]:
        """Convert sequence to set of hashed k-mers."""
        kmers = set()
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i : i + self.k]
            h = int(hashlib.md5(kmer.encode()).hexdigest(), 16) % self.prime
            kmers.add(h)
        return kmers

    def _minhash_signature(self, kmers: Set[int]) -> np.ndarray:
        """Compute MinHash signature for a set of k-mers."""
        if not kmers:
            return np.full(self.num_hashes, np.inf)

        signature = np.full(self.num_hashes, np.inf)
        kmers_arr = np.array(list(kmers))

        for i in range(self.num_hashes):
            # Hash: (a*x + b) mod prime
            hashed = (self.hash_a[i] * kmers_arr + self.hash_b[i]) % self.prime
            signature[i] = hashed.min()

        return signature

    def cluster(
        self, sequences: Dict[str, str], min_similarity: float = 0.7
    ) -> Dict[str, List[str]]:
        """
        Cluster sequences using MinHash LSH.

        Args:
            sequences: Dict mapping protein IDs to sequences.
            min_similarity: Minimum Jaccard similarity for same cluster.

        Returns:
            Dict mapping cluster representative to list of member IDs.
        """
        protein_ids = list(sequences.keys())
        n = len(protein_ids)

        print(f"Computing MinHash signatures for {n} sequences...")

        # Compute signatures
        signatures = {}
        for pid in tqdm(protein_ids, desc="MinHash"):
            kmers = self._get_kmers(sequences[pid])
            signatures[pid] = self._minhash_signature(kmers)

        # LSH: group by band hashes
        print("Building LSH index...")
        buckets = defaultdict(set)

        for pid in protein_ids:
            sig = signatures[pid]
            for band_idx in range(self.bands):
                start = band_idx * self.rows_per_band
                end = start + self.rows_per_band
                band_hash = hash(tuple(sig[start:end]))
                buckets[(band_idx, band_hash)].add(pid)

        # Find candidate pairs from buckets
        print("Finding candidate pairs...")
        candidates = defaultdict(set)
        for bucket_ids in buckets.values():
            if len(bucket_ids) > 1:
                bucket_list = list(bucket_ids)
                for i, pid1 in enumerate(bucket_list):
                    for pid2 in bucket_list[i + 1 :]:
                        candidates[pid1].add(pid2)
                        candidates[pid2].add(pid1)

        # Union-Find clustering
        print("Clustering...")
        parent = {pid: pid for pid in protein_ids}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Verify candidates with signature similarity
        for pid1, cands in tqdm(candidates.items(), desc="Verifying"):
            sig1 = signatures[pid1]
            for pid2 in cands:
                if find(pid1) == find(pid2):
                    continue
                sig2 = signatures[pid2]
                # Estimate Jaccard from signature similarity
                sim = np.mean(sig1 == sig2)
                if sim >= min_similarity:
                    union(pid1, pid2)

        # Collect clusters
        clusters = defaultdict(list)
        for pid in protein_ids:
            root = find(pid)
            clusters[root].append(pid)

        print(f"Created {len(clusters)} clusters")
        return dict(clusters)


def fast_random_cluster(
    sequences: Dict[str, str], n_clusters: int = 5000, seed: int = 42
) -> Dict[str, List[str]]:
    """
    Fast random clustering (for testing/baseline).

    NOT homology-aware, but fast. Use for quick pipeline testing.
    """
    np.random.seed(seed)
    protein_ids = list(sequences.keys())
    n = len(protein_ids)

    # Assign each protein to a random cluster
    cluster_ids = np.random.randint(0, n_clusters, size=n)

    clusters = defaultdict(list)
    for pid, cid in zip(protein_ids, cluster_ids):
        clusters[f"cluster_{cid}"].append(pid)

    return dict(clusters)


def fast_length_cluster(
    sequences: Dict[str, str], n_bins: int = 100
) -> Dict[str, List[str]]:
    """
    Cluster by sequence length bins.

    Rough approximation: similar length sequences are more likely to be similar.
    Much faster than pairwise comparison.
    """
    lengths = [(pid, len(seq)) for pid, seq in sequences.items()]
    lengths.sort(key=lambda x: x[1])

    clusters = defaultdict(list)
    bin_size = len(lengths) // n_bins + 1

    for i, (pid, _) in enumerate(lengths):
        cluster_id = f"len_cluster_{i // bin_size}"
        clusters[cluster_id].append(pid)

    return dict(clusters)
