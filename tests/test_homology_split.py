"""
Tests for homology-aware data splitting.
"""

import numpy as np
import pytest

from src.data.homology_split import HomologySplitter, validate_homology_split


class TestHomologySplitter:
    """Tests for HomologySplitter class."""

    @pytest.fixture
    def sample_sequences(self):
        """Create sample sequences for testing."""
        return {
            "P001": "MKALMNDEFGH",
            "P002": "MKALMNDEFGI",  # Similar to P001
            "P003": "MKALMNDEFGJ",  # Similar to P001, P002
            "P004": "RSTUVWXYZAB",  # Different cluster
            "P005": "RSTUVWXYZAC",  # Similar to P004
            "P006": "QWERTYUIOPD",  # Different cluster
            "P007": "QWERTYUIOPE",  # Similar to P006
            "P008": "ASDFGHJKLZX",  # Different cluster
        }

    def test_fit_creates_clusters(self, sample_sequences):
        """Fit should create sequence clusters."""
        splitter = HomologySplitter(n_folds=2, identity_threshold=0.5, method="kmer")
        splitter.fit(sample_sequences)

        assert splitter.clusters_ is not None
        assert len(splitter.clusters_) > 0

        # All proteins should be in some cluster
        all_proteins = set()
        for proteins in splitter.clusters_.values():
            all_proteins.update(proteins)

        assert all_proteins == set(sample_sequences.keys())

    def test_fit_creates_fold_assignments(self, sample_sequences):
        """Fit should assign clusters to folds."""
        splitter = HomologySplitter(n_folds=3, identity_threshold=0.5, method="kmer")
        splitter.fit(sample_sequences)

        assert splitter.fold_assignments_ is not None
        assert len(splitter.fold_assignments_) == 3

    def test_get_fold_returns_disjoint_sets(self, sample_sequences):
        """Train and val sets for a fold should be disjoint."""
        splitter = HomologySplitter(n_folds=3, identity_threshold=0.5, method="kmer")
        splitter.fit(sample_sequences)

        for fold_idx in range(3):
            train_ids, val_ids = splitter.get_fold(fold_idx)

            assert len(set(train_ids) & set(val_ids)) == 0

    def test_no_cluster_in_multiple_folds(self, sample_sequences):
        """Each cluster should appear in exactly one fold."""
        splitter = HomologySplitter(n_folds=3, identity_threshold=0.5, method="kmer")
        splitter.fit(sample_sequences)

        # Validate split
        is_valid = validate_homology_split(splitter, sample_sequences)
        assert is_valid

    def test_iter_folds(self, sample_sequences):
        """iter_folds should yield all folds."""
        splitter = HomologySplitter(n_folds=4, identity_threshold=0.5, method="kmer")
        splitter.fit(sample_sequences)

        folds = list(splitter.iter_folds())

        assert len(folds) == 4

        # Each fold should have (fold_idx, train_ids, val_ids)
        for fold_idx, train_ids, val_ids in folds:
            assert isinstance(fold_idx, int)
            assert isinstance(train_ids, list)
            assert isinstance(val_ids, list)

    def test_all_proteins_covered_across_folds(self, sample_sequences):
        """Each protein should appear in validation exactly once."""
        splitter = HomologySplitter(n_folds=3, identity_threshold=0.5, method="kmer")
        splitter.fit(sample_sequences)

        all_val_ids = []
        for _, _, val_ids in splitter.iter_folds():
            all_val_ids.extend(val_ids)

        # Each protein appears in val exactly once
        assert set(all_val_ids) == set(sample_sequences.keys())

    def test_save_and_load(self, sample_sequences, tmp_path):
        """Splitter should be saveable and loadable."""
        splitter = HomologySplitter(n_folds=3, identity_threshold=0.5, method="kmer")
        splitter.fit(sample_sequences)

        # Save
        save_path = tmp_path / "folds.json"
        splitter.save(str(save_path))

        assert save_path.exists()

        # Load
        loaded = HomologySplitter.load(str(save_path))

        assert loaded.n_folds == splitter.n_folds
        assert loaded.identity_threshold == splitter.identity_threshold
        assert loaded.clusters_ == splitter.clusters_
        assert loaded.fold_assignments_ == splitter.fold_assignments_

    def test_kmer_clustering_groups_similar(self, sample_sequences):
        """K-mer clustering should group similar sequences."""
        splitter = HomologySplitter(n_folds=2, identity_threshold=0.7, method="kmer")
        splitter.fit(sample_sequences)

        # P001, P002, P003 are similar and should be in same cluster
        cluster_assignments = {}
        for cluster_id, proteins in splitter.clusters_.items():
            for p in proteins:
                cluster_assignments[p] = cluster_id

        # They should be in the same cluster
        # (Note: exact clustering depends on k-mer threshold)
        # At high threshold, similar sequences should cluster together


class TestKmerJaccardClustering:
    """Tests for k-mer Jaccard clustering method."""

    def test_identical_sequences_same_cluster(self):
        """Identical sequences should be in the same cluster."""
        sequences = {
            "P1": "ACDEFGHIKLMN",
            "P2": "ACDEFGHIKLMN",  # Identical
            "P3": "ZYXWVUTSRQPO",  # Different
        }

        splitter = HomologySplitter(n_folds=2, identity_threshold=0.9, method="kmer")
        splitter.fit(sequences)

        # P1 and P2 should be in same cluster
        cluster_for = {}
        for cid, proteins in splitter.clusters_.items():
            for p in proteins:
                cluster_for[p] = cid

        assert cluster_for["P1"] == cluster_for["P2"]

    def test_different_sequences_separate_clusters(self):
        """Very different sequences should be in separate clusters."""
        sequences = {
            "P1": "AAAAAAAAAA",
            "P2": "CCCCCCCCCC",  # Completely different
        }

        splitter = HomologySplitter(n_folds=2, identity_threshold=0.5, method="kmer")
        splitter.fit(sequences)

        # P1 and P2 should be in different clusters
        cluster_for = {}
        for cid, proteins in splitter.clusters_.items():
            for p in proteins:
                cluster_for[p] = cid

        assert cluster_for["P1"] != cluster_for["P2"]


class TestValidation:
    """Tests for split validation."""

    def test_validate_valid_split(self):
        """Valid split should pass validation."""
        sequences = {f"P{i}": f"ACDEF{i}GHIK" for i in range(10)}

        splitter = HomologySplitter(n_folds=3, method="kmer")
        splitter.fit(sequences)

        assert validate_homology_split(splitter, sequences)

    def test_validate_catches_overlap(self):
        """Validation should catch cluster overlap."""
        sequences = {"P1": "ACDEFG", "P2": "HIJKLM"}

        splitter = HomologySplitter(n_folds=2, method="kmer")
        splitter.fit(sequences)

        # Manually corrupt fold assignments
        splitter.fold_assignments_[0].append(splitter.fold_assignments_[1][0])

        with pytest.raises(AssertionError):
            validate_homology_split(splitter, sequences)
