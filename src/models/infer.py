"""
Batch inference CLI for CAFA 6.

Usage:
    python src/models/infer.py --config configs/default.yaml --input test.fasta --output predictions.npy
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.ingest import load_sequences
from src.features.tfidf_kmers import TfidfKmerFeaturizer
from src.features.physchem import PhysChemFeaturizer, extract_features
from src.models.baseline_lr import BaselineLR
from src.models.knn_transfer import KNNTransfer
from src.utils import load_config


class InferencePipeline:
    """
    End-to-end inference pipeline.

    Loads trained models and generates predictions for new sequences.
    """

    def __init__(self, config_path: str, model_dir: str = "experiments/models"):
        """
        Args:
            config_path: Path to configuration file.
            model_dir: Directory containing trained models.
        """
        self.config = load_config(config_path)
        self.model_dir = model_dir

        self.tfidf_featurizer: Optional[TfidfKmerFeaturizer] = None
        self.physchem_featurizer: Optional[PhysChemFeaturizer] = None
        self.lr_model: Optional[BaselineLR] = None
        self.knn_model: Optional[KNNTransfer] = None
        self.go_terms: Optional[List[str]] = None

        self._load_components()

    def _load_components(self) -> None:
        """Load all pipeline components."""
        import joblib

        # Load TF-IDF featurizer
        tfidf_path = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")
        if os.path.exists(tfidf_path):
            self.tfidf_featurizer = TfidfKmerFeaturizer(
                k=self.config["features"]["kmer_k"],
                max_features=self.config["features"]["tfidf_max_features"],
            )
            self.tfidf_featurizer.load(tfidf_path)

        # Load LR model
        lr_path = os.path.join(self.model_dir, "baseline_lr.joblib")
        if os.path.exists(lr_path):
            self.lr_model = BaselineLR.load(lr_path)

        # Load KNN model
        knn_path = os.path.join(self.model_dir, "knn_transfer.joblib")
        if os.path.exists(knn_path):
            self.knn_model = KNNTransfer.load(knn_path)

        # Load GO terms
        terms_path = os.path.join(self.model_dir, "go_terms.json")
        if os.path.exists(terms_path):
            import json

            with open(terms_path, "r") as f:
                self.go_terms = json.load(f)

    def predict(
        self, sequences: Dict[str, str], batch_size: int = 2048, model: str = "all"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate predictions for input sequences.

        Args:
            sequences: Dict mapping protein IDs to sequences.
            batch_size: Batch size for processing.
            model: Which model to use ('lr', 'knn', 'all').

        Returns:
            Tuple of (predictions array, protein ID list).
        """
        protein_ids = list(sequences.keys())
        seq_list = [sequences[pid] for pid in protein_ids]

        predictions = []

        # LR predictions
        if model in ["lr", "all"] and self.lr_model is not None:
            if self.tfidf_featurizer is not None:
                X_tfidf = self.tfidf_featurizer.transform(seq_list)
                pred_lr = self.lr_model.predict_proba(X_tfidf)
                predictions.append(pred_lr)

        # KNN predictions
        if model in ["knn", "all"] and self.knn_model is not None:
            X_phys = extract_features(seq_list)
            pred_knn = self.knn_model.predict_proba(X_phys, batch_size=batch_size)
            predictions.append(pred_knn)

        if not predictions:
            raise ValueError("No models available for inference.")

        # Combine predictions
        if len(predictions) > 1:
            # Weighted average
            knn_weight = self.config.get("ensemble", {}).get("knn_weight", 0.32)
            lr_weight = 1.0 - knn_weight

            combined = lr_weight * predictions[0] + knn_weight * predictions[1]
        else:
            combined = predictions[0]

        return combined, protein_ids

    def to_submission(
        self,
        predictions: np.ndarray,
        protein_ids: List[str],
        output_path: str,
        top_k: int = 1500,
        min_score: float = 0.01,
    ) -> None:
        """
        Convert predictions to submission TSV format.

        Args:
            predictions: Prediction matrix, shape (N, C).
            protein_ids: List of protein IDs.
            output_path: Output file path.
            top_k: Maximum predictions per protein.
            min_score: Minimum score threshold.
        """
        if self.go_terms is None:
            self.go_terms = [f"GO:{i:07d}" for i in range(predictions.shape[1])]

        with open(output_path, "w") as f:
            for i, pid in enumerate(protein_ids):
                scores = predictions[i]

                # Get top predictions
                sorted_idx = np.argsort(-scores)[:top_k]

                for idx in sorted_idx:
                    score = scores[idx]
                    if score >= min_score:
                        f.write(f"{pid}\t{self.go_terms[idx]}\t{score:.4f}\n")


def main():
    parser = argparse.ArgumentParser(description="CAFA 6 Batch Inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--input", type=str, required=True, help="Input FASTA file")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument(
        "--format",
        type=str,
        choices=["npy", "tsv"],
        default="npy",
        help="Output format",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="experiments/models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["lr", "knn", "all"],
        default="all",
        help="Which model(s) to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for prediction"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1500,
        help="Max predictions per protein (for TSV output)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.01,
        help="Minimum score threshold (for TSV output)",
    )

    args = parser.parse_args()

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    sequences = load_sequences(args.input)
    print(f"Loaded {len(sequences)} sequences")

    # Initialize pipeline
    print("Initializing inference pipeline...")
    pipeline = InferencePipeline(args.config, args.model_dir)

    # Generate predictions
    print("Generating predictions...")
    predictions, protein_ids = pipeline.predict(
        sequences, batch_size=args.batch_size, model=args.model
    )

    print(f"Generated predictions: shape {predictions.shape}")

    # Save output
    if args.format == "npy":
        np.save(args.output, predictions)

        # Also save protein IDs
        ids_path = args.output.replace(".npy", "_ids.txt")
        with open(ids_path, "w") as f:
            for pid in protein_ids:
                f.write(f"{pid}\n")

        print(f"Saved predictions to {args.output}")
        print(f"Saved protein IDs to {ids_path}")

    else:  # TSV format
        pipeline.to_submission(
            predictions,
            protein_ids,
            args.output,
            top_k=args.top_k,
            min_score=args.min_score,
        )
        print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
