"""
Optuna hyperparameter optimization for CAFA 6.

Optimizes model hyperparameters to maximize weighted F-Max.
"""

import argparse
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Import project modules
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.ingest import load_sequences, load_terms, get_top_go_terms, prepare_labels
from src.data.homology_split import HomologySplitter
from src.features.tfidf_kmers import TfidfKmerFeaturizer
from src.features.physchem import PhysChemFeaturizer
from src.models.baseline_lr import BaselineLR
from src.score.scorer import CafaScorer
from src.utils import load_config, set_seed


class CafaObjective:
    """
    Optuna objective for CAFA hyperparameter optimization.

    Uses validation weighted F-Max as the optimization target.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        protein_ids: list,
        splitter: HomologySplitter,
        ia_weights: Optional[np.ndarray] = None,
        val_fold: int = -1,
    ):
        """
        Args:
            X: Full feature matrix.
            y: Full label matrix.
            protein_ids: List of protein IDs.
            splitter: Fitted HomologySplitter.
            ia_weights: Optional IA weights.
            val_fold: Which fold to use for validation (-1 for last).
        """
        self.X = X
        self.y = y
        self.protein_ids = protein_ids
        self.splitter = splitter
        self.ia_weights = ia_weights
        self.val_fold = val_fold if val_fold >= 0 else splitter.n_folds - 1

        # Get train/val split
        self.train_ids, self.val_ids = splitter.get_fold(self.val_fold)

        # Build index mapping
        self.pid_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

        self.train_idx = [
            self.pid_to_idx[pid] for pid in self.train_ids if pid in self.pid_to_idx
        ]
        self.val_idx = [
            self.pid_to_idx[pid] for pid in self.val_ids if pid in self.pid_to_idx
        ]

        self.X_train = self.X[self.train_idx]
        self.y_train = self.y[self.train_idx]
        self.X_val = self.X[self.val_idx]
        self.y_val = self.y[self.val_idx]

        self.scorer = CafaScorer(ia_weights=ia_weights)

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Optuna trial objective.

        Args:
            trial: Optuna trial object.

        Returns:
            Validation F-Max (to maximize).
        """
        # Sample hyperparameters
        C = trial.suggest_float("C", 1e-4, 1e2, log=True)
        solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
        max_iter = trial.suggest_int("max_iter", 500, 2000, step=500)
        calibrate = trial.suggest_categorical("calibrate", [True, False])
        calibration_method = (
            trial.suggest_categorical("calibration_method", ["sigmoid", "isotonic"])
            if calibrate
            else "sigmoid"
        )

        # Train model
        model = BaselineLR(
            C=C,
            max_iter=max_iter,
            solver=solver,
            calibrate=calibrate,
            calibration_method=calibration_method,
        )

        if calibrate:
            # Split for calibration
            n_train = int(0.8 * len(self.X_train))
            model.fit(
                self.X_train[:n_train],
                self.y_train[:n_train],
                self.X_train[n_train:],
                self.y_train[n_train:],
            )
        else:
            model.fit(self.X_train, self.y_train)

        # Predict
        proba = model.predict_proba(self.X_val)

        # Score
        result = self.scorer.score(self.y_val, proba)

        # Report intermediate value for pruning
        trial.report(result.f1, step=0)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return result.f1


def run_optuna_search(
    config: Dict[str, Any],
    n_trials: int = 100,
    study_name: str = "cafa6_optimization",
    storage: Optional[str] = None,
    seed: int = 42,
) -> optuna.Study:
    """
    Run Optuna hyperparameter search.

    Args:
        config: Configuration dictionary.
        n_trials: Number of trials.
        study_name: Name for the study.
        storage: SQLite path for persistence (optional).
        seed: Random seed.

    Returns:
        Completed Optuna study.
    """
    set_seed(seed)

    # Load data
    print("Loading data...")
    sequences = load_sequences(config["data"]["train_sequences"])
    terms_df = load_terms(config["data"]["train_terms"])

    protein_ids = list(sequences.keys())
    top_go = get_top_go_terms(
        terms_df, n_terms=config["features"].get("top_go_terms", 1500)
    )
    y, go2idx = prepare_labels(protein_ids, terms_df, top_go)

    # Extract features
    print("Extracting features...")
    seq_list = [sequences[pid] for pid in protein_ids]
    featurizer = TfidfKmerFeaturizer(
        k=config["features"]["kmer_k"],
        max_features=config["features"]["tfidf_max_features"],
    )
    X = featurizer.fit_transform(seq_list).toarray()

    # Load or create splits
    print("Loading splits...")
    folds_path = config["data"].get("folds", "data/folds.json")

    if os.path.exists(folds_path):
        splitter = HomologySplitter.load(folds_path)
    else:
        print("Creating new splits...")
        splitter = HomologySplitter(
            n_folds=config["data"]["folds"],
            identity_threshold=config["eval"].get("identity_threshold", 0.7),
        )
        splitter.fit(sequences)
        splitter.save(folds_path)

    # Create objective
    objective = CafaObjective(X=X, y=y, protein_ids=protein_ids, splitter=splitter)

    # Create study
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    # Optimize
    print(f"Running {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"Best F-Max: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Save best params
    best_params_path = os.path.join(
        os.path.dirname(storage) if storage else "experiments", "best_params.json"
    )

    with open(best_params_path, "w") as f:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "n_trials": len(study.trials),
            },
            f,
            indent=2,
        )

    print(f"Best params saved to {best_params_path}")

    return study


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--study-name", type=str, default="cafa6_optimization")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    config = load_config(args.config)

    storage = args.storage or config.get("optuna", {}).get("storage")

    run_optuna_search(
        config=config,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=storage,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
