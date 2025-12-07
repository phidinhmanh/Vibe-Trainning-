# Utility functions for CAFA 6
# ============================

import os
import json
import random
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash for reproducibility."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file for data versioning."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_experiment_metadata(
    output_dir: str,
    config: Dict[str, Any],
    data_hashes: Dict[str, str],
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """Save experiment metadata for reproducibility."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "config": config,
        "data_hashes": data_hashes,
    }
    if additional_info:
        metadata.update(additional_info)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, "experiment_metadata.json")

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return output_path


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Automatic path fix for Kaggle vs Local
    # If we are in Kaggle or have the directory structure, override the config paths
    kaggle_input = Path("../kaggle/input")
    kaggle_train = kaggle_input / "Train"

    # Check if ../kaggle/input/Train exists
    if kaggle_train.exists():
        print(f"[INFO] Detected Kaggle input directory at {kaggle_input}")
        config["data"]["raw_dir"] = str(kaggle_input)
        config["data"]["train_sequences"] = str(kaggle_train / "train_sequences.fasta")
        config["data"]["train_terms"] = str(kaggle_train / "train_terms.tsv")
        config["data"]["train_taxonomy"] = str(kaggle_train / "train_taxonomy.tsv")
        config["data"]["test_sequences"] = str(
            kaggle_input / "Test" / "testsuperset.fasta"
        )
        config["data"]["go_obo"] = str(kaggle_train / "go-basic.obo")
        config["data"]["ia_weights"] = str(kaggle_input / "IA.tsv")
    else:
        # Fallback to local data/raw if we are not in Kaggle
        # This is useful if the config defaults to Kaggle but we are running locally
        local_raw = Path("data/raw")
        current_raw = Path(config["data"].get("raw_dir", "data/raw"))

        if local_raw.exists() and not current_raw.exists():
            print(
                f"[INFO] Configured raw_dir {current_raw} not found, falling back to local {local_raw}"
            )
            config["data"]["raw_dir"] = str(local_raw)
            config["data"]["train_sequences"] = str(local_raw / "train_sequences.fasta")
            config["data"]["train_terms"] = str(local_raw / "train_terms.tsv")
            config["data"]["train_taxonomy"] = str(local_raw / "train_taxonomy.tsv")
            config["data"]["test_sequences"] = str(local_raw / "testsuperset.fasta")
            # Updated resources layout vs old layout check could go here, but assuming standard
            config["data"]["go_obo"] = str(Path("resources/go-basic.obo"))
            config["data"]["ia_weights"] = str(Path("resources/IA.tsv"))

            # Check if resources exist, if not, maybe they are in data/raw (old layout)
            if (
                not Path(config["data"]["go_obo"]).exists()
                and (local_raw / "go-basic.obo").exists()
            ):
                config["data"]["go_obo"] = str(local_raw / "go-basic.obo")
            if (
                not Path(config["data"]["ia_weights"]).exists()
                and (local_raw / "IA.tsv").exists()
            ):
                config["data"]["ia_weights"] = str(local_raw / "IA.tsv")

    return config
