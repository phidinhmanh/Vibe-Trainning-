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

    # Robust data discovery using os.walk
    # We look for key files in potential root directories
    potential_roots = ["/kaggle/input", "data/raw"]
    found_root = None

    # Maps config keys to filenames we need to find
    required_files = {
        "train_sequences": "train_sequences.fasta",
        "train_terms": "train_terms.tsv",
        "train_taxonomy": "train_taxonomy.tsv",
        "test_sequences": "testsuperset.fasta",
        "go_obo": "go-basic.obo",
        "ia_weights": "IA.tsv",
    }

    found_paths = {}

    for root in potential_roots:
        if os.path.exists(root):
            print(f"[INFO] Searching for data in {root}...")
            # Walk through the directory to find files
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    for key, target_name in required_files.items():
                        if key not in found_paths and filename == target_name:
                            full_path = os.path.join(dirpath, filename)
                            found_paths[key] = full_path
                            # If we found a file, this root is likely the correct one
                            if found_root is None:
                                found_root = root
                                config["data"]["raw_dir"] = root

    # Update config with found paths
    if found_paths:
        print(
            f"[INFO] Found {len(found_paths)}/{len(required_files)} required data files"
        )
        for key, path in found_paths.items():
            config["data"][key] = path
            print(f"  - {key}: {path}")
    else:
        print(
            "[WARNING] No data files found in standard locations. Using default config paths."
        )

    return config
