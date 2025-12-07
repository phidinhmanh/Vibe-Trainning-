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
        return yaml.safe_load(f)
