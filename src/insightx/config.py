"""Configuration loader and cloud environment resolver for InsightX AI.

Supports seamless execution across:
1. Local Workstation (Antigravity)
2. Kaggle Notebooks (/kaggle/input/brazilian-ecommerce)
3. Google Colab (/content/data and /content/drive)
"""

import os
from pathlib import Path
import sys
from typing import Any, Dict
import yaml

from insightx.utils.logger import get_logger

logger = get_logger(__name__)

# Repository root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = ROOT_DIR / "configs"


def is_kaggle_environment() -> bool:
    """Detects if the code is running inside a Kaggle Notebook."""
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or Path("/kaggle/input/brazilian-ecommerce").exists()
    )


def is_colab_environment() -> bool:
    """Detects if the code is running inside Google Colab."""
    return "google.colab" in sys.modules or Path("/content").exists()


def get_environment_name() -> str:
    """Returns the name of the current runtime environment."""
    if is_kaggle_environment():
        return "kaggle"
    elif is_colab_environment():
        return "colab"
    return "local"


def resolve_data_dir() -> Path:
    """Resolves the raw dataset directory according to the runtime environment.

    Resolution hierarchy:
    1. Explicit OLIST_DATA_DIR environment variable if provided.
    2. Kaggle native dataset mount: /kaggle/input/brazilian-ecommerce
    3. Google Colab data path: /content/data (or /content)
    4. Local fallback: ROOT_DIR / data / raw

    Returns:
        Path: Existing or intended directory for raw Olist CSVs.
    """
    # 1. Custom override via environment variable
    env_override = os.environ.get("OLIST_DATA_DIR")
    if env_override:
        path = Path(env_override)
        if path.exists():
            logger.info(f"Using environment variable data directory: {path}")
            return path

    # 2. Kaggle native mount
    kaggle_path = Path("/kaggle/input/brazilian-ecommerce")
    if kaggle_path.exists():
        logger.info("Detected Kaggle environment. Mounted dataset at: /kaggle/input/brazilian-ecommerce")
        return kaggle_path

    # 3. Google Colab path
    colab_path = Path("/content/data")
    if colab_path.exists():
        logger.info("Detected Google Colab environment. Using: /content/data")
        return colab_path

    # Check alternative Colab root if unzipped directly to /content
    if Path("/content/olist_orders_dataset.csv").exists():
        logger.info("Detected Olist tables directly in Google Colab root: /content")
        return Path("/content")

    # 4. Local workspace fallback
    local_path = ROOT_DIR / "data" / "raw"
    logger.info(f"Using local data directory: {local_path}")
    return local_path


def resolve_artifacts_dir() -> Path:
    """Resolves the output directory for model weights, plots, and reports.

    Returns:
        Path: Target artifacts directory.
    """
    # 1. Google Colab with mounted Google Drive
    gdrive_artifacts = Path("/content/drive/MyDrive/InsightX-AI/artifacts")
    if is_colab_environment() and Path("/content/drive/MyDrive").exists():
        gdrive_artifacts.mkdir(parents=True, exist_ok=True)
        return gdrive_artifacts

    # 2. Kaggle working directory
    if is_kaggle_environment():
        kaggle_output = Path("/kaggle/working/artifacts")
        kaggle_output.mkdir(parents=True, exist_ok=True)
        return kaggle_output

    # 3. Colab without Drive or Local workspace
    if is_colab_environment():
        colab_local = Path("/content/artifacts")
        colab_local.mkdir(parents=True, exist_ok=True)
        return colab_local

    local_artifacts = ROOT_DIR / "artifacts"
    local_artifacts.mkdir(parents=True, exist_ok=True)
    return local_artifacts


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """Loads a YAML configuration file safely.

    Args:
        filepath: Path to the YAML file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if not filepath.exists():
        logger.warning(f"Config file not found at {filepath}")
        return {}

    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_base_config() -> Dict[str, Any]:
    """Returns the base project configuration."""
    return load_yaml(CONFIG_DIR / "base_config.yaml")


def get_model_config() -> Dict[str, Any]:
    """Returns the machine learning and analytics configuration."""
    return load_yaml(CONFIG_DIR / "model_config.yaml")
