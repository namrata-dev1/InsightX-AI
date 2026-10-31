"""Tests for configuration loading, cloud path resolution, and environment detection."""

from pathlib import Path
from insightx.config import (
    get_base_config,
    get_environment_name,
    get_model_config,
    resolve_artifacts_dir,
    resolve_data_dir,
)


def test_base_config_loading():
    """Ensures base configuration contains essential project keys and hybrid-cloud settings."""
    cfg = get_base_config()
    assert "project" in cfg
    assert "environments" in cfg
    assert cfg["project"]["name"] == "InsightX AI"
    assert cfg["project"]["architecture"] == "hybrid-cloud"


def test_model_config_loading():
    """Ensures model configuration defines required modeling sections."""
    cfg = get_model_config()
    assert "rfm" in cfg
    assert "segmentation" in cfg
    assert "repeat_purchase" in cfg
    assert "forecasting" in cfg


def test_path_resolvers():
    """Verifies that environment resolvers return valid Path objects."""
    env = get_environment_name()
    assert env in ["local", "kaggle", "colab"]

    data_dir = resolve_data_dir()
    assert isinstance(data_dir, Path)

    artifacts_dir = resolve_artifacts_dir()
    assert isinstance(artifacts_dir, Path)
