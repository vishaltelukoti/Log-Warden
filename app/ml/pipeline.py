"""Inference pipeline for loading ML models and analyzing log input."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .models import MLModels

DEFAULT_MODEL_DIR = "models"
DEFAULT_DATA_FILE = "logs.csv"


def _get_project_root() -> Path:
    """
    Return the project root directory.

    Returns:
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parents[2]


def _get_data_file_path(filename: str = DEFAULT_DATA_FILE) -> Path:
    """
    Build the absolute path to the training dataset file.

    Args:
        filename: Name of the dataset file.

    Returns:
        Absolute path to the dataset file.
    """
    return _get_project_root() / filename


def _initialize_model_pipeline() -> MLModels:
    """
    Initialize the ML pipeline by loading pre-trained models.

    If model artifacts are missing, train models from the default dataset,
    save them, and reload them.

    Returns:
        A ready-to-use MLModels instance.

    Raises:
        FileNotFoundError: If neither trained models nor the training dataset exist.
    """
    model_manager = MLModels()

    try:
        model_manager.load(DEFAULT_MODEL_DIR)
        return model_manager
    except FileNotFoundError:
        data_file_path = _get_data_file_path()

        if not data_file_path.is_file():
            raise FileNotFoundError(
                f"Training data not found at {data_file_path}"
            )

        print("Pre-trained models not found. Training new models...")
        model_manager.train(str(data_file_path))
        model_manager.save(DEFAULT_MODEL_DIR)
        model_manager.load(DEFAULT_MODEL_DIR)

        return model_manager


ml_instance = _initialize_model_pipeline()


def analyze_log(log_text: str) -> Dict[str, float | int]:
    """
    Analyze a log message and return ML predictions.

    Args:
        log_text: Raw log text to analyze.

    Returns:
        A dictionary containing:
            - time_to_failure: Predicted time to failure
            - high_severity: Predicted severity label
    """
    time_to_failure, high_severity = ml_instance.predict(log_text)

    return {
        "time_to_failure": time_to_failure,
        "high_severity": high_severity,
    }