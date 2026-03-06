"""Model training script for the Log Warden ML pipeline."""

from __future__ import annotations

from app.ml.models import MLModels

DEFAULT_DATASET_PATH = "logs.csv"
DEFAULT_MODEL_DIR = "models"


def train_models(dataset_path: str = DEFAULT_DATASET_PATH) -> None:
    """
    Train ML models and persist them to disk.

    Args:
        dataset_path: Path to the dataset CSV file used for training.
    """
    model_manager = MLModels()
    model_manager.train(dataset_path)
    model_manager.save(DEFAULT_MODEL_DIR)


def main() -> None:
    """
    Entry point for the training script.
    """
    train_models(DEFAULT_DATASET_PATH)
    print("Model training completed successfully.")


if __name__ == "__main__":
    main()