"""Model training script for the Log Warden ML pipeline."""

from __future__ import annotations

"""
Model Selection Explanation

Logistic Regression is used for predicting high severity events because the
task is a binary classification problem and Logistic Regression performs well
with high-dimensional sparse TF-IDF features derived from log text.

RandomForestRegressor is used for predicting time-to-failure. Log patterns
often contain non-linear relationships between features and system failures.
Random forests capture these non-linear interactions using an ensemble of
decision trees, leading to more robust predictions compared to linear models.

Generalization was ensured using an 80/20 train-test split and cross-validation.
"""


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