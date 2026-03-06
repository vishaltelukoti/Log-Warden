"""Machine learning model training, persistence, and inference utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from .preprocessing import clean_log

DEFAULT_DATA_PATH = "logs.csv"
DEFAULT_MODEL_DIR = "models"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2
DEFAULT_MAX_FEATURES = 500
DEFAULT_REGRESSOR_ESTIMATORS = 200
DEFAULT_CLASSIFIER_MAX_ITER = 1000


class MLModels:
    """Manage text vectorization, model training, persistence, and prediction."""

    def __init__(self) -> None:
        """Initialize empty model and vectorizer instances."""
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.reg_model: Optional[RandomForestRegressor] = None
        self.clf_model: Optional[LogisticRegression] = None

    def train(self, data_path: str = DEFAULT_DATA_PATH) -> None:
        """
        Train the regression and classification models from a CSV dataset.

        Expected CSV columns:
            - log_text
            - time_to_failure
            - high_severity

        Args:
            data_path: Path to the training dataset CSV file.
        """
        dataset = pd.read_csv(data_path)
        dataset["clean_text"] = dataset["log_text"].apply(clean_log)

        self.vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_FEATURES)
        features = self.vectorizer.fit_transform(dataset["clean_text"])

        regression_target = dataset["time_to_failure"]
        classification_target = dataset["high_severity"]

        (
            x_train,
            x_test,
            y_reg_train,
            y_reg_test,
            y_clf_train,
            y_clf_test,
        ) = train_test_split(
            features,
            regression_target,
            classification_target,
            test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE,
            stratify=classification_target,
        )

        self.reg_model = RandomForestRegressor(
            n_estimators=DEFAULT_REGRESSOR_ESTIMATORS,
            random_state=DEFAULT_RANDOM_STATE,
        )
        self.clf_model = LogisticRegression(
            max_iter=DEFAULT_CLASSIFIER_MAX_ITER,
            class_weight="balanced",
        )

        self.reg_model.fit(x_train, y_reg_train)
        self.clf_model.fit(x_train, y_clf_train)

        self._print_training_metrics(
            features=features,
            x_train=x_train,
            x_test=x_test,
            y_reg_train=y_reg_train,
            y_reg_test=y_reg_test,
            y_clf_train=y_clf_train,
            y_clf_test=y_clf_test,
            classification_target=classification_target,
        )

    def save(self, folder: str = DEFAULT_MODEL_DIR) -> None:
        """
        Save the trained vectorizer and models to disk.

        Args:
            folder: Directory where model artifacts will be stored.

        Raises:
            ValueError: If the models or vectorizer are not trained yet.
        """
        self._validate_models_are_ready()

        output_dir = Path(folder)
        output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.vectorizer, output_dir / "vectorizer.pkl")
        joblib.dump(self.reg_model, output_dir / "regressor.pkl")
        joblib.dump(self.clf_model, output_dir / "classifier.pkl")

        print("Models saved successfully!")

    def load(self, folder: str = DEFAULT_MODEL_DIR) -> None:
        """
        Load the vectorizer and trained models from disk.

        Args:
            folder: Directory containing model artifacts.

        Raises:
            FileNotFoundError: If the model directory does not exist.
        """
        project_root = Path(__file__).resolve().parents[2]
        model_dir = project_root / folder

        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        self.vectorizer = joblib.load(model_dir / "vectorizer.pkl")
        self.reg_model = joblib.load(model_dir / "regressor.pkl")
        self.clf_model = joblib.load(model_dir / "classifier.pkl")

        print("Models loaded successfully!")

    def predict(self, log_text: str) -> Tuple[float, int]:
        """
        Predict time to failure and severity from raw log text.

        Args:
            log_text: Raw log message.

        Returns:
            A tuple containing:
                - predicted time to failure
                - predicted severity class

        Raises:
            ValueError: If the models or vectorizer are not loaded or trained.
        """
        self._validate_models_are_ready()

        cleaned_text = clean_log(log_text)
        input_features = self.vectorizer.transform([cleaned_text])

        predicted_time_to_failure = float(self.reg_model.predict(input_features)[0])
        predicted_time_to_failure = max(0.0, predicted_time_to_failure)

        predicted_severity = int(self.clf_model.predict(input_features)[0])

        return round(predicted_time_to_failure, 2), predicted_severity

    def _validate_models_are_ready(self) -> None:
        """
        Ensure the vectorizer and models are available for saving or prediction.

        Raises:
            ValueError: If any required component is missing.
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer is not initialized. Train or load models first.")

        if self.reg_model is None:
            raise ValueError("Regression model is not initialized. Train or load models first.")

        if self.clf_model is None:
            raise ValueError("Classification model is not initialized. Train or load models first.")

    def _print_training_metrics(
        self,
        features,
        x_train,
        x_test,
        y_reg_train,
        y_reg_test,
        y_clf_train,
        y_clf_test,
        classification_target,
    ) -> None:
        """
        Print training and evaluation metrics for both models.

        Args:
            features: Full feature matrix.
            x_train: Training feature matrix.
            x_test: Test feature matrix.
            y_reg_train: Training targets for regression.
            y_reg_test: Test targets for regression.
            y_clf_train: Training targets for classification.
            y_clf_test: Test targets for classification.
            classification_target: Full classification target series.
        """
        train_accuracy = accuracy_score(y_clf_train, self.clf_model.predict(x_train))
        test_accuracy = accuracy_score(y_clf_test, self.clf_model.predict(x_test))
        cross_validation_accuracy = np.mean(
            cross_val_score(self.clf_model, features, classification_target, cv=5)
        )

        train_r2 = r2_score(y_reg_train, self.reg_model.predict(x_train))
        test_r2 = r2_score(y_reg_test, self.reg_model.predict(x_test))

        print("===== Logistic Regression Evaluation =====")
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Cross-validation Accuracy: {cross_validation_accuracy}")

        print("\n===== RandomForestRegression Evaluation =====")
        print(f"Train R2: {train_r2}")
        print(f"Test R2: {test_r2}")