"""Machine learning model training, persistence, and inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    r2_score,
    recall_score,
    mean_absolute_error,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

from .preprocessing import clean_log, get_numeric_feature_array

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
        # RandomForestRegressor used instead of LinearRegression 
        # due to significantly better R² on this dataset (non-linear log relationships).
        self.reg_model: Optional[RandomForestRegressor] = None
        self.clf_model: Optional[LogisticRegression] = None

    def train(self, data_path: str = DEFAULT_DATA_PATH) -> None:
        """
        Train the regression and classification models from a CSV dataset.

        Feature matrix is a horizontal stack of TF-IDF text features and
        engineered numeric features (log length, error flags, keyword counts, etc.).

        Expected CSV columns:
            - log_text: Raw log message string.
            - time_to_failure: Numeric regression target (minutes until failure).
            - high_severity: Binary classification target (0 or 1).

        Args:
            data_path: Path to the training dataset CSV file.
        """
        dataset = pd.read_csv(data_path)
        dataset["clean_text"] = dataset["log_text"].apply(clean_log)

        # TF-IDF features 
        self.vectorizer = TfidfVectorizer(max_features=DEFAULT_MAX_FEATURES)
        tfidf_features = self.vectorizer.fit_transform(dataset["clean_text"])

        # Engineered numeric features 
        numeric_features = np.vstack(
            dataset["log_text"].apply(get_numeric_feature_array).values
        )

        # Combined feature matrix (TF-IDF + numeric)
        from scipy.sparse import csr_matrix
        features = hstack([tfidf_features, csr_matrix(numeric_features)])

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

        Builds the same combined feature vector (TF-IDF + numeric) used
        during training before passing to both models.

        Args:
            log_text: Raw log message.

        Returns:
            A tuple containing:
                - predicted time to failure (float, minutes)
                - predicted severity class (int, 0 or 1)

        Raises:
            ValueError: If the models or vectorizer are not loaded or trained.
        """
        self._validate_models_are_ready()

        from scipy.sparse import csr_matrix, hstack as sp_hstack

        cleaned_text = clean_log(log_text)
        tfidf_input = self.vectorizer.transform([cleaned_text])

        numeric_input = csr_matrix(get_numeric_feature_array(log_text).reshape(1, -1))
        input_features = sp_hstack([tfidf_input, numeric_input])

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

        Classification metrics include accuracy, precision, recall, F1, and
        cross-validation accuracy to demonstrate generalization.

        Regression metrics include R² and MAE on both train and test sets.

        Args:
            features: Full combined feature matrix (TF-IDF + numeric).
            x_train: Training feature matrix.
            x_test: Test feature matrix.
            y_reg_train: Training targets for regression.
            y_reg_test: Test targets for regression.
            y_clf_train: Training targets for classification.
            y_clf_test: Test targets for classification.
            classification_target: Full classification target series.
        """
        y_clf_test_pred = self.clf_model.predict(x_test)
        y_clf_train_pred = self.clf_model.predict(x_train)

        train_accuracy = accuracy_score(y_clf_train, y_clf_train_pred)
        test_accuracy = accuracy_score(y_clf_test, y_clf_test_pred)
        test_precision = precision_score(y_clf_test, y_clf_test_pred, zero_division=0)
        test_recall = recall_score(y_clf_test, y_clf_test_pred, zero_division=0)
        test_f1 = f1_score(y_clf_test, y_clf_test_pred, zero_division=0)
        cross_validation_accuracy = np.mean(
            cross_val_score(self.clf_model, features, classification_target, cv=5)
        )

        train_r2 = r2_score(y_reg_train, self.reg_model.predict(x_train))
        test_r2 = r2_score(y_reg_test, self.reg_model.predict(x_test))
        test_mae = mean_absolute_error(y_reg_test, self.reg_model.predict(x_test))

        print("===== Logistic Regression Evaluation =====")
        print(f"  Train Accuracy:            {train_accuracy:.4f}")
        print(f"  Test Accuracy:             {test_accuracy:.4f}")
        print(f"  Test Precision:            {test_precision:.4f}")
        print(f"  Test Recall:               {test_recall:.4f}")
        print(f"  Test F1 Score:             {test_f1:.4f}")
        print(f"  Cross-validation Accuracy: {cross_validation_accuracy:.4f}")
        print()
        print("  Full Classification Report:")
        print(classification_report(y_clf_test, y_clf_test_pred, zero_division=0))

        print("===== RandomForest Regressor Evaluation =====")
        print(f"  Train R²:  {train_r2:.4f}")
        print(f"  Test R²:   {test_r2:.4f}")
        print(f"  Test MAE:  {test_mae:.4f} minutes")