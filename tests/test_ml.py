"""Unit tests for the ML log analysis pipeline."""

from __future__ import annotations
import pytest 
from app.ml.pipeline import analyze_log


def test_analyze_log_returns_expected_keys() -> None:
    """
    Verify that analyze_log returns the required prediction fields.
    """
    result = analyze_log("ERROR database connection failed")

    assert isinstance(result, dict)
    assert "high_severity" in result
    assert "time_to_failure" in result


def test_high_severity_prediction_is_binary() -> None:
    """
    Verify that the severity prediction is binary (0 or 1).
    """
    result = analyze_log("ERROR disk failure detected")

    severity_value = result["high_severity"]

    assert severity_value in (0, 1)


def test_time_to_failure_prediction_is_float() -> None:
    """
    Verify that the predicted time to failure is a float value.
    """
    result = analyze_log("INFO application started")

    time_prediction = result["time_to_failure"]

    assert isinstance(time_prediction, float)