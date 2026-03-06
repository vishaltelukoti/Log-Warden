"""Pytest configuration and shared fixtures for the Log Warden test suite."""

from __future__ import annotations
import sys
import pytest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from app.main import app

@pytest.fixture(scope="function")
def client():
    """
    Provide a Flask test client for API endpoint testing.

    This fixture enables testing mode in the Flask application and
    yields a test client instance that can be used to simulate HTTP
    requests to the API.

    Yields:
        FlaskClient: A Flask test client instance.
    """
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client