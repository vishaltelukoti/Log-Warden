"""API endpoint tests for the Log Warden Flask application."""

from __future__ import annotations


def test_health_endpoint_returns_running_status(client) -> None:
    """
    Verify that the health endpoint returns a successful running status.
    """
    response = client.get("/health")
    response_data = response.get_json()

    assert response.status_code == 200
    assert response_data["status"] == "ok"


def test_analyze_endpoint_returns_successful_analysis_for_low_severity(
    client,
    mocker,
) -> None:
    """
    Verify that the analyze endpoint returns a valid success response
    for a low-severity prediction.
    """
    mocker.patch(
        "app.main.analyze_log",
        return_value={
            "high_severity": 0,
            "time_to_failure": 50.0,
        },
    )

    response = client.post("/analyze", json={"log_text": "INFO service started"})
    response_data = response.get_json()

    assert response.status_code == 200
    assert response_data["status"] == "success"
    assert "analysis" in response_data
    assert "predictions" in response_data["analysis"]
    assert "high_severity" in response_data["analysis"]["predictions"]
    assert response_data["analysis"]["predictions"]["high_severity"] is False


def test_analyze_endpoint_returns_bad_request_for_empty_log(client) -> None:
    """
    Verify that the analyze endpoint rejects an empty log string.
    """
    response = client.post("/analyze", json={"log_text": ""})

    assert response.status_code == 400


def test_analyze_endpoint_returns_bad_request_when_log_field_is_missing(
    client,
) -> None:
    """
    Verify that the analyze endpoint rejects requests missing the log_text field.
    """
    response = client.post("/analyze", json={})

    assert response.status_code == 400


def test_analyze_endpoint_returns_bad_request_for_invalid_log_type(client) -> None:
    """
    Verify that the analyze endpoint rejects invalid log_text data types.
    """
    response = client.post("/analyze", json={"log_text": 12345})

    assert response.status_code == 400


def test_analyze_endpoint_handles_large_log_payload(client, mocker) -> None:
    """
    Verify that the analyze endpoint handles large log inputs successfully.
    """
    mocker.patch(
        "app.main.analyze_log",
        return_value={
            "high_severity": 0,
            "time_to_failure": 120.0,
        },
    )

    large_log = "ERROR database connection failed " * 1000
    response = client.post("/analyze", json={"log_text": large_log})
    response_data = response.get_json()

    assert response.status_code == 200
    assert response_data["analysis"]["predictions"]["high_severity"] is False


def test_llm_is_not_called_for_low_severity_prediction(client, mocker) -> None:
    """
    Verify that remediation generation is not triggered for low-severity logs.
    """
    mocker.patch(
        "app.main.analyze_log",
        return_value={
            "high_severity": 0,
            "time_to_failure": 45.0,
        },
    )

    mocked_generate_remediation = mocker.patch(
        "app.main.llm_agent.generate_remediation"
    )

    client.post("/analyze", json={"log_text": "INFO application started"})

    mocked_generate_remediation.assert_not_called()


def test_llm_is_called_for_high_severity_prediction(client, mocker) -> None:
    """
    Verify that remediation generation is triggered for high-severity logs.
    """
    mocker.patch(
        "app.main.analyze_log",
        return_value={
            "high_severity": 1,
            "time_to_failure": 5.0,
        },
    )

    mocker.patch(
        "app.main.llm_agent.generate_remediation",
        return_value={
            "issue_category": "Database Connectivity Issue",
            "confidence": 0.85,
            "script": "print('restart service')",
        },
    )

    response = client.post("/analyze", json={"log_text": "ERROR database crashed"})
    response_data = response.get_json()

    assert response.status_code == 200
    assert response_data["remediation"]["generated"] is True
    assert (
        response_data["remediation"]["issue_category"]
        == "Database Connectivity Issue"
    )