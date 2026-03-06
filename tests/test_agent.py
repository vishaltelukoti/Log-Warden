"""Unit tests for the Groq-based LLM agent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent.groq_agent import LLMAgent


def _build_mock_completion_response(content: str) -> SimpleNamespace:
    """
    Build a mock Groq completion response object.

    Args:
        content: Mock message content returned by the LLM.

    Returns:
        A nested object that mimics the Groq completion response structure.
    """
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_generate_remediation_returns_expected_response_structure() -> None:
    """
    Verify that remediation generation returns the expected dictionary structure.
    """
    agent = LLMAgent()

    response = agent.generate_remediation("ERROR disk failure")

    assert isinstance(response, dict)
    assert "script" in response
    assert "issue_category" in response
    assert "confidence" in response


def test_generate_remediation_returns_fallback_on_llm_failure(mocker) -> None:
    """
    Verify that a fallback response is returned when the Groq API call fails.
    """
    agent = LLMAgent()

    mocker.patch.object(
        agent.client.chat.completions,
        "create",
        side_effect=Exception("API failed"),
    )

    response = agent.generate_remediation("ERROR disk failure")

    assert isinstance(response, dict)
    assert response["script"].startswith("Unable to generate remediation")


def test_generate_remediation_removes_markdown_code_fences(mocker) -> None:
    """
    Verify that fenced markdown code blocks are stripped from the generated script.
    """
    agent = LLMAgent()

    mocker.patch.object(
        agent.client.chat.completions,
        "create",
        return_value=_build_mock_completion_response(
            "```python\nimport logging\n\nlogging.info('hi')\n```"
        ),
    )

    response = agent.generate_remediation("ERROR disk failure")

    assert "```" not in response["script"]
    assert "import logging" in response["script"]