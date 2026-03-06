"""Pydantic request and response schemas for the Log Warden API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class LogRequest(BaseModel):
    """
    Schema for incoming log analysis requests.

    Attributes:
        log_text: Raw log message submitted for analysis.
    """

    log_text: str = Field(
        ...,
        min_length=1,
        description="Raw log message submitted for analysis",
    )

    @field_validator("log_text")
    @classmethod
    def validate_log_text(cls, value: str) -> str:
        """
        Ensure log text is not empty or whitespace.

        Args:
            value: Incoming log text.

        Returns:
            Validated log text.

        Raises:
            ValueError: If the log text is empty or contains only whitespace.
        """
        if not value.strip():
            raise ValueError("log_text must be a non-empty string")

        return value


class Prediction(BaseModel):
    """
    ML prediction results for the analyzed log.

    Attributes:
        high_severity: Indicates whether the log represents a high-severity issue.
        time_to_failure_minutes: Estimated time to failure in minutes.
    """

    high_severity: bool
    time_to_failure_minutes: float


class Analysis(BaseModel):
    """
    Log analysis details returned by the ML pipeline.

    Attributes:
        log_text: Original log text provided in the request.
        predictions: Prediction results produced by the ML model.
    """

    log_text: str
    predictions: Prediction


class Agent(BaseModel):
    """
    Metadata describing LLM agent decisions.

    Attributes:
        decision: Explanation of whether the LLM was triggered.
        model: Name of the LLM model used, if applicable.
    """

    decision: str
    model: Optional[str]


class Remediation(BaseModel):
    """
    LLM remediation output information.

    Attributes:
        generated: Whether a remediation script was generated.
        issue_category: Predicted issue category.
        confidence: Confidence score returned by the LLM pipeline.
        script_language: Programming language of the remediation script.
        script_lines: Generated script split into lines.
    """

    generated: bool
    issue_category: Optional[str]
    confidence: Optional[float]
    script_language: Optional[str]
    script_lines: Optional[List[str]] = None


class AnalyzeResponse(BaseModel):
    """
    Top-level API response schema for log analysis.

    Attributes:
        status: Request status indicator.
        analysis: ML analysis results.
        agent: LLM agent decision metadata.
        remediation: Generated remediation details.
    """

    status: str
    analysis: Analysis
    agent: Agent
    remediation: Remediation