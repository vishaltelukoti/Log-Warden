"""Flask application entry point for log analysis and remediation generation."""

from __future__ import annotations

from flask import Flask, jsonify, request
from pydantic import ValidationError
from dotenv import load_dotenv

from app.agent.groq_agent import LLMAgent
from app.ml.pipeline import analyze_log
from app.schemas import LogRequest

load_dotenv()

LLM_MODEL_NAME = "llama-3.1-8b-instant"

app = Flask(__name__)
llm_agent = LLMAgent()


@app.route("/health", methods=["GET"])
def health() -> tuple:
    """
    Health check endpoint.

    Returns:
        JSON response indicating service status.
    """
    return jsonify({"status": "ok"}), 200


@app.route("/analyze", methods=["POST"])
def analyze() -> tuple:
    """
    Analyze a log message and optionally generate remediation guidance.

    Workflow:
        1. Validate the incoming request payload.
        2. Run the ML pipeline to predict severity and time to failure.
        3. Trigger the LLM agent only for high-severity predictions.
        4. Return a structured JSON response.

    Returns:
        A Flask JSON response with analysis and remediation details.
    """
    try:
        request_payload = request.get_json() or {}
        validated_request = LogRequest(**request_payload)

        prediction_result = analyze_log(validated_request.log_text)
        high_severity = prediction_result["high_severity"]
        time_to_failure = prediction_result["time_to_failure"]

        remediation_response = _build_remediation_response(
            log_text=validated_request.log_text,
            high_severity=high_severity,
        )

        response = {
            "status": "success",
            "analysis": {
                "log_text": validated_request.log_text,
                "predictions": {
                    "high_severity": bool(high_severity),
                    "time_to_failure_minutes": round(time_to_failure, 2),
                },
            },
            "agent": {
                "decision": remediation_response["decision"],
                "model": remediation_response["model"],
            },
            "remediation": {
                "generated": high_severity == 1,
                "issue_category": remediation_response["issue_category"],
                "confidence": remediation_response["confidence"],
                "script_language": remediation_response["script_language"],
                "script_lines": remediation_response["script_lines"],
            },
        }

        return jsonify(response), 200

    except ValidationError as validation_error:
        return jsonify(
            {
                "status": "error",
                "message": validation_error.errors(),
            }
        ), 400

    except Exception as exc:
        return jsonify(
            {
                "status": "error",
                "message": str(exc),
            }
        ), 500


def _build_remediation_response(log_text: str, high_severity: int) -> dict:
    """
    Build remediation metadata and generated script details.

    Args:
        log_text: Raw log text received in the request.
        high_severity: Predicted severity flag from the ML model.

    Returns:
        A dictionary containing agent decision details and remediation output.
    """
    if high_severity != 1:
        return {
            "decision": "LLM not triggered because high_severity == 0",
            "model": None,
            "issue_category": None,
            "confidence": None,
            "script_language": None,
            "script_lines": None,
        }

    remediation_data = llm_agent.generate_remediation(log_text)
    script = remediation_data.get("script")
    script_lines = script.splitlines() if script else None

    return {
        "decision": "LLM triggered because high_severity == 1",
        "model": LLM_MODEL_NAME,
        "issue_category": remediation_data.get("issue_category"),
        "confidence": remediation_data.get("confidence"),
        "script_language": "python",
        "script_lines": script_lines,
    }


if __name__ == "__main__":
    app.run(debug=True)