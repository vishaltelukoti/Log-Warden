"""Groq-powered LLM agent for issue classification and remediation generation."""

from __future__ import annotations

import os
import re
from typing import Any, Dict

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_FAILURE_MESSAGE = "Unable to generate remediation script at this time."
DEFAULT_CONFIDENCE = 0.0
SUCCESS_CONFIDENCE = 0.85
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
MINIMUM_SCRIPT_LENGTH = 20


class LLMAgent:
    """Agent responsible for classifying log issues and generating remediation scripts."""

    def __init__(self, api_key: str | None = None) -> None:
        """
        Initialize the LLM agent.

        Args:
            api_key: Optional Groq API key. If not provided, the value is read
                from the GROQ_API_KEY environment variable.
        """
        resolved_api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=resolved_api_key)

    def detect_issue_type(self, log_text: str) -> str:
        """
        Detect the issue category from raw log text using keyword heuristics.

        Args:
            log_text: Raw log content.

        Returns:
            A human-readable issue category.
        """
        normalized_log_text = log_text.lower()

        if "memory" in normalized_log_text:
            return "Memory Issue"
        if "disk" in normalized_log_text or "/dev/" in normalized_log_text:
            return "Disk Issue"
        if "database" in normalized_log_text:
            return "Database Connectivity Issue"
        if (
            "connection refused" in normalized_log_text
            or "timeout" in normalized_log_text
        ):
            return "Network Issue"
        if "crash" in normalized_log_text or "exception" in normalized_log_text:
            return "Application Crash"
        if "login" in normalized_log_text or "authentication" in normalized_log_text:
            return "Authentication Issue"
        if "slow" in normalized_log_text or "response time" in normalized_log_text:
            return "Performance Degradation"

        return "Generic System Issue"

    def generate_remediation(self, log_text: str) -> Dict[str, Any]:
        """
        Generate a safe Python remediation script based on log text.

        Args:
            log_text: Raw application or system log content.

        Returns:
            A dictionary containing:
                - issue_category: Detected issue category
                - confidence: Confidence score for the generated output
                - script: Generated remediation script or fallback message
        """
        issue_category = self.detect_issue_type(log_text)
        prompt = self._build_remediation_prompt(
            log_text=log_text,
            issue_category=issue_category,
        )

        try:
            completion = self.client.chat.completions.create(
                model=GROQ_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "Return ONLY valid Python code. No explanations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=900,
            )

            raw_response = completion.choices[0].message.content or ""
            script = self._extract_python_code(raw_response)
            script = self._ensure_main_footer(script)

            if not self._is_valid_script(script):
                return self._build_failure_response(issue_category)

            return {
                "issue_category": issue_category,
                "confidence": SUCCESS_CONFIDENCE,
                "script": script,
            }

        except Exception:
            return self._build_failure_response(issue_category)

    def _build_remediation_prompt(self, log_text: str, issue_category: str) -> str:
        """
        Build the prompt used for remediation script generation.

        Args:
            log_text: Raw log content.
            issue_category: Predicted issue category.

        Returns:
            Prompt string for the LLM.
        """
        return f"""
You are an expert DevOps automation assistant.

Predicted Issue Category: {issue_category}

Log:
{log_text}

Generate a SAFE Python remediation script.

Rules:
- Output ONLY Python code (no markdown, no explanations).
- Validate system state before any action.
- Use logging + try/except.
- Avoid hardcoded PIDs.
- Avoid destructive commands.

Forbidden:
kill -9, SIGKILL, rm -rf, fsck -f, systemctl restart

If a restart is needed, use a graceful approach (e.g., notify/alert or suggest).
If you define main(), end with:
if __name__ == "__main__":
    main()
""".strip()

    def _extract_python_code(self, response_text: str) -> str:
        """
        Extract Python code from an LLM response.

        Extraction priority:
            1. Fenced code blocks
            2. Heuristic match from likely Python statement start

        Args:
            response_text: Raw LLM response.

        Returns:
            Extracted Python code only.
        """
        if not response_text:
            return ""

        fenced_code_match = re.search(
            r"```(?:python)?\s*(.*?)```",
            response_text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced_code_match:
            extracted_code = fenced_code_match.group(1).strip()
        else:
            cleaned_text = response_text.strip()
            cleaned_text = re.sub(
                r"^\s*python\s*\n",
                "",
                cleaned_text,
                flags=re.IGNORECASE,
            )

            python_start_match = re.search(
                r"(^|\n)(import |from |def |class |if |try:|for |while |with |#)",
                cleaned_text,
            )

            if python_start_match:
                extracted_code = cleaned_text[python_start_match.start():].strip()
            else:
                extracted_code = cleaned_text

        extracted_code = re.split(
            r"\n\s*(Explanation:|Notes?:|Here('| i)s|This script)",
            extracted_code,
            maxsplit=1,
        )[0].strip()

        return extracted_code

    def _ensure_main_footer(self, code: str) -> str:
        """
        Ensure the script includes a runnable main guard when appropriate.

        The main guard is added only if a main() function exists and the
        __name__ == "__main__" block is missing.

        Args:
            code: Generated Python code.

        Returns:
            Python code with main guard appended when needed.
        """
        if not code:
            return code

        has_main_function = re.search(r"\ndef\s+main\s*\(", code) is not None
        has_main_guard = "__name__" in code and "__main__" in code

        if has_main_function and not has_main_guard:
            code += '\n\nif __name__ == "__main__":\n    main()\n'

        return code

    def _is_valid_script(self, code: str) -> bool:
        """
        Validate whether the generated script is usable.

        Args:
            code: Extracted Python code.

        Returns:
            True if the script appears usable, otherwise False.
        """
        return bool(code and len(code.strip()) >= MINIMUM_SCRIPT_LENGTH)

    def _build_failure_response(self, issue_category: str) -> Dict[str, Any]:
        """
        Build a standardized failure response.

        Args:
            issue_category: Detected issue category.

        Returns:
            Failure response payload.
        """
        return {
            "issue_category": issue_category,
            "confidence": DEFAULT_CONFIDENCE,
            "script": DEFAULT_FAILURE_MESSAGE,
        }