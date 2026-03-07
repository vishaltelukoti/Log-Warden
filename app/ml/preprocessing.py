"""Log preprocessing utilities for ML feature extraction."""

from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import spacy

# Load SpaCy model once at module initialization
NLP_MODEL_NAME = "en_core_web_sm"
nlp = spacy.load(NLP_MODEL_NAME)


LOG_LEVEL_TOKENS = {
    "CRITICAL": "loglevel_critical",
    "ERROR": "loglevel_error",
    "WARNING": "loglevel_warning",
    "INFO": "loglevel_info",
    "FATAL": "loglevel_fatal",
}

# Keywords that signal high-severity events
HIGH_SEVERITY_KEYWORDS = [
    "nullpointer", "exception", "crash", "fatal", "critical",
    "outofmemory", "segfault", "unreachable", "timeout", "refused",
    "exhausted", "lost", "failure", "error",
]


def _extract_log_level(text: str) -> str:
    """
    Extract the log level token from the original log message.

    Args:
        text: Original log text.

    Returns:
        Encoded log level token or an empty string if not detected.
    """
    for level, token in LOG_LEVEL_TOKENS.items():
        if level in text:
            return token
    return ""


def _remove_noise(text: str) -> str:
    """
    Remove timestamps, IP addresses, and bracketed metadata.

    Args:
        text: Raw log text.

    Returns:
        Cleaned text with noise removed.
    """
    text = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "", text)  # timestamps
    text = re.sub(r"\d+\.\d+\.\d+\.\d+", "", text)                    # IP addresses
    text = re.sub(r"\[.*?\]", "", text)                                # bracketed metadata
    return text.lower()


def _lemmatize_tokens(text: str) -> List[str]:
    """
    Tokenize and lemmatize text using SpaCy.

    Args:
        text: Pre-cleaned text.

    Returns:
        List of processed tokens.
    """
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]

    return tokens


def clean_log(text: str) -> str:
    """
    Clean and normalize log text for machine learning models.

    Processing steps:
        1. Extract log level token
        2. Remove timestamps, IP addresses, and metadata
        3. Lowercase normalization
        4. Lemmatization with stop-word removal
        5. Append log level feature

    Args:
        text: Raw log message.

    Returns:
        Processed text string suitable for vectorization.
    """
    log_level_token = _extract_log_level(text)

    cleaned_text = _remove_noise(text)
    tokens = _lemmatize_tokens(cleaned_text)

    if log_level_token:
        tokens.append(log_level_token)

    return " ".join(tokens)


def extract_numeric_features(text: str) -> Dict[str, float]:
    """
    Extract engineered numeric features from a raw log message.

    These features complement TF-IDF by capturing structural and
    statistical properties of the log that pure text vectorization misses.

    Features extracted:
        - log_length: Total character count of the log message.
        - word_count: Number of whitespace-separated tokens.
        - contains_error: 1 if the log contains ERROR or FATAL, else 0.
        - contains_exception: 1 if the log references an exception or crash, else 0.
        - severity_keyword_count: Number of high-severity keywords present.
        - has_stack_trace: 1 if the log looks like a stack trace (contains 'at ' or '::'), else 0.
        - digit_ratio: Ratio of digit characters to total characters.

    Args:
        text: Raw log message.

    Returns:
        Dictionary mapping feature names to numeric values.
    """
    lower_text = text.lower()

    log_length = float(len(text))
    word_count = float(len(text.split()))
    contains_error = float(any(kw in lower_text for kw in ("error", "fatal", "critical")))
    contains_exception = float(any(kw in lower_text for kw in ("exception", "crash", "traceback")))
    severity_keyword_count = float(sum(1 for kw in HIGH_SEVERITY_KEYWORDS if kw in lower_text))
    has_stack_trace = float(bool(re.search(r"\bat\s+\w|::", text)))
    digit_ratio = float(sum(c.isdigit() for c in text)) / max(log_length, 1.0)

    return {
        "log_length": log_length,
        "word_count": word_count,
        "contains_error": contains_error,
        "contains_exception": contains_exception,
        "severity_keyword_count": severity_keyword_count,
        "has_stack_trace": has_stack_trace,
        "digit_ratio": digit_ratio,
    }


def get_numeric_feature_array(text: str) -> np.ndarray:
    """
    Return engineered numeric features as a 1-D numpy array.

    The order of features is fixed and matches the column order used
    during training. Do not change the order without retraining.

    Args:
        text: Raw log message.

    Returns:
        1-D numpy array of numeric feature values.
    """
    features = extract_numeric_features(text)
    return np.array(list(features.values()), dtype=np.float64)