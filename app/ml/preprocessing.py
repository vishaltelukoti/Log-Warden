"""Log preprocessing utilities for ML feature extraction."""

from __future__ import annotations

import re
from typing import List

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
    text = re.sub(r"\d+\.\d+\.\d+\.\d+", "", text)  # IP addresses
    text = re.sub(r"\[.*?\]", "", text)  # bracketed metadata
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