"""Synthetic dataset generator for Log Warden training data."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import List, Tuple

import pandas as pd

DEFAULT_NUM_ENTRIES = 500
DEFAULT_OUTPUT_FILE = "logs.csv"
DEFAULT_NOISE_RATIO = 0.05
DEFAULT_GENERATED_ENTRIES = 600

SEVERITY_LEVELS = ["INFO", "WARN", "ERROR"]
SEVERITY_WEIGHTS = [0.45, 0.30, 0.25]

INFO_MESSAGES = [
    "Application started successfully",
    "Health check passed",
    "Scheduled job completed",
    "User login successful",
    "Configuration loaded",
    "Background sync finished",
    "Service heartbeat received",
]

WARN_MESSAGES = [
    "Slow response detected in API",
    "High memory usage detected",
    "Cache miss rate increasing",
    "Disk usage above 80%",
    "Thread pool nearing capacity",
    "Temporary DB failover triggered",
    "Latency spike observed",
]

ERROR_MESSAGES = [
    "NullPointerException in module X",
    "Database connection lost",
    "TimeoutException in service Y",
    "OutOfMemoryError occurred",
    "Service crashed unexpectedly",
    "Connection pool exhausted",
    "Segmentation fault detected",
    "Primary node unreachable",
]


def generate_log_entry(base_time: datetime) -> Tuple[str, int, int]:
    """
    Generate a single synthetic log entry with labels.

    Args:
        base_time: Timestamp to use for the generated log entry.

    Returns:
        A tuple containing:
            - log_text: Generated log message
            - time_to_failure: Simulated time to failure in minutes
            - high_severity: Binary severity label
    """
    severity_level = random.choices(
        SEVERITY_LEVELS,
        weights=SEVERITY_WEIGHTS,
        k=1,
    )[0]

    if severity_level == "INFO":
        message = random.choice(INFO_MESSAGES)
        time_to_failure = random.randint(80, 200)
        high_severity = 1 if random.random() < 0.02 else 0

    elif severity_level == "WARN":
        message = random.choice(WARN_MESSAGES)
        high_severity = _determine_warn_severity(message)
        time_to_failure = random.randint(20, 90)

    else:
        message = random.choice(ERROR_MESSAGES)
        high_severity = 1 if random.random() < 0.85 else 0
        time_to_failure = random.randint(0, 30)

    message = _apply_message_variation(message)
    time_to_failure = _apply_time_noise(time_to_failure)

    timestamp = base_time.strftime("%Y-%m-%d %H:%M:%S")
    log_text = f"{timestamp} {severity_level} [service] {message}"

    return log_text, time_to_failure, high_severity


def generate_dataset(num_entries: int = DEFAULT_NUM_ENTRIES) -> pd.DataFrame:
    """
    Generate a synthetic log dataset.

    Args:
        num_entries: Number of log rows to generate.

    Returns:
        A pandas DataFrame with log text, time to failure, and severity labels.
    """
    dataset_rows: List[List[object]] = []
    current_time = datetime.now()

    for _ in range(num_entries):
        log_text, time_to_failure, high_severity = generate_log_entry(current_time)
        dataset_rows.append([log_text, time_to_failure, high_severity])
        current_time += timedelta(minutes=random.randint(1, 5))

    return pd.DataFrame(
        dataset_rows,
        columns=["log_text", "time_to_failure", "high_severity"],
    )


def introduce_label_noise(
    dataset: pd.DataFrame,
    noise_ratio: float = DEFAULT_NOISE_RATIO,
) -> pd.DataFrame:
    """
    Introduce controlled label noise into the dataset.

    Args:
        dataset: Input dataset.
        noise_ratio: Fraction of rows whose severity labels should be flipped.

    Returns:
        A copy of the dataset with noisy severity labels.
    """
    noisy_dataset = dataset.copy()
    noisy_row_count = int(len(noisy_dataset) * noise_ratio)

    selected_indices = random.sample(range(len(noisy_dataset)), noisy_row_count)

    for row_index in selected_indices:
        noisy_dataset.loc[row_index, "high_severity"] = (
            1 - noisy_dataset.loc[row_index, "high_severity"]
        )

    return noisy_dataset


def _determine_warn_severity(message: str) -> int:
    """
    Determine severity label for warning messages using heuristic probabilities.

    Args:
        message: Warning log message.

    Returns:
        Binary severity label.
    """
    normalized_message = message.lower()

    if "memory" in normalized_message:
        return 1 if random.random() < 0.7 else 0
    if "disk" in normalized_message:
        return 1 if random.random() < 0.6 else 0
    if "failover" in normalized_message:
        return 1 if random.random() < 0.5 else 0

    return 1 if random.random() < 0.2 else 0


def _apply_message_variation(message: str) -> str:
    """
    Apply small random variations to a message for realism.

    Args:
        message: Original message.

    Returns:
        Modified message with optional variation.
    """
    updated_message = message

    if random.random() < 0.15:
        updated_message += " - retry attempt logged"

    if random.random() < 0.10:
        updated_message = updated_message.replace("detected", "observed")

    return updated_message


def _apply_time_noise(time_to_failure: int) -> int:
    """
    Add bounded random noise to the time-to-failure value.

    Args:
        time_to_failure: Original simulated time-to-failure value.

    Returns:
        Adjusted non-negative time-to-failure value.
    """
    adjusted_time = time_to_failure + random.randint(-10, 15)
    return max(adjusted_time, 0)


def main() -> None:
    """
    Generate the dataset, inject light label noise, and save it to CSV.
    """
    dataset = generate_dataset(DEFAULT_GENERATED_ENTRIES)
    dataset = introduce_label_noise(dataset, noise_ratio=DEFAULT_NOISE_RATIO)
    dataset.to_csv(DEFAULT_OUTPUT_FILE, index=False)

    print("Realistic dataset generated successfully!")


if __name__ == "__main__":
    main()