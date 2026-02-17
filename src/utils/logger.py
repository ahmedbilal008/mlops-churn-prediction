"""
Structured Logging
==================

Provides consistent logging across all modules.

WHY: In production ML systems, structured logging is critical for:
- Debugging training failures
- Monitoring data drift
- Auditing model decisions (regulatory requirement)
- Performance profiling

INDUSTRY PRACTICE: Teams use structured logging (JSON format) with
tools like ELK Stack, Datadog, or CloudWatch for centralized monitoring.
We use Python's built-in logging for simplicity but structure it for
easy migration to production logging frameworks.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a configured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(level)

        # Console handler with structured format
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent duplicate logs from propagating
        logger.propagate = False

    return logger
