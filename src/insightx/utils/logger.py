"""Standardized logging utility for InsightX AI."""

import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = "insightx") -> logging.Logger:
    """Returns a configured logger instance with formatted console output.

    Args:
        name: Name of the logger (defaults to 'insightx').

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
