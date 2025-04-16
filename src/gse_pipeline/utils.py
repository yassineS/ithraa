"""Utility functions for gene set enrichment analysis."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Set up logging configuration.

    Args:
        log_dir: Directory to store log files
        level: Logging level (default: INFO)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'pipeline.log'

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path 