"""Utility functions for gene set enrichment analysis."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Set up logging configuration.

    Args:
        log_dir: Directory to store log files
        level: Logging level (default: INFO)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp to logfile name
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f'pipeline_{timestamp}.log'

    # Clear any existing handlers
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)

    # Configure the root logger
    root_logger.setLevel(logging.DEBUG)  # Capture all possible logs

    # Create file handler with detailed formatting (maintains full verbosity)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Store everything in the log file
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Create console handler with minimal formatting
    # Default to warnings and errors only unless --verbose is passed
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)  # This will be controlled by --verbose flag
    console_format = logging.Formatter('%(message)s')  # Simplified format for console
    console_handler.setFormatter(console_format)
    
    # Add handlers to logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path