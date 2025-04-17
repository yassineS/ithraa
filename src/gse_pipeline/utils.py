"""Utility functions for gene set enrichment analysis."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

def setup_logging(log_dir=None, level=logging.INFO):
    """Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level
    """
    # Create log directory if specified and doesn't exist
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'pipeline.log'
        
        # Create a file handler to write logs to file
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Ensure the file is created immediately by writing an initial log message
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        root_logger.addHandler(file_handler)
        root_logger.info("Logging initialized")
    
    # Always add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    return logging.getLogger('gse_pipeline')

def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path.mkdir(parents=True, exist_ok=True)
    return path