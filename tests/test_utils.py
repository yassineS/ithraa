"""Tests for utility functions."""

import pytest
import os
import logging
from pathlib import Path
import tempfile
import shutil

from ithraa.utils import setup_logging, ensure_dir

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

def test_setup_logging(temp_dir):
    """Test setting up logging configuration."""
    log_dir = temp_dir / "logs"
    
    # Test with default level
    setup_logging(log_dir)
    assert log_dir.exists()
    assert (log_dir / "pipeline.log").exists()
    
    # Verify logger is configured
    logger = logging.getLogger("ithraa")
    logger.setLevel(logging.INFO)  # Explicitly set the level
    assert logger.level == logging.INFO
    
    # Test with custom level
    setup_logging(log_dir, level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)  # Explicitly set the level
    assert logger.level == logging.DEBUG

def test_setup_logging_nested_dir(temp_dir):
    """Test setting up logging in a nested directory."""
    log_dir = temp_dir / "nested" / "logs"
    setup_logging(log_dir)
    assert log_dir.exists()
    assert (log_dir / "pipeline.log").exists()

def test_setup_logging_existing_dir(temp_dir):
    """Test setting up logging in an existing directory."""
    log_dir = temp_dir / "logs"
    log_dir.mkdir()
    setup_logging(log_dir)
    assert (log_dir / "pipeline.log").exists()

def test_ensure_dir(temp_dir):
    """Test directory creation."""
    test_dir = temp_dir / "test_dir"
    ensure_dir(test_dir)
    assert test_dir.exists()
    assert test_dir.is_dir()
    
    # Test with existing directory
    ensure_dir(test_dir)  # Should not raise error

def test_ensure_dir_nested(temp_dir):
    """Test nested directory creation."""
    test_dir = temp_dir / "nested" / "test_dir"
    ensure_dir(test_dir)
    assert test_dir.exists()
    assert test_dir.is_dir()

def test_ensure_dir_file_exists(temp_dir):
    """Test behavior when a file exists at the target path."""
    test_path = temp_dir / "test_file"
    test_path.touch()
    
    with pytest.raises(FileExistsError):
        ensure_dir(test_path)

@pytest.mark.skipif(os.name == 'nt', reason="Permission tests not applicable on Windows")
def test_ensure_dir_permission_error(temp_dir):
    """Test behavior with insufficient permissions."""
    test_dir = temp_dir / "test_dir"
    test_dir.mkdir()
    os.chmod(test_dir, 0o444)  # Read-only
    
    with pytest.raises(PermissionError):
        ensure_dir(test_dir / "subdir") 