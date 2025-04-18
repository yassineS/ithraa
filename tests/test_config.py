"""Tests for configuration management."""

import pytest
import tomli
from tomli_w import dump as tomli_w_dump
from pathlib import Path
import tempfile
import os
from ithraa.config import PipelineConfig

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def minimal_config_file(tmp_path):
    """Create a minimal valid configuration file."""
    config = {
        'input': {
            'gene_list_file': 'genes.txt',
            'gene_coords_file': 'coords.bed',
            'factors_file': 'factors.tsv'
        },
        'output': {
            'output_dir': 'results',
            'directory': 'results'
        },
        'analysis': {
            'gene_filtering': {
                'exclude_prefixes': ['HLA', 'HIST', 'OR'],
                'require_protein_coding': False
            }
        }
    }
    config_path = tmp_path / 'config.toml'
    with open(config_path, 'wb') as f:
        tomli_w_dump(config, f)
    return config_path

@pytest.fixture
def full_config_file(tmp_path):
    """Create a configuration file with all optional parameters."""
    config = {
        'input': {
            'gene_list_file': 'genes.txt',
            'gene_coords_file': 'coords.bed',
            'factors_file': 'factors.tsv',
            'valid_genes_file': 'valid_genes.txt',
            'hgnc_file': 'hgnc.json'
        },
        'output': {
            'output_dir': 'results',
            'directory': 'results'
        },
        'analysis': {
            'gene_filtering': {
                'exclude_prefixes': ['HLA', 'HIST', 'OR'],
                'require_protein_coding': True
            }
        }
    }
    config_path = tmp_path / 'config.toml'
    with open(config_path, 'wb') as f:
        tomli_w_dump(config, f)
    return config_path

def test_load_minimal_config(minimal_config_file):
    """Test loading a minimal valid configuration."""
    config = PipelineConfig(minimal_config_file)
    assert 'valid_genes_file' not in config.config['input']
    assert 'hgnc_file' not in config.config['input']
    assert config.config['analysis']['gene_filtering']['require_protein_coding'] is False

def test_load_full_config(full_config_file):
    """Test loading a configuration with all optional parameters."""
    config = PipelineConfig(full_config_file)
    assert config.config['input']['valid_genes_file'] == 'valid_genes.txt'
    assert config.config['input']['hgnc_file'] == 'hgnc.json'
    assert config.config['analysis']['gene_filtering']['require_protein_coding'] is True

def test_save_config(minimal_config_file, tmp_path):
    """Test saving configuration to a new file."""
    config = PipelineConfig(minimal_config_file)
    output_path = tmp_path / 'saved_config.toml'
    config.save_config(output_path)
    
    # Load saved config and verify contents
    with open(output_path, 'rb') as f:
        saved_config = tomli.load(f)
    assert saved_config == config.config

def test_nonexistent_config_file():
    """Test error handling for nonexistent configuration file."""
    with pytest.raises(ValueError, match="Error loading configuration file"):
        PipelineConfig('nonexistent.toml')

def test_missing_required_section(tmp_path):
    """Test error handling for missing required section."""
    config = {
        'input': {
            'gene_list_file': 'genes.txt',
            'gene_coords_file': 'coords.bed',
            'factors_file': 'factors.tsv'
        }
        # Missing 'output' and 'analysis' sections
    }
    config_path = tmp_path / 'invalid_config.toml'
    with open(config_path, 'wb') as f:
        tomli_w_dump(config, f)
    
    with pytest.raises(ValueError, match="Missing required sections"):
        PipelineConfig(config_path)

def test_missing_required_files(tmp_path):
    """Test error handling for missing required input files."""
    config = {
        'input': {
            'gene_list_file': 'genes.txt'
            # Missing 'gene_coords_file' and 'factors_file'
        },
        'output': {
            'output_dir': 'results'
        },
        'analysis': {}
    }
    config_path = tmp_path / 'invalid_config.toml'
    with open(config_path, 'wb') as f:
        tomli_w_dump(config, f)
    
    with pytest.raises(ValueError, match="Missing required input files"):
        PipelineConfig(config_path)

def test_get_output_path(minimal_config_file):
    """Test getting output paths."""
    config = PipelineConfig(minimal_config_file)
    
    base_path = config.get_output_path()
    assert base_path.name == 'results'
    
    sub_path = config.get_output_path('subdir')
    assert sub_path.name == 'subdir'
    assert sub_path.parent.name == 'results'

def test_invalid_config_file(temp_dir):
    """Test error with invalid config file."""
    file_path = temp_dir / "config.toml"
    file_path.write_text("invalid toml content")
    
    with pytest.raises(ValueError, match="Error loading configuration file"):
        PipelineConfig(file_path) 