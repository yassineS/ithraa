"""
Test cases for the gene set enrichment pipeline.
"""

import pytest
import os
import logging
import polars as pl
import numba as nb
import numba.np.unsafe.ndarray as np  # Use Numba's NumPy API
from pathlib import Path
import tempfile
import shutil
from tomli_w import dump as tomli_w_dump

from ithraa.pipeline import (
    GeneSetEnrichmentPipeline,
    _perform_permutation,
    _process_threshold,
    _calculate_enrichment_ratio
)
from ithraa.config import PipelineConfig
from ithraa.data import (
    load_gene_list,
    process_gene_set,
    find_control_genes,
    match_confounding_factors,
    shuffle_genome,
    compute_gene_distances
)
from ithraa.stats import (
    calculate_enrichment,
    perform_fdr_analysis,
    bootstrap_analysis
)
from unittest.mock import patch

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    # Create sample gene data
    genes = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3', 'gene4'],
        'score': [0.1, 0.2, 0.3, 0.4]
    })

    # Create sample gene coordinates data with larger distances
    gene_coords = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3', 'gene4'],
        'chrom': ['1', '1', '2', '2'],
        'start': [1000, 10000, 20000, 30000],
        'end': [1500, 10500, 20500, 30500]
    })

    # Create sample factor data with similar values
    factors = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3', 'gene4'],
        'factor': [0.1, 0.2, 0.11, 0.4]  # gene1 and gene3 are similar
    })

    return {
        'genes': genes,
        'gene_coords': gene_coords,
        'factors': factors
    }

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def minimal_config_file(temp_dir):
    """Create a minimal valid config file."""
    # Create input files
    gene_list_file = temp_dir / "gene_list.txt"
    gene_list_file.write_text("""gene_id\tPop1\tPop2
ENSG1\t0.1\t0.2
ENSG2\t0.3\t0.4
ENSG3\t0.5\t0.6""")

    gene_coords_file = temp_dir / "gene_coords.txt"
    gene_coords_file.write_text("""gene_id\tchrom\tstart\tend
ENSG1\t1\t1000\t2000
ENSG2\t1\t3000\t4000
ENSG3\t2\t1000\t2000""")

    factors_file = temp_dir / "factors.txt"
    factors_file.write_text("""gene_id\tfactor
ENSG1\t0.5
ENSG2\t0.7
ENSG3\t0.9""")

    # Create config file
    config = {
        'input': {
            'gene_list_file': str(gene_list_file),
            'gene_coords_file': str(gene_coords_file),
            'factors_file': str(factors_file)
        },
        'output': {
            'directory': str(temp_dir / "results")
        },
        'analysis': {
            'num_threads': 4
        }
    }
    
    config_file = temp_dir / "config.toml"
    with open(config_file, 'wb') as f:
        tomli_w_dump(config, f)
    
    return config_file

def test_pipeline_initialization(minimal_config_file):
    """Test pipeline initialization with valid config."""
    pipeline = GeneSetEnrichmentPipeline(minimal_config_file)
    
    assert isinstance(pipeline.config, PipelineConfig)
    assert isinstance(pipeline.logger, logging.Logger)
    assert hasattr(pipeline, 'gene_list')
    assert hasattr(pipeline, 'gene_coords')
    assert hasattr(pipeline, 'factors')

def test_pipeline_initialization_invalid_config(temp_dir):
    """Test pipeline initialization with invalid config."""
    invalid_config = temp_dir / "invalid.toml"
    invalid_config.write_text("invalid content")
    
    with pytest.raises(ValueError):
        GeneSetEnrichmentPipeline(invalid_config)

def test_pipeline_initialization_missing_files(temp_dir):
    """Test pipeline initialization with missing input files."""
    config = {
        'input': {
            'gene_list_file': str(temp_dir / "nonexistent.txt"),
            'gene_coords_file': str(temp_dir / "nonexistent.txt"),
            'factors_file': str(temp_dir / "nonexistent.txt")
        },
        'output': {
            'directory': str(temp_dir / "results")
        },
        'analysis': {
            'num_threads': 4
        }
    }
    
    config_file = temp_dir / "config.toml"
    with open(config_file, 'wb') as f:
        tomli_w_dump(config, f)
    
    with pytest.raises(FileNotFoundError):
        GeneSetEnrichmentPipeline(config_file)

def test_pipeline_run(minimal_config_file):
    """Test running the pipeline."""
    pipeline = GeneSetEnrichmentPipeline(minimal_config_file)
    pipeline.run()  # Should not raise any errors

@patch("ithraa.pipeline.GeneSetEnrichmentPipeline._run_permutation_fdr")
@patch("ithraa.pipeline.GeneSetEnrichmentPipeline.save_results")
def test_pipeline_initialization_and_run(mock_save, mock_fdr, minimal_config_file):
    """Test pipeline initialization and run with mocked expensive operations."""
    pipeline = GeneSetEnrichmentPipeline(minimal_config_file)
    
    # Check that files are loaded correctly
    assert pipeline.gene_list is not None
    assert pipeline.gene_coords is not None
    assert pipeline.factors is not None
    
    # Test run with shortened steps (mocked expensive operations)
    pipeline.run()
    
    # Check that save_results was called
    mock_save.assert_called_once()
    
    # Check that _run_permutation_fdr was called
    assert mock_fdr.called

def test_pipeline_save_results(minimal_config_file, temp_dir):
    """Test saving pipeline results."""
    pipeline = GeneSetEnrichmentPipeline(minimal_config_file)
    
    # Create some mock results to save
    pipeline.threshold_results = {
        100: {
            'pop1': {
                'enrichment_ratio': 2.5,
                'p_value': 0.001,
                'observed_mean': 0.2,
                'control_mean': 0.08,
                'observed_size': 20,
                'control_size': 30,
                'control_ci_low': 0.06,
                'control_ci_high': 0.10,
                'significant': True
            },
            'pop2': {
                'enrichment_ratio': 1.2,
                'p_value': 0.2,
                'observed_mean': 0.12,
                'control_mean': 0.10,
                'observed_size': 20,
                'control_size': 30,
                'control_ci_low': 0.08,
                'control_ci_high': 0.12,
                'significant': False
            }
        }
    }
    
    # Test with default output directory
    pipeline.save_results()
    assert (temp_dir / "results").exists()
    
    # Test with custom output directory
    custom_dir = temp_dir / "custom_results"
    pipeline.save_results(str(custom_dir))
    assert custom_dir.exists()

def test_pipeline_data_loading(minimal_config_file):
    """Test data loading in pipeline."""
    pipeline = GeneSetEnrichmentPipeline(minimal_config_file)
    
    # Verify gene list
    assert isinstance(pipeline.gene_list, tuple)
    assert len(pipeline.gene_list) == 2  # DataFrame and population names
    assert isinstance(pipeline.gene_list[0], pl.DataFrame)
    assert isinstance(pipeline.gene_list[1], list)
    
    # Verify gene coordinates
    assert isinstance(pipeline.gene_coords, pl.DataFrame)
    assert all(col in pipeline.gene_coords.columns for col in ['gene_id', 'chrom', 'start', 'end'])
    
    # Verify factors
    assert isinstance(pipeline.factors, pl.DataFrame)
    assert all(col in pipeline.factors.columns for col in ['gene_id', 'factor'])

def test_pipeline_logging(tmp_path, minimal_config_file, caplog):
    """Test logging functionality in the pipeline."""
    caplog.set_level(logging.INFO)
    
    # Create a pipeline instance with a non-existent file
    config = {
        'input': {
            'gene_list_file': str(tmp_path / "nonexistent.txt"),
            'gene_coords_file': str(tmp_path / "nonexistent.txt"),
            'factors_file': str(tmp_path / "nonexistent.txt")
        },
        'output': {
            'directory': str(tmp_path / "results")
        },
        'analysis': {
            'num_threads': 4
        }
    }
    
    config_file = tmp_path / "config.toml"
    with open(config_file, 'wb') as f:
        tomli_w_dump(config, f)
    
    # Test error logging
    with pytest.raises(FileNotFoundError):
        pipeline = GeneSetEnrichmentPipeline(config_file)
        pipeline._load_input_data()
    
    # Check for error log messages
    assert any("Input file not found" in record.message for record in caplog.records)

def test_data_processing(sample_data):
    """Test data processing functions with custom column names."""
    # Test gene set processing
    processed_genes = process_gene_set(
        sample_data['genes'],
        sample_data['gene_coords'],
        sample_data['factors']
    )
    assert len(processed_genes) == len(sample_data['genes'])

    # Compute distances before finding control genes
    distances = compute_gene_distances(sample_data['gene_coords'])
    print("\nComputed distances:")
    print(distances)

    # Test control gene finding with a smaller minimum distance
    # Use only first two genes as targets
    target_genes = pl.DataFrame({
        'gene_id': ['gene1', 'gene2']
    })
    control_genes = find_control_genes(
        target_genes,
        distances,
        min_distance=10  # Further reduced for testing
    )
    print("\nControl genes:")
    print(control_genes)
    assert len(control_genes) > 0
    
    # Test confounding factor matching
    matched_controls = match_confounding_factors(
        target_genes,
        control_genes,
        sample_data['factors']
    )
    print("\nMatched controls:")
    print(matched_controls)
    assert len(matched_controls) > 0

def test_statistical_analysis(sample_data):
    """Test statistical analysis functions."""
    import numpy as np  # Use standard NumPy instead of Numba's NumPy
    
    # Test enrichment calculation
    target_counts = np.array([5, 10, 15])
    control_counts = np.array([3, 6, 9])
    enrichment = calculate_enrichment(target_counts, control_counts)
    
    assert 'enrichment_ratio' in enrichment
    assert 'p_value' in enrichment
    assert enrichment['enrichment_ratio'] > 0
    
    # Test FDR analysis
    p_values = np.array([0.05, 0.01, 0.2])
    fdr_results = perform_fdr_analysis(p_values)
    
    assert 'reject' in fdr_results
    assert 'pvals_corrected' in fdr_results
    assert len(fdr_results['reject']) == len(p_values)
    
    # Test bootstrap analysis
    data = np.array([0.1, 0.2, 0.3])
    bootstrap_results = bootstrap_analysis(data)
    
    assert 'mean' in bootstrap_results
    assert 'ci_lower' in bootstrap_results
    assert 'ci_upper' in bootstrap_results
    assert bootstrap_results['ci_lower'] < bootstrap_results['ci_upper']

def test_genome_shuffling(sample_data):
    """Test genome shuffling functionality."""
    # Add position column for shuffling
    genome_data = sample_data['gene_coords'].clone()
    
    # Create chromosome sizes dictionary
    chrom_sizes = {'1': 1000000, '2': 1000000}
    
    # Shuffle genome
    shuffled_data = shuffle_genome(
        genome_data,
        chrom_sizes
    )
    
    assert len(shuffled_data) == len(genome_data)
    assert all(col in shuffled_data.columns for col in ['gene_id', 'chrom', 'start', 'end'])
    assert not shuffled_data['start'].equals(genome_data['start'])  # Positions should be shuffled

def test_match_confounding_factors(sample_data):
    """Test matching of confounding factors."""
    # Create sample data with confounding factors
    # Use very similar values for gc_content and length
    genes = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3', 'gene4'],
        'gc_content': [0.4, 0.42, 0.405, 0.55],  # gene1 and gene3 are extremely similar
        'length': [1000, 1200, 1005, 900]  # gene1 and gene3 are extremely similar
    })

    target_genes = pl.DataFrame({
        'gene_id': ['gene1']  # Only use one target gene
    })

    print("\nTarget genes:")
    print(target_genes)
    print("\nAll genes with factors:")
    print(genes)

    # Test matching with multiple factors
    matched_controls = match_confounding_factors(
        target_genes,
        genes,
        genes.select(['gene_id', 'gc_content', 'length']),
        tolerance=0.2  # Increased tolerance to allow for more matches
    )

    print("\nMatched controls:")
    print(matched_controls)
    assert isinstance(matched_controls, pl.DataFrame)
    assert len(matched_controls) > 0
    assert 'gene3' in matched_controls['gene_id'].to_list()  # gene3 should match gene1