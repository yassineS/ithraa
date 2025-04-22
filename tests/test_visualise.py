import pytest
import tempfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import os

from ithraa.visualise import (
    visualise_gene_matching,
    create_comparison_plots,
    plot_distance_distributions,
    calculate_distances
)

def test_visualise_gene_matching():
    """Test visualization of gene matching."""
    # Create test data
    gene_set = pl.DataFrame({
        'gene_id': ['g1', 'g2', 'g3'],
        'gc': [0.4, 0.5, 0.6],
        'length': [1000, 2000, 3000],
        'distance': [0, 0, 0]
    })
    
    matched_genes = pl.DataFrame({
        'gene_id': ['g4', 'g5', 'g6'],
        'gc': [0.41, 0.51, 0.61],
        'length': [1050, 2050, 3050],
        'distance': [100, 200, 300]
    })
    
    # Test PCA visualization
    result = visualise_gene_matching(
        gene_set,
        matched_genes,
        method='pca',
        confounders=['gc', 'length'],
        distance_col='distance'
    )
    
    assert 'coordinates' in result
    assert 'labels' in result
    assert 'variance_explained' in result
    assert result['coordinates'].shape == (6, 2)  # 3 target + 3 matched genes, 2 components
    assert len(result['labels']) == 6
    assert len(result['variance_explained']) == 2
    
    # Test t-SNE visualization
    result = visualise_gene_matching(
        gene_set,
        matched_genes,
        method='tsne',
        confounders=['gc', 'length'],
        distance_col='distance'
    )
    
    assert 'coordinates' in result
    assert 'labels' in result
    assert result['variance_explained'] is None  # t-SNE doesn't provide variance explained
    
    # Test with empty input
    empty_df = pl.DataFrame({
        'gene_id': [],
        'gc': [],
        'length': [],
        'distance': []
    })
    
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        visualise_gene_matching(empty_df, matched_genes)
    
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        visualise_gene_matching(gene_set, empty_df)
    
    # Test with invalid method
    with pytest.raises(ValueError, match="Unknown visualisation method"):
        visualise_gene_matching(gene_set, matched_genes, method='invalid')

def test_calculate_distances():
    """Test distance calculation between target genes and matches."""
    # Create test data
    gene_set = pl.DataFrame({
        'gene_id': ['g1', 'g2', 'g3']
    })
    
    matched = pl.DataFrame({
        'gene_id': ['g4', 'g5', 'g6'],
        'target_gene_id': ['g1', 'g2', 'g3']
    })
    
    factors = pl.DataFrame({
        'gene_id': ['g1', 'g2', 'g3', 'g4', 'g5', 'g6'],
        'gc': [0.4, 0.5, 0.6, 0.41, 0.51, 0.61],
        'length': [1000, 2000, 3000, 1050, 2050, 3050]
    })
    
    # Test with normal data
    result = calculate_distances(gene_set, matched, factors)
    
    assert 'gene_id' in result.columns
    assert 'target_gene_id' in result.columns
    assert 'mahalanobis_distance' in result.columns
    assert 'euclidean_distance' in result.columns
    assert result.height == 3  # One row per matched gene
    
    # Test with empty input
    empty_df = pl.DataFrame({
        'gene_id': [],
        'target_gene_id': []
    })
    
    # Should return empty df with correct schema
    result = calculate_distances(gene_set, empty_df, factors)
    assert result.height == 0
    assert 'gene_id' in result.columns
    assert 'mahalanobis_distance' in result.columns
    
    # Test error when target_gene_id is missing
    bad_matched = pl.DataFrame({
        'gene_id': ['g4', 'g5', 'g6'],
        # Missing target_gene_id column
    })
    
    with pytest.raises(ValueError, match="matched_df must have a target_gene_id column"):
        calculate_distances(gene_set, bad_matched, factors)

def test_plot_distance_distributions(monkeypatch):
    """Test plotting distance distributions."""
    # Mock plt.savefig to avoid actual file creation
    called_save = False
    def mock_savefig(*args, **kwargs):
        nonlocal called_save
        called_save = True
    
    monkeypatch.setattr(plt, 'savefig', mock_savefig)
    monkeypatch.setattr(plt, 'close', lambda: None)
    
    # Create test data
    traditional = pl.DataFrame({
        'gene_id': ['g4', 'g5', 'g6'],
        'target_gene_id': ['g1', 'g2', 'g3'],
        'mahalanobis_distance': [1.0, 2.0, 3.0],
        'euclidean_distance': [0.1, 0.2, 0.3]
    })
    
    mahalanobis = pl.DataFrame({
        'gene_id': ['g7', 'g8', 'g9'],
        'target_gene_id': ['g1', 'g2', 'g3'],
        'mahalanobis_distance': [0.5, 1.5, 2.5],
        'euclidean_distance': [0.05, 0.15, 0.25]
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_distance_distributions(traditional, mahalanobis, Path(temp_dir))
        assert called_save
    
    # Test with empty inputs
    empty_df = pl.DataFrame({
        'gene_id': [],
        'target_gene_id': [],
        'mahalanobis_distance': [],
        'euclidean_distance': []
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Should not raise an error, just print a message and return
        plot_distance_distributions(empty_df, empty_df, Path(temp_dir))

def test_create_comparison_plots(monkeypatch):
    """Test creation of comparison plots for gene matching methods."""
    # Mock plt.savefig to avoid actual file creation
    saved_files = []
    def mock_savefig(filepath, *args, **kwargs):
        saved_files.append(str(filepath))
    
    monkeypatch.setattr(plt, 'savefig', mock_savefig)
    monkeypatch.setattr(plt, 'close', lambda: None)
    
    # Create test data
    gene_set = pl.DataFrame({
        'gene_id': ['g1', 'g2', 'g3']
    })
    
    traditional = pl.DataFrame({
        'gene_id': ['g4', 'g5', 'g6'],
        'target_gene_id': ['g1', 'g2', 'g3']
    })
    
    mahalanobis = pl.DataFrame({
        'gene_id': ['g7', 'g8', 'g9'],
        'target_gene_id': ['g1', 'g2', 'g3'],
        'mahalanobis_distance': [0.5, 1.5, 2.5]
    })
    
    factors = pl.DataFrame({
        'gene_id': ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9'],
        'gc': [0.4, 0.5, 0.6, 0.41, 0.51, 0.61, 0.42, 0.52, 0.62],
        'length': [1000, 2000, 3000, 1050, 2050, 3050, 1100, 2100, 3100]
    })
    
    distances = pl.DataFrame({
        'gene_id': ['g4', 'g5', 'g6', 'g7', 'g8', 'g9'],
        'target_gene_id': ['g1', 'g2', 'g3', 'g1', 'g2', 'g3'],
        'mahalanobis_distance': [1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
        'euclidean_distance': [0.1, 0.2, 0.3, 0.05, 0.15, 0.25]
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Reset saved files
        saved_files.clear()
        
        create_comparison_plots(
            gene_set,
            traditional,
            mahalanobis,
            factors,
            distances,
            Path(temp_dir),
            methods=['pca', 'tsne']
        )
        
        # Check if two files were saved (one for PCA, one for t-SNE)
        assert len(saved_files) == 2
        assert any('pca' in file.lower() for file in saved_files)
        assert any('tsne' in file.lower() for file in saved_files)
    
    # Test with empty mahalanobis matches (should still work with traditional only)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Reset saved files
        saved_files.clear()
        
        empty_maha = pl.DataFrame({
            'gene_id': [],
            'target_gene_id': [],
            'mahalanobis_distance': []
        })
        
        create_comparison_plots(
            gene_set,
            traditional,
            empty_maha,
            factors,
            distances,
            Path(temp_dir),
            methods=['pca']
        )
        
        # Should still save one file for PCA
        assert len(saved_files) == 1

    # Test with empty inputs for all dataframes
    with tempfile.TemporaryDirectory() as temp_dir:
        # Reset saved files
        saved_files.clear()
        
        empty_df = pl.DataFrame({
            'gene_id': []
        })
        
        empty_matches = pl.DataFrame({
            'gene_id': [],
            'target_gene_id': []
        })
        
        empty_factors = pl.DataFrame({
            'gene_id': [],
            'gc': [],
            'length': []
        })
        
        empty_distances = pl.DataFrame({
            'gene_id': [],
            'target_gene_id': [],
            'mahalanobis_distance': [],
            'euclidean_distance': []
        })
        
        # Should not save any files but also not crash
        create_comparison_plots(
            empty_df,
            empty_matches,
            empty_matches,
            empty_factors,
            empty_distances,
            Path(temp_dir)
        )
        
        assert len(saved_files) == 0