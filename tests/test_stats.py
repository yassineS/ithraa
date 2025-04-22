"""Tests for statistical analysis functions."""

import pytest
import numba as nb
import numpy as np  # Use standard NumPy instead of Numba's NumPy
import pandas as pd
import polars as pl
from scipy import stats
import os
import tempfile
from pathlib import Path

from ithraa.stats import (
    calculate_enrichment,
    perform_fdr_analysis,
    bootstrap_analysis,
    control_for_confounders,
    compute_enrichment_score,
    calculate_significance,
    match_genes_mahalanobis
)

def test_calculate_enrichment():
    """Test enrichment calculation."""
    # Test with normal data - using more distinct values to avoid warnings
    target_counts = np.array([10, 20, 30, 40, 50])
    control_counts = np.array([1, 2, 3, 4, 5])
    
    result = calculate_enrichment(target_counts, control_counts)
    
    assert isinstance(result, dict)
    assert 'enrichment_ratio' in result
    assert 'p_value' in result
    assert 'observed_mean' in result
    assert 'control_mean' in result
    assert result['enrichment_ratio'] == 10.0
    assert result['observed_mean'] == 30.0
    assert result['control_mean'] == 3.0
    
    # Test with zero control mean
    target_counts = np.array([1, 2, 3])
    control_counts = np.array([0, 0, 0])
    
    result = calculate_enrichment(target_counts, control_counts)
    assert 'nan' in str(result['enrichment_ratio']).lower()  # Check for NaN
    
    # Test with empty arrays
    result = calculate_enrichment(np.array([]), np.array([]))
    assert result['enrichment_ratio'] != result['enrichment_ratio']  # NaN check
    
    # Test with identical arrays (should not trigger warning)
    identical_data = np.array([1, 2, 3])
    result = calculate_enrichment(identical_data, identical_data)
    assert result['p_value'] == 1.0
    assert result['enrichment_ratio'] == 1.0

def test_perform_fdr_analysis():
    """Test FDR analysis."""
    # Test with normal p-values
    p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    result = perform_fdr_analysis(p_values)
    
    assert isinstance(result, dict)
    assert 'reject' in result
    assert 'pvals_corrected' in result
    assert isinstance(result['reject'], list)
    assert isinstance(result['pvals_corrected'], list)
    assert len(result['reject']) == len(p_values)
    assert len(result['pvals_corrected']) == len(p_values)
    
    # Test with custom alpha
    result = perform_fdr_analysis(p_values, alpha=0.01)
    assert sum(result['reject']) <= sum(1 for p in p_values if p <= 0.01)
    
    # Test with empty array
    with pytest.raises(ValueError, match="Input p-values array cannot be empty"):
        perform_fdr_analysis(np.array([]))

def test_bootstrap_analysis():
    """Test bootstrap analysis."""
    # Test with normal data
    data = np.array([1, 2, 3, 4, 5])
    result = bootstrap_analysis(data, n_iterations=100)
    
    assert isinstance(result, dict)
    assert 'mean' in result
    assert 'median' in result
    assert 'ci_lower' in result
    assert 'ci_upper' in result
    assert result['ci_lower'] <= result['mean'] <= result['ci_upper']
    
    # Test with median statistic
    result = bootstrap_analysis(data, statistic='median')
    assert round(result['median']) == 3.0
    
    # Test with empty array - with strict validation
    with pytest.raises(ValueError, match="Input data array cannot be empty"):
        bootstrap_analysis(np.array([]), strict_validation=True)
    
    # Test with empty array - normal operation
    result = bootstrap_analysis(np.array([]))
    assert result['mean'] == 0.0
    assert result['median'] == 0.0
    assert result['ci_lower'] == 0.0
    assert result['ci_upper'] == 0.0
    
    # Test with invalid statistic
    with pytest.raises(ValueError):
        bootstrap_analysis(data, statistic='invalid')

def test_control_for_confounders():
    """Test confounder control."""
    # Create test data
    target_data = pl.DataFrame({
        'score': [1, 2, 3],
        'confounder1': [0.1, 0.2, 0.3],
        'confounder2': [0.4, 0.5, 0.6]
    })
    
    control_data = pl.DataFrame({
        'score': [0, 1, 2],
        'confounder1': [0.1, 0.2, 0.3],
        'confounder2': [0.4, 0.5, 0.6]
    })
    
    confounders = ['confounder1', 'confounder2']
    result = control_for_confounders(target_data, control_data, confounders)
    
    assert isinstance(result, dict)
    assert 'coefficients' in result
    assert 'p_values' in result
    assert 'aic' in result
    assert 'bic' in result
    
    # Test with empty data
    empty_data = pl.DataFrame({
        'score': [],
        'confounder1': [],
        'confounder2': []
    })
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        control_for_confounders(empty_data, empty_data, confounders)
    
    # Test with missing confounders
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        control_for_confounders(target_data, control_data, ['missing'])

def test_compute_enrichment_score():
    """Test enrichment score computation."""
    # Create test data with more distinct values
    target_data = pl.DataFrame({
        'score': [10, 20, 30, 40, 50]
    })
    
    control_data = pl.DataFrame({
        'score': [1, 2, 3, 4, 5]
    })
    
    score, p_value = compute_enrichment_score(target_data, control_data)
    
    assert isinstance(score, float)
    assert isinstance(p_value, float)
    assert score == 27.0  # target_mean - control_mean
    assert 0 <= p_value <= 1
    
    # Test with custom score column
    target_data = target_data.rename({'score': 'custom_score'})
    control_data = control_data.rename({'score': 'custom_score'})
    score, p_value = compute_enrichment_score(
        target_data,
        control_data,
        score_col='custom_score'
    )
    assert score == 27.0
    
    # Test with identical data (should not trigger warning)
    identical_data = pl.DataFrame({'score': [1, 2, 3]})
    score, p_value = compute_enrichment_score(identical_data, identical_data)
    assert score == 0.0
    assert p_value == 1.0
    
    # Test with empty data
    empty_data = pl.DataFrame({'score': []})
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        compute_enrichment_score(empty_data, empty_data)

def test_calculate_significance():
    """Test significance calculation."""
    # Test with normal data
    observed_score = 2.0
    null_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    p_value, is_significant = calculate_significance(observed_score, null_scores)
    
    assert isinstance(p_value, float)
    assert isinstance(is_significant, bool)
    assert 0 <= p_value <= 1
    assert isinstance(is_significant, bool)  # Make sure it's a Python bool, not numpy.bool_
    
    # Test with custom alpha
    p_value, is_significant = calculate_significance(observed_score, null_scores, alpha=0.01)
    assert is_significant
    
    # Test with empty null scores
    with pytest.raises(ValueError, match="Null scores array cannot be empty"):
        calculate_significance(observed_score, [])
    
    # Test with non-significant result
    observed_score = 0.3
    p_value, is_significant = calculate_significance(observed_score, null_scores)
    assert not is_significant

def test_match_genes_mahalanobis():
    """Test Mahalanobis gene matching function."""
    import polars as pl
    import numpy as np
    import tempfile
    from pathlib import Path
    from ithraa.stats import match_genes_mahalanobis
    
    # Create test data
    gene_set_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3'],
        'chrom': ['chr1', 'chr2', 'chr3'],
        'gc': [0.4, 0.5, 0.6],
        'length': [1000, 2000, 3000]
    })
    
    all_genes_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'gene6'],
        'chrom': ['chr1', 'chr2', 'chr3', 'chr1', 'chr2', 'chr3'],
        'gc': [0.4, 0.5, 0.6, 0.41, 0.51, 0.61],
        'length': [1000, 2000, 3000, 1050, 2050, 3050]
    })

    # Test with normal parameters
    with tempfile.TemporaryDirectory() as temp_dir:
        matches = match_genes_mahalanobis(
            gene_set_data=gene_set_data,
            all_genes_data=all_genes_data,
            confounders=['gc', 'length'],
            exclude_gene_ids=['gene1', 'gene2', 'gene3'],
            n_matches=1,
            sd_threshold=1.0,
            save_intermediate=True,
            intermediate_dir=temp_dir
        )
        
        # Check if matches were found
        assert matches.height == 3  # One match for each target gene
        assert 'gene_id' in matches.columns
        assert 'target_gene_id' in matches.columns
        assert 'mahalanobis_distance' in matches.columns
        
        # Verify that excluded genes are not in matches
        assert not any(gene_id in ['gene1', 'gene2', 'gene3'] for gene_id in matches['gene_id'])
    
    # Test with empty gene set
    empty_gene_set = pl.DataFrame({
        'gene_id': [],
        'chrom': [],
        'gc': [],
        'length': []
    })
    
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        match_genes_mahalanobis(
            gene_set_data=empty_gene_set,
            all_genes_data=all_genes_data,
            confounders=['gc', 'length']
        )
    
    # Test with empty background genes
    empty_background = pl.DataFrame({
        'gene_id': [],
        'chrom': [],
        'gc': [],
        'length': []
    })
    
    with pytest.raises(ValueError, match="Input data cannot be empty"):
        match_genes_mahalanobis(
            gene_set_data=gene_set_data,
            all_genes_data=empty_background,
            confounders=['gc', 'length']
        )
        
    # Test with no confounders
    no_confounder_matches = match_genes_mahalanobis(
        gene_set_data=gene_set_data,
        all_genes_data=all_genes_data,
        confounders=[]
    )
    
    # Should return empty DataFrame with expected schema
    assert no_confounder_matches.height == 0
    assert 'gene_id' in no_confounder_matches.columns
    assert 'target_gene_id' in no_confounder_matches.columns
    assert 'mahalanobis_distance' in no_confounder_matches.columns
    
    # Test with missing confounder columns
    missing_conf_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3'],
        'chrom': ['chr1', 'chr2', 'chr3'],
        # Missing gc column
        'length': [1000, 2000, 3000]
    })
    
    # Should still work but print warning
    matches = match_genes_mahalanobis(
        gene_set_data=missing_conf_data,
        all_genes_data=all_genes_data,
        confounders=['gc', 'length']
    )
    
    # Test with all background genes excluded
    matches = match_genes_mahalanobis(
        gene_set_data=gene_set_data,
        all_genes_data=all_genes_data,
        confounders=['gc', 'length'],
        exclude_gene_ids=['gene1', 'gene2', 'gene3', 'gene4', 'gene5', 'gene6']
    )
    
    # Should return empty DataFrame with expected schema
    assert matches.height == 0
    
    # Test with different standard deviation threshold
    with tempfile.TemporaryDirectory() as temp_dir:
        matches_high_sd = match_genes_mahalanobis(
            gene_set_data=gene_set_data,
            all_genes_data=all_genes_data,
            confounders=['gc', 'length'],
            exclude_gene_ids=['gene1', 'gene2', 'gene3'],
            n_matches=1,
            sd_threshold=2.0,  # Higher threshold
            save_intermediate=True,
            intermediate_dir=temp_dir
        )
        
        # With higher threshold, we should get matches for all genes
        assert matches_high_sd.height == 3

def test_match_genes_mahalanobis_edge_cases():
    """Test edge cases for Mahalanobis gene matching function."""
    import polars as pl
    import numpy as np
    import tempfile
    from pathlib import Path
    from ithraa.stats import match_genes_mahalanobis
    
    # Test with non-numeric columns
    gene_set_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2'],
        'chrom': ['chr1', 'chr2'],
        'category': ['A', 'B'],  # Non-numeric column
        'length': [1000, 2000]
    })
    
    all_genes_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3', 'gene4'],
        'chrom': ['chr1', 'chr2', 'chr1', 'chr2'],
        'category': ['A', 'B', 'A', 'B'],  # Non-numeric column
        'length': [1000, 2000, 1050, 2050]
    })
    
    # Should work but ignore non-numeric columns
    with tempfile.TemporaryDirectory() as temp_dir:
        matches = match_genes_mahalanobis(
            gene_set_data=gene_set_data,
            all_genes_data=all_genes_data,
            confounders=['length'],  # Only use numeric column
            exclude_gene_ids=['gene1', 'gene2'],
            n_matches=1,
            save_intermediate=True,
            intermediate_dir=temp_dir
        )
        
        # Check if matches were found
        assert matches.height == 2  # One match for each target gene
    
    # Test with singular covariance matrix (columns are perfectly correlated)
    gene_set_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2'],
        'chrom': ['chr1', 'chr2'],
        'factor1': [1.0, 2.0],
        'factor2': [2.0, 4.0]  # Exactly 2x factor1, will cause singularity
    })
    
    all_genes_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3', 'gene4'],
        'chrom': ['chr1', 'chr2', 'chr1', 'chr2'],
        'factor1': [1.0, 2.0, 3.0, 4.0],
        'factor2': [2.0, 4.0, 6.0, 8.0]  # Exactly 2x factor1, will cause singularity
    })
    
    # Should use pseudoinverse instead of regular inverse
    matches = match_genes_mahalanobis(
        gene_set_data=gene_set_data,
        all_genes_data=all_genes_data,
        confounders=['factor1', 'factor2'],
        exclude_gene_ids=['gene1', 'gene2'],
        n_matches=1
    )
    
    # Should still find matches using pseudoinverse
    assert matches.height > 0
    
    # Test with zero variance in a feature
    gene_set_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2'],
        'chrom': ['chr1', 'chr2'],
        'const_factor': [5.0, 5.0],  # Zero variance
        'normal_factor': [1.0, 2.0]
    })
    
    all_genes_data = pl.DataFrame({
        'gene_id': ['gene1', 'gene2', 'gene3', 'gene4'],
        'chrom': ['chr1', 'chr2', 'chr1', 'chr2'],
        'const_factor': [5.0, 5.0, 5.0, 5.0],  # Zero variance
        'normal_factor': [1.0, 2.0, 3.0, 4.0]
    })
    
    # Should handle zero variance by adding regularization
    matches = match_genes_mahalanobis(
        gene_set_data=gene_set_data,
        all_genes_data=all_genes_data,
        confounders=['const_factor', 'normal_factor'],
        exclude_gene_ids=['gene1', 'gene2'],
        n_matches=1
    )
    
    # Should still find matches
    assert matches.height > 0