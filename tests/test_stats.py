"""Tests for statistical analysis functions."""

import pytest
import numba as nb
import numpy as np  # Use standard NumPy instead of Numba's NumPy
import pandas as pd
import polars as pl
from scipy import stats

from gse_pipeline.stats import (
    calculate_enrichment,
    perform_fdr_analysis,
    bootstrap_analysis,
    control_for_confounders,
    compute_enrichment_score,
    calculate_significance
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