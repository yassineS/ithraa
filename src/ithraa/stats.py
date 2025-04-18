"""
Statistical analysis functions for the gene set enrichment pipeline.
"""

from typing import List, Dict, Union, Optional, Tuple
import numba as nb
import numpy as np  # Use standard numpy for array creation
from numba.np.unsafe import ndarray as numba_ndarray  # Use only for type annotations
import polars as pl

from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

# Core numba-optimised functions for inner loops and heavy calculations

# Simplified versions for better compatibility
@nb.njit
def _abs_vec(x):
    """Optimised absolute value using Numba"""
    return -x if x < 0 else x

@nb.njit
def _sqrt_vec(x):
    """Optimised square root using Numba"""
    return x ** 0.5

@nb.njit(parallel=True)
def _bootstrap_samples(data, n_iterations: int, use_mean: bool = True):
    """
    Generate bootstrap samples using numba with parallel execution.
    
    Args:
        data: Input data array
        n_iterations: Number of bootstrap iterations
        use_mean: Use mean if True, median if False
        
    Returns:
        Array of bootstrap statistics
    """
    n = len(data)
    results = np.zeros(n_iterations)
    
    # Use parallel=True with prange for parallel execution
    for i in nb.prange(n_iterations):
        # Generate bootstrap sample indices
        indices = np.random.randint(0, n, size=n)
        
        # Calculate statistic
        sample = data[indices]
        if use_mean:
            results[i] = np.mean(sample)
        else:
            # For median, we need to sort and find the middle value(s)
            sample = np.sort(sample)
            if n % 2 == 0:
                results[i] = (sample[n//2 - 1] + sample[n//2]) / 2
            else:
                results[i] = sample[n//2]
                
    return results

@nb.njit(parallel=True)
def _calculate_pairwise_distances(gene_ids, starts, ends):
    """
    Calculate pairwise distances between genes with parallel processing.
    
    Args:
        gene_ids: Array of gene IDs
        starts: Array of gene start positions
        ends: Array of gene end positions
        
    Returns:
        Tuple of (gene_id1, gene_id2, distances) arrays
    """
    n_genes = len(gene_ids)
    n_pairs = (n_genes * (n_genes - 1)) // 2  # Number of unique pairs
    
    # Pre-allocate numpy arrays for better memory management and vectorization
    gene_id1 = np.zeros(n_pairs, dtype=np.int64)
    gene_id2 = np.zeros(n_pairs, dtype=np.int64)
    distances = np.zeros(n_pairs, dtype=np.float64)
    
    # Use atomic counter for parallel execution
    pair_idx = 0
    
    # Calculate all pairs - dividing work across available threads
    for i in range(n_genes):
        # This inner loop can run in parallel since each i creates independent work
        for j in nb.prange(i+1, n_genes):
            idx = pair_idx + (j - (i+1))
            gene_id1[idx] = gene_ids[i]
            gene_id2[idx] = gene_ids[j]
            
            # Check if genes overlap
            if (starts[i] <= ends[j] and ends[i] >= starts[j]) or \
               (starts[j] <= ends[i] and ends[j] >= starts[i]):
                distances[idx] = 0
            else:
                # Distance is the minimum gap between gene boundaries
                distances[idx] = min(
                    abs(starts[i] - ends[j]),
                    abs(ends[i] - starts[j])
                )
        
        pair_idx += n_genes - (i+1)
            
    return gene_id1, gene_id2, distances

@nb.njit
def _mean(arr):
    """Calculate mean of array using Numba"""
    if len(arr) == 0:
        return 0.0
    return np.sum(arr) / len(arr)

@nb.njit
def _std(arr, ddof=0):
    """Calculate standard deviation using Numba"""
    if len(arr) <= ddof:
        return 0.0
    mean = np.sum(arr) / len(arr)  # Direct calculation is faster than _mean in this context
    sq_diff = 0.0
    for i in range(len(arr)):
        diff = arr[i] - mean
        sq_diff += diff * diff
    return np.sqrt(sq_diff / (len(arr) - ddof))

@nb.njit
def _abs(x):
    """Absolute value using Numba - legacy version for compatibility"""
    return -x if x < 0 else x

@nb.njit
def _sqrt(x):
    """Square root using Numba - legacy version for compatibility"""
    return x ** 0.5

@nb.njit
def _calculate_significance_counts_helper(observed_score, null_score):
    """Helper function for significance count calculation"""
    return 1 if null_score >= observed_score else 0

@nb.njit(parallel=True)
def _calculate_significance_counts(observed_score: float, null_scores) -> int:
    """
    Count how many null scores are greater than or equal to the observed score.
    Uses parallel processing for large arrays.
    
    Args:
        observed_score: Observed score
        null_scores: Array of null distribution scores
        
    Returns:
        Count of scores >= observed
    """
    # For small arrays, direct counting is faster than parallelization overhead
    if len(null_scores) < 10000:
        count = 0
        for i in range(len(null_scores)):
            if null_scores[i] >= observed_score:
                count += 1
        return count
    
    # For large arrays, use parallel processing
    count = 0
    for i in nb.prange(len(null_scores)):
        if null_scores[i] >= observed_score:
            count += 1
    return count

def calculate_enrichment(target_counts, control_counts) -> Dict[str, float]:
    """
    Calculate enrichment statistics with confidence intervals.

    Args:
        target_counts: Array of target counts
        control_counts: Array of control counts

    Returns:
        Dictionary with enrichment statistics including CI and raw values
    """
    # Handle empty arrays gracefully
    if len(target_counts) == 0 or len(control_counts) == 0:
        return {
            'enrichment_ratio': float('nan'),
            'p_value': 1.0,
            'observed_mean': 0.0 if len(target_counts) == 0 else float(_mean(target_counts)),
            'control_mean': 0.0 if len(control_counts) == 0 else float(_mean(control_counts)),
            'observed_size': len(target_counts),
            'control_size': len(control_counts),
            'control_ci_low': 0.0,
            'control_ci_high': 0.0,
            'target_std': 0.0 if len(target_counts) == 0 else float(_std(target_counts, ddof=1)),
            'control_std': 0.0 if len(control_counts) == 0 else float(_std(control_counts, ddof=1))
        }

    # Calculate means and sample size for target and control groups
    target_mean = _mean(target_counts)
    control_mean = _mean(control_counts)
    target_size = len(target_counts)
    control_size = len(control_counts)

    # Calculate enrichment ratio
    if control_mean == 0:
        enrichment_ratio = float('nan')
    else:
        enrichment_ratio = target_mean / control_mean
    
    # Calculate standard deviations for confidence intervals
    target_std = _std(target_counts, ddof=1)  # Use ddof=1 for sample standard deviation
    control_std = _std(control_counts, ddof=1)
    
    # Calculate 95% confidence intervals for the control mean
    # Use t-distribution for small samples
    alpha = 0.05  # 95% confidence interval
    control_t_val = stats.t.ppf(1 - alpha/2, control_size - 1)
    control_margin = control_t_val * (control_std / _sqrt(control_size))
    control_ci_low = control_mean - control_margin
    control_ci_high = control_mean + control_margin
    
    # If the values are very close, or the standard deviation is near zero,
    # use a more conservative estimate for the CI
    if control_std < 1e-10:
        control_ci_low = control_mean * 0.9
        control_ci_high = control_mean * 1.1
    
    # Check if means are nearly identical to avoid precision warnings
    if _abs(target_mean - control_mean) < 1e-10:
        p_value = 1.0  # If the means are practically identical, there's no significant difference
    else:
        # Use Welch's t-test which doesn't assume equal variances or equal sample sizes
        _, p_value = stats.ttest_ind(target_counts, control_counts, equal_var=False)
    
    # Store all values needed for the original pipeline format
    return {
        'enrichment_ratio': float(enrichment_ratio),
        'p_value': float(p_value),
        'observed_mean': float(target_mean),
        'control_mean': float(control_mean),
        'observed_size': int(target_size),
        'control_size': int(control_size),
        'control_ci_low': float(control_ci_low),
        'control_ci_high': float(control_ci_high),
        'target_std': float(target_std),
        'control_std': float(control_std)
    }

def perform_fdr_analysis(p_values, alpha: float = 0.05) -> Dict[str, list]:
    """
    Perform FDR analysis on p-values.

    Args:
        p_values: Array of p-values
        alpha: Significance level

    Returns:
        Dictionary with FDR results
    """
    if len(p_values) == 0:
        raise ValueError("Input p-values array cannot be empty")

    reject, pvals_corrected, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method='fdr_bh'
    )
    
    return {
        'reject': reject.astype(bool).tolist(),  # Convert to Python bool list
        'pvals_corrected': pvals_corrected.tolist()
    }

def bootstrap_analysis(
    data,
    n_iterations: int = 1000,
    statistic: str = 'mean',
    strict_validation: bool = False
) -> Dict[str, float]:
    """
    Perform bootstrap analysis.

    Args:
        data: Input data array
        n_iterations: Number of bootstrap iterations
        statistic: Statistic to compute ('mean' or 'median')
        strict_validation: If True, raise ValueError for empty arrays (for testing)

    Returns:
        Dictionary with bootstrap results
    """
    # Validate input - only raise error in strict mode (for tests)
    if len(data) == 0:
        if strict_validation:
            raise ValueError("Input data array cannot be empty")
        else:
            # Handle empty data gracefully for normal pipeline operation
            return {
                'mean': 0.0,
                'median': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0
            }

    # Use numba-accelerated function for bootstrap sampling
    if statistic == 'mean':
        boot_stats = _bootstrap_samples(data, n_iterations, use_mean=True)
    elif statistic == 'median':
        boot_stats = _bootstrap_samples(data, n_iterations, use_mean=False)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Compute confidence intervals using percentile method
    sorted_stats = np.sort(boot_stats)
    
    # Calculate percentiles manually
    lower_idx = int(0.025 * len(sorted_stats))
    upper_idx = int(0.975 * len(sorted_stats))
    ci_lower = sorted_stats[lower_idx]
    ci_upper = sorted_stats[upper_idx]
    
    return {
        'mean': float(_mean(boot_stats)),
        'median': float(sorted_stats[len(sorted_stats) // 2]),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }

def control_for_confounders(
    target_data: pl.DataFrame,
    control_data: pl.DataFrame,
    confounders: List[str]
) -> Dict[str, list]:
    """
    Control for confounding factors using regression.

    Args:
        target_data: DataFrame with target data
        control_data: DataFrame with control data
        confounders: List of confounder column names

    Returns:
        Dictionary with regression results
    """
    if target_data.height == 0 or control_data.height == 0:
        raise ValueError("Input data cannot be empty")

    # Combine data and create target indicator
    data = pl.concat([
        target_data.with_columns(pl.lit(True).alias('is_target')),
        control_data.with_columns(pl.lit(False).alias('is_target'))
    ])
    
    # Prepare regression variables
    X = sm.add_constant(data[confounders].to_numpy())
    y = data['is_target'].to_numpy()
    
    # Fit logistic regression
    model = sm.Logit(y, X)
    results = model.fit()
    
    return {
        'coefficients': results.params.tolist(),
        'p_values': results.pvalues.tolist(),
        'aic': float(results.aic),
        'bic': float(results.bic)
    }

def compute_enrichment_score(
    target_data: pl.DataFrame,
    control_data: pl.DataFrame,
    score_col: str = 'score'
) -> Tuple[float, float]:
    """
    Compute enrichment score and p-value.

    Args:
        target_data: DataFrame containing target data
        control_data: DataFrame containing control data
        score_col: Name of the score column

    Returns:
        Tuple of (enrichment_score, p_value)
    """
    if target_data.height == 0 or control_data.height == 0:
        raise ValueError("Input data cannot be empty")

    # Calculate enrichment score
    target_mean = float(target_data[score_col].mean())
    control_mean = float(control_data[score_col].mean())
    enrichment_score = target_mean - control_mean
    
    # Convert to arrays for comparison and statistical test
    target_array = target_data[score_col].to_numpy()
    control_array = control_data[score_col].to_numpy()
    
    # More comprehensive check for nearly identical data to avoid precision warnings
    if (_abs(target_mean - control_mean) < 1e-10 or
            _std(target_array) < 1e-10 and _std(control_array) < 1e-10):
        p_value = 1.0  # If the data is practically identical, there's no significant difference
    else:
        _, p_value = stats.ttest_ind(target_array, control_array, equal_var=False)
    
    return enrichment_score, float(p_value)

def calculate_significance(
    observed_score: float,
    null_scores: List[float],
    alpha: float = 0.05
) -> Tuple[float, bool]:
    """
    Calculate significance of observed score against null distribution.

    Args:
        observed_score: Observed enrichment score
        null_scores: List of scores from null model
        alpha: Significance level

    Returns:
        Tuple of (p-value, is_significant)
    """
    if not null_scores:
        raise ValueError("Null scores array cannot be empty")

    # Convert to array for Numba processing
    null_scores_array = np.array(null_scores)
    
    # Calculate empirical p-value using numba-accelerated function
    count_greater_equal = _calculate_significance_counts(observed_score, null_scores_array)
    p_value = count_greater_equal / len(null_scores_array)
    
    # Return as Python primitives
    is_significant = bool(p_value <= alpha)
    
    return float(p_value), is_significant