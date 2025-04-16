"""
Statistical analysis functions for the gene set enrichment pipeline.
"""

from typing import List, Dict, Union, Optional, Tuple
import numpy as np
import polars as pl
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

def calculate_enrichment(target_counts: np.ndarray, control_counts: np.ndarray) -> Dict[str, float]:
    """
    Calculate enrichment statistics with confidence intervals.

    Args:
        target_counts: Array of target counts
        control_counts: Array of control counts

    Returns:
        Dictionary with enrichment statistics including CI and raw values
    """
    if len(target_counts) == 0 or len(control_counts) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Calculate means and sample size for target and control groups
    target_mean = np.mean(target_counts)
    control_mean = np.mean(control_counts)
    target_size = len(target_counts)
    control_size = len(control_counts)
    
    # Calculate enrichment ratio
    if control_mean == 0:
        enrichment_ratio = np.nan
    else:
        enrichment_ratio = target_mean / control_mean
    
    # Calculate standard deviations for confidence intervals
    target_std = np.std(target_counts, ddof=1)  # Use ddof=1 for sample standard deviation
    control_std = np.std(control_counts, ddof=1)
    
    # Calculate 95% confidence intervals for the control mean
    # Use t-distribution for small samples
    alpha = 0.05  # 95% confidence interval
    control_t_val = stats.t.ppf(1 - alpha/2, control_size - 1)
    control_margin = control_t_val * (control_std / np.sqrt(control_size))
    control_ci_low = control_mean - control_margin
    control_ci_high = control_mean + control_margin
    
    # If the values are very close, or the standard deviation is near zero,
    # use a more conservative estimate for the CI
    if control_std < 1e-10:
        control_ci_low = control_mean * 0.9
        control_ci_high = control_mean * 1.1
    
    # Check if means are nearly identical to avoid precision warnings
    if np.abs(target_mean - control_mean) < 1e-10:
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

def perform_fdr_analysis(p_values: np.ndarray, alpha: float = 0.05) -> Dict[str, np.ndarray]:
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
        'reject': reject.astype(bool),  # Convert to Python bool
        'pvals_corrected': pvals_corrected
    }

def bootstrap_analysis(
    data: np.ndarray,
    n_iterations: int = 1000,
    statistic: str = 'mean'
) -> Dict[str, float]:
    """
    Perform bootstrap analysis.

    Args:
        data: Input data array
        n_iterations: Number of bootstrap iterations
        statistic: Statistic to compute ('mean' or 'median')

    Returns:
        Dictionary with bootstrap results
    """
    if len(data) == 0:
        raise ValueError("Input data array cannot be empty")

    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Compute statistic on original data
    orig_stat = stat_func(data)
    
    # Bootstrap
    boot_stats = []
    for _ in range(n_iterations):
        boot_sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(boot_sample))
    
    # Compute confidence intervals
    ci_lower = np.percentile(boot_stats, 2.5)
    ci_upper = np.percentile(boot_stats, 97.5)
    
    return {
        'mean': float(np.mean(boot_stats)),
        'median': float(np.median(boot_stats)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }

def control_for_confounders(
    target_data: pl.DataFrame,
    control_data: pl.DataFrame,
    confounders: List[str]
) -> Dict[str, np.ndarray]:
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
        'coefficients': results.params,
        'p_values': results.pvalues,
        'aic': results.aic,
        'bic': results.bic
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
    
    # Convert to numpy arrays for comparison and statistical test
    target_array = target_data[score_col].to_numpy()
    control_array = control_data[score_col].to_numpy()
    
    # More comprehensive check for nearly identical data to avoid precision warnings
    if (np.abs(target_mean - control_mean) < 1e-10 or 
            np.allclose(target_array, control_array, rtol=1e-10, atol=1e-10) or
            (np.std(target_array) < 1e-10 and np.std(control_array) < 1e-10)):
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

    # Convert to numpy array
    null_scores = np.array(null_scores)
    
    # Calculate empirical p-value
    p_value = np.mean(null_scores >= observed_score)
    
    # Convert numpy.bool_ to Python bool
    is_significant = bool(p_value <= alpha)
    
    return float(p_value), is_significant