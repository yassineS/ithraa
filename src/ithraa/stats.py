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



#  Core numba-optimised functions for inner loops and heavy calculations

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
    
    # Pre-allocate numpy arrays for better memory management and vectorisation
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
    # For small arrays, direct counting is faster than parallelsation overhead
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

@nb.njit
def _calculate_mahalanobis_distance(vec1, vec2, inv_cov):
    """
    Calculate Mahalanobis distance between two vectors using numba for performance.
    
    Args:
        vec1: First vector
        vec2: Second vector
        inv_cov: Inverse covariance matrix
        
    Returns:
        Mahalanobis distance (scalar)
    """
    diff = vec1 - vec2
    return np.sqrt(diff @ inv_cov @ diff.T)

@nb.njit(parallel=True)
def _calculate_mahalanobis_distances_batch(target_features, background_features, inv_cov, exclude_indices):
    """
    Calculate Mahalanobis distances for one target against all background features.
    Uses parallel processing for better performance.
    
    Args:
        target_features: Feature vector for the target gene
        background_features: Feature matrix for all background genes
        inv_cov: Inverse covariance matrix
        exclude_indices: Set of indices to exclude (as a sorted array)
        
    Returns:
        Tuple of (distances, valid_indices)
    """
    n_bg = background_features.shape[0]
    max_size = n_bg  # Maximum possible number of valid genes
    
    # Pre-allocate arrays
    distances = np.zeros(max_size, dtype=np.float64)
    valid_indices = np.zeros(max_size, dtype=np.int64)
    
    # Keep track of how many valid items we've found
    valid_count = 0
    
    # Calculate distances in parallel
    for j in nb.prange(n_bg):
        # Check if this index should be excluded
        exclude = False
        for idx in exclude_indices:
            if j == idx:
                exclude = True
                break
        
        if not exclude:
            # Calculate distance
            diff = background_features[j] - target_features
            dist = np.sqrt(np.sum(diff * (inv_cov @ diff)))
            
            # Store the result
            distances[valid_count] = dist
            valid_indices[valid_count] = j
            valid_count += 1
    
    # Return only the valid entries
    return distances[:valid_count], valid_indices[:valid_count]

@nb.njit
def _find_best_matches(distances, indices, n_matches):
    """
    Find the indices of the n_matches smallest distances.
    
    Args:
        distances: Array of distances
        indices: Array of corresponding indices
        n_matches: Number of matches to find
        
    Returns:
        Array of indices for the best matches
    """
    if len(distances) <= n_matches:
        # If we have fewer distances than requested matches, return all sorted by distance
        sorted_idx = np.argsort(distances)
        return indices[sorted_idx]
    else:
        # Otherwise, use argpartition to efficiently find the n smallest distances
        best_idx = np.argpartition(distances, n_matches)[:n_matches]
        # Sort these by distance for consistent results
        best_sorted_idx = best_idx[np.argsort(distances[best_idx])]
        return indices[best_sorted_idx]

@nb.njit(parallel=True)
def _batch_mahalanobis_distances(X, Y, inv_cov):
    """
    Calculate Mahalanobis distances between all pairs of points in X and Y.
    
    This is a highly optimized function that uses matrix operations and
    parallel processing to calculate Mahalanobis distances between all pairs
    of points in X and Y.
    
    Args:
        X: Array of shape (n, p) containing n points in p-dimensional space
        Y: Array of shape (m, p) containing m points in p-dimensional space
        inv_cov: Inverse covariance matrix of shape (p, p)
        
    Returns:
        Array of shape (n, m) containing distances between all pairs of points
    """
    n = X.shape[0]
    m = Y.shape[0]
    result = np.zeros((n, m))
    
    # Calculate distances in parallel across rows of X
    for i in nb.prange(n):
        x_i = X[i]
        for j in range(m):
            diff = x_i - Y[j]
            result[i, j] = np.sqrt(np.sum(diff * (inv_cov @ diff)))
    
    return result

@nb.njit(parallel=True)
def _find_candidates_by_sd_threshold(distances_matrix, sd_threshold, min_candidates=10):
    """
    Efficiently find candidate genes based on standard deviation thresholds.
    Uses highly optimized parallel processing to handle multiple targets at once.
    
    Args:
        distances_matrix: Matrix of shape (n_targets, n_background) containing distances
        sd_threshold: Initial standard deviation threshold
        min_candidates: Minimum number of candidates to find (will increase threshold if needed)
        
    Returns:
        List of arrays containing candidate indices for each target
    """
    n_targets = distances_matrix.shape[0]
    n_bg = distances_matrix.shape[1]
    result = []
    
    # Process each target gene in parallel
    for i in nb.prange(n_targets):
        target_distances = distances_matrix[i]
        
        # Calculate mean and std using optimized methods (faster than np.mean and np.std)
        dist_sum = 0.0
        for j in range(n_bg):
            dist_sum += target_distances[j]
        mean_dist = dist_sum / n_bg
        
        # Calculate standard deviation
        var_sum = 0.0
        for j in range(n_bg):
            diff = target_distances[j] - mean_dist
            var_sum += diff * diff
        std_dist = np.sqrt(var_sum / n_bg)
        
        # First attempt with initial threshold
        threshold = mean_dist + sd_threshold * std_dist
        
        # Count candidates (faster than creating a mask then counting)
        count = 0
        for j in range(n_bg):
            if target_distances[j] <= threshold:
                count += 1
        
        # If not enough candidates, try with double threshold
        if count < min_candidates:
            threshold = mean_dist + 2.0 * sd_threshold * std_dist
            
        # Create array just large enough to hold the candidates
        # Count again to determine size
        count = 0
        for j in range(n_bg):
            if target_distances[j] <= threshold:
                count += 1
                
        # Allocate array of exact size needed
        candidates = np.empty(count, dtype=np.int32)
        
        # Fill array with candidate indices
        idx = 0
        for j in range(n_bg):
            if target_distances[j] <= threshold:
                candidates[idx] = j
                idx += 1
        
        result.append(candidates)
    
    return result

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


def match_genes_mahalanobis(
    gene_set_data: pl.DataFrame,
    all_genes_data: pl.DataFrame,
    distance_col: str = "distance",
    confounders: List[str] = None,
    exclude_gene_ids: List[str] = None,
    n_matches: int = 1,
    sd_threshold: float = 1.0,
    save_intermediate: bool = True,
    intermediate_dir: str = None,
    chr_col: str = "chrom",
    batch_size: int = 100
) -> pl.DataFrame:
    """
    Match each gene in gene_set with other genes using Mahalanobis distance.
    
    Implements a multi-step Mahalanobis distance calculation:
    1. Build matrix of pairwise Mahalanobis distances using confounding factors
    2. For each gene, find candidate control genes within SD_threshold standard deviations
    3. Recalculate Mahalanobis distances with additional penalty for physical proximity
    4. If fewer than 10 matches are found, retry with 2x standard deviation threshold
    
    Args:
        gene_set_data: DataFrame containing genes to be matched (target genes)
        all_genes_data: DataFrame containing all genes to match against (background)
        distance_col: Column name containing the distance metric
        confounders: List of column names to use as confounding factors
        exclude_gene_ids: List of gene IDs to exclude from matching (e.g., the gene set itself)
        n_matches: Number of matching genes to return per gene in gene_set
        sd_threshold: Standard deviation threshold for initial candidate selection
        save_intermediate: Whether to save intermediate candidate sets
        intermediate_dir: Directory to save intermediate files
        chr_col: Column containing chromosome information
        batch_size: Number of genes to process at once (for matrix operations)

    Returns:
        DataFrame containing matched genes with their matching target gene IDs
    """
    import os
    from datetime import datetime
    from pathlib import Path
    import time
    
    start_time = time.time()
    
    if gene_set_data.height == 0 or all_genes_data.height == 0:
        raise ValueError("Input data cannot be empty")

    # Set default empty list if confounders is None
    confounders = confounders or []
    exclude_gene_ids = exclude_gene_ids or set()
    
    # Convert exclude_gene_ids to a set for faster lookups
    if not isinstance(exclude_gene_ids, set):
        exclude_gene_ids = set(exclude_gene_ids)
    
    # Set up intermediate directory if saving intermediate files
    if save_intermediate and intermediate_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        intermediate_dir = f"results/intermediate_mahalanobis_{timestamp}"
    
    if save_intermediate:
        os.makedirs(intermediate_dir, exist_ok=True)
    
    # Prepare features for distance calculation
    feature_cols = confounders.copy()
    
    # Extract feature data
    target_genes = gene_set_data.select(
        ["gene_id"] + ([chr_col] if chr_col in gene_set_data.columns else []) + confounders
    )
    
    background_genes = all_genes_data.select(
        ["gene_id"] + ([chr_col] if chr_col in all_genes_data.columns else []) + confounders
    )
    
    # Convert to numpy arrays for faster computation
    target_features = target_genes.select(feature_cols).to_numpy()
    target_ids = target_genes["gene_id"].to_numpy()
    background_features = background_genes.select(feature_cols).to_numpy()
    background_ids = background_genes["gene_id"].to_numpy()
    
    # Create mask for excluding genes from background
    valid_bg_mask = np.ones(len(background_ids), dtype=bool)
    for i, bg_id in enumerate(background_ids):
        if bg_id in exclude_gene_ids:
            valid_bg_mask[i] = False
            
    # Filter background features and IDs
    valid_bg_indices = np.where(valid_bg_mask)[0]
    valid_bg_features = background_features[valid_bg_mask]
    valid_bg_ids = background_ids[valid_bg_mask]
        
    print(f"Processing {len(target_ids)} target genes against {len(valid_bg_ids)} background genes")
    
    # Calculate covariance matrix for initial Mahalanobis distance
    # Use a regularized covariance estimation to ensure numerical stability
    all_features = np.vstack([target_features, valid_bg_features])
    
    # Calculate mean and covariance
    cov_matrix = np.cov(all_features, rowvar=False)
    
    # Add a small regularization term to ensure the matrix is invertible
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    
    # Calculate inverse covariance matrix
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if regular inverse fails
        inv_cov = np.linalg.pinv(cov_matrix)
    
    # Prepare to store results
    all_matched_data = []
    
    # Create dictionary to map gene_id to chromosome if chromosome info is available
    chr_map = {}
    if chr_col in all_genes_data.columns:
        for i, gene_id in enumerate(background_ids):
            chr_map[gene_id] = background_genes[chr_col][i]
    
    # Calculate pairwise distances if distance_col is present in the data
    has_distances = distance_col in all_genes_data.columns
    distances_dict = {}
    
    if has_distances:
        # Extract distance data
        distance_df = all_genes_data.select(["gene_id", distance_col])
        # Create a dictionary for quick lookups of pairwise distances
        all_genes = set(background_ids)
        for g1 in all_genes:
            distances_dict[g1] = {}
            for g2 in all_genes:
                if g1 != g2:
                    # In a real implementation, this would look up the distance from a precomputed matrix
                    # For now, we'll use a placeholder value from the distance column
                    filtered = distance_df.filter((pl.col("gene_id") == g1) | (pl.col("gene_id") == g2))
                    if filtered.height >= 2:
                        distances_dict[g1][g2] = abs(filtered[distance_col][0] - filtered[distance_col][1])
                    else:
                        distances_dict[g1][g2] = 0.0
    
    # Find the maximum pairwise distance for chromosome penalty
    max_pairwise_distance = 0.0
    if has_distances:
        for g1 in distances_dict:
            for g2 in distances_dict[g1]:
                max_pairwise_distance = max(max_pairwise_distance, distances_dict[g1][g2])
    
    # Process targets in batches for matrix operations
    n_targets = len(target_features)
    dist_calc_time = 0
    sd_calc_time = 0
    penalty_calc_time = 0
    
    for batch_start in range(0, n_targets, batch_size):
        batch_end = min(batch_start + batch_size, n_targets)
        batch_size_actual = batch_end - batch_start
        target_batch_features = target_features[batch_start:batch_end]
        target_batch_ids = target_ids[batch_start:batch_end]
        
        # Calculate initial distances between all targets in batch and all background genes
        t0 = time.time()
        initial_distances = _batch_mahalanobis_distances(target_batch_features, valid_bg_features, inv_cov)
        t1 = time.time()
        dist_calc_time += (t1 - t0)
        
        # Use our optimized SD-threshold function to find candidates for all targets in the batch
        t0 = time.time()
        candidates_by_target = _find_candidates_by_sd_threshold(initial_distances, sd_threshold, min_candidates=10)
        t1 = time.time()
        sd_calc_time += (t1 - t0)
        
        # Process each target in the batch with its candidates
        t0 = time.time()
        for i in range(batch_size_actual):
            target_id = target_batch_ids[i]
            target_feature = target_batch_features[i]
            candidate_indices = candidates_by_target[i]
            target_chr = chr_map.get(target_id, None) if chr_col in target_genes.columns else None
            
            # If no candidates, skip this target
            if len(candidate_indices) == 0:
                continue
            
            # Get candidate info
            # Get the actual SD threshold used (1x or 2x) based on number of candidates
            current_sd_threshold = 2.0 * sd_threshold if len(candidate_indices) >= 10 else sd_threshold
            
            # Get distances and features for candidates
            candidate_ids = valid_bg_ids[candidate_indices]
            candidate_features = valid_bg_features[candidate_indices]
            candidate_distances = initial_distances[i][candidate_indices]
            
            # Save intermediate candidates if requested
            if save_intermediate and len(candidate_ids) > 0:
                candidate_df = pl.DataFrame({
                    "gene_id": candidate_ids,
                    "target_gene_id": [target_id] * len(candidate_ids),
                    "initial_distance": candidate_distances
                })
                
                # Save to file
                candidate_file = Path(intermediate_dir) / f"candidates_{target_id}.csv"
                candidate_df.write_csv(candidate_file)
            
            # Recalculate distances with penalty
            final_distances = []
            
            for j in range(len(candidate_indices)):
                bg_id = candidate_ids[j]
                bg_chr = chr_map.get(bg_id, None) if chr_map else None
                
                # Add pairwise distance penalty
                pairwise_penalty = 0.0
                if has_distances:
                    if target_chr is not None and bg_chr is not None and target_chr != bg_chr:
                        pairwise_penalty = -max_pairwise_distance
                    elif target_id in distances_dict and bg_id in distances_dict[target_id]:
                        pairwise_penalty = -distances_dict[target_id][bg_id]
                
                # Create augmented feature vectors with the penalty
                augmented_feature_target = np.append(target_feature, [pairwise_penalty])
                augmented_feature_bg = np.append(candidate_features[j], [0.0])
                
                # Create augmented covariance matrix with the penalty dimension
                aug_dim = cov_matrix.shape[0] + 1
                aug_cov = np.zeros((aug_dim, aug_dim))
                aug_cov[:cov_matrix.shape[0], :cov_matrix.shape[0]] = cov_matrix
                aug_cov[-1, -1] = 1.0  # Variance for penalty dimension
                
                # Calculate inverse of augmented covariance
                try:
                    aug_inv_cov = np.linalg.inv(aug_cov)
                except np.linalg.LinAlgError:
                    aug_inv_cov = np.linalg.pinv(aug_cov)
                
                # Calculate final Mahalanobis distance with penalty
                diff = augmented_feature_bg - augmented_feature_target
                final_dist = np.sqrt(diff @ aug_inv_cov @ diff.T)
                final_distances.append(final_dist)
            
            # Check if we have any candidates after recalculation
            if not final_distances:
                continue
            
            # Convert to numpy for faster sorting
            final_distances = np.array(final_distances)
            
            # Sort candidates by final distance and take top n
            top_indices = np.argsort(final_distances)[:n_matches]
            
            # Add matches to results
            for idx in top_indices:
                # Get matched gene details
                matched_id = candidate_ids[idx]
                final_dist = final_distances[idx]
                initial_dist = candidate_distances[idx]
                
                # Create a dictionary with match data
                match_data = {
                    "gene_id": matched_id,
                    "target_gene_id": target_id,
                    "mahalanobis_distance": float(final_dist),
                    "initial_distance": float(initial_dist),
                    "sd_threshold_used": float(current_sd_threshold)
                }
                
                # Add factor values from background data
                matched_row = background_genes.filter(pl.col("gene_id") == matched_id)
                if matched_row.height > 0:
                    for col in confounders:
                        if col in matched_row.columns:
                            match_data[col] = matched_row[col][0]
                    
                    # Add distance column if available
                    if distance_col in all_genes_data.columns:
                        matched_row_full = all_genes_data.filter(pl.col("gene_id") == matched_id)
                        if matched_row_full.height > 0 and distance_col in matched_row_full.columns:
                            match_data[distance_col] = matched_row_full[distance_col][0]
                        
                all_matched_data.append(match_data)
        
        t1 = time.time()
        penalty_calc_time += (t1 - t0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Timing breakdown: ")
    print(f"  - Initial distance calculation: {dist_calc_time:.2f}s ({dist_calc_time/total_time*100:.1f}%)")
    print(f"  - SD-threshold candidate selection: {sd_calc_time:.2f}s ({sd_calc_time/total_time*100:.1f}%)")
    print(f"  - Penalty application and final selection: {penalty_calc_time:.2f}s ({penalty_calc_time/total_time*100:.1f}%)")
    print(f"  - Total processing time: {total_time:.2f}s")
    
    # Convert results to DataFrame
    if not all_matched_data:
        # Return empty DataFrame with expected columns if no matches found
        return pl.DataFrame(schema={
            "gene_id": pl.Utf8,
            "target_gene_id": pl.Utf8, 
            "mahalanobis_distance": pl.Float64,
            "initial_distance": pl.Float64,
            "sd_threshold_used": pl.Float64,
            **{col: all_genes_data[col].dtype if col in all_genes_data.schema else pl.Float64 
               for col in confounders + ([distance_col] if distance_col in all_genes_data.columns else [])}
        })
    
    return pl.DataFrame(all_matched_data)
