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
from scipy.spatial.distance import mahalanobis


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
    import numpy as np
    from scipy.spatial.distance import mahalanobis
    
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
    if save_intermediate:
        if intermediate_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            intermediate_dir = f"results/intermediate_mahalanobis_{timestamp}"
        
        # Create the directory if it doesn't exist
        os.makedirs(intermediate_dir, exist_ok=True)
    
    # Prepare features for distance calculation
    feature_cols = confounders.copy()
    
    # Early return if no confounders specified
    if not feature_cols:
        # Return empty DataFrame with expected schema
        schema = {
            "gene_id": pl.Utf8,
            "target_gene_id": pl.Utf8, 
            "mahalanobis_distance": pl.Float64,
            "initial_distance": pl.Float64,
            "sd_threshold_used": pl.Float64
        }
        return pl.DataFrame(schema=schema)
    
    # Extract feature data
    target_genes = gene_set_data.select(
        ["gene_id"] + ([chr_col] if chr_col in gene_set_data.columns else []) + 
        [col for col in confounders if col in gene_set_data.columns]
    )
    
    background_genes = all_genes_data.select(
        ["gene_id"] + ([chr_col] if chr_col in all_genes_data.columns else []) + 
        [col for col in confounders if col in all_genes_data.columns]
    )
    
    # Check if all confounders are present in both dataframes
    missing_confounders = [col for col in confounders 
                           if col not in target_genes.columns or col not in background_genes.columns]
    if missing_confounders:
        print(f"Warning: Missing confounder columns: {missing_confounders}")
        # Remove missing confounders from feature_cols
        feature_cols = [col for col in feature_cols if col not in missing_confounders]
        
        # If no valid confounders remain, return empty DataFrame
        if not feature_cols:
            schema = {
                "gene_id": pl.Utf8,
                "target_gene_id": pl.Utf8, 
                "mahalanobis_distance": pl.Float64,
                "initial_distance": pl.Float64,
                "sd_threshold_used": pl.Float64
            }
            return pl.DataFrame(schema=schema)
    
    # Filter background genes to exclude target genes
    background_genes = background_genes.filter(~pl.col("gene_id").is_in(exclude_gene_ids))
    
    if background_genes.height == 0:
        # If filtering removed all background genes, return empty DataFrame
        schema = {
            "gene_id": pl.Utf8,
            "target_gene_id": pl.Utf8, 
            "mahalanobis_distance": pl.Float64,
            "initial_distance": pl.Float64,
            "sd_threshold_used": pl.Float64
        }
        return pl.DataFrame(schema=schema)
    
    # Get chromosome information if available
    has_chr = chr_col in target_genes.columns and chr_col in background_genes.columns
    
    # Calculate covariance matrix for initial Mahalanobis distance
    # Combine features from both target and background genes
    try:
        all_features = pl.concat([
            target_genes.select(feature_cols),
            background_genes.select(feature_cols)
        ])
        
        # Convert to numpy array
        all_features_np = all_features.to_numpy()
        
        # Handle single feature case - ensure array is 2D
        if len(feature_cols) == 1:
            all_features_np = all_features_np.reshape(-1, 1)
            
        # Calculate mean and covariance
        cov_matrix = np.cov(all_features_np, rowvar=False)
        
        # Handle single feature case for covariance matrix
        if len(feature_cols) == 1:
            # If there's only one feature, cov returns a scalar, convert to 2D array
            cov_matrix = np.array([[cov_matrix]])
        
        # Add a small regularization term to ensure the matrix is invertible
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        
        # Calculate inverse covariance matrix
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if regular inverse fails
            inv_cov = np.linalg.pinv(cov_matrix)
    except Exception as e:
        print(f"Error calculating covariance matrix: {e}")
        # Return empty DataFrame with expected schema
        schema = {
            "gene_id": pl.Utf8,
            "target_gene_id": pl.Utf8, 
            "mahalanobis_distance": pl.Float64,
            "initial_distance": pl.Float64,
            "sd_threshold_used": pl.Float64
        }
        return pl.DataFrame(schema=schema)
    
    # Prepare to store results
    all_matched_data = []
    
    # Convert Polars DataFrames to lists of dictionaries for easier access
    target_gene_rows = target_genes.to_dicts()
    background_gene_rows = background_genes.to_dicts()
    
    print(f"Processing {len(target_gene_rows)} target genes against {len(background_gene_rows)} background genes")

    # Process each target gene
    for target_idx, target_row in enumerate(target_gene_rows):
        if target_idx % batch_size == 0 and target_idx > 0:
            print(f"Processed {target_idx} of {len(target_gene_rows)} target genes")
            
        target_id = target_row["gene_id"]
        
        # Ensure all feature columns exist in the target row
        if not all(col in target_row for col in feature_cols):
            print(f"Warning: Target gene {target_id} missing some feature columns, skipping")
            continue
            
        try:
            target_features = np.array([target_row[col] for col in feature_cols])
            
            # Reshape if only one feature to match format required by mahalanobis function
            if len(feature_cols) == 1:
                target_features = target_features.reshape(1)
        except Exception as e:
            print(f"Error extracting features for target gene {target_id}: {e}")
            continue
        
        # Get target chromosome if available
        target_chr = target_row.get(chr_col, None) if has_chr else None
        
        # Calculate Mahalanobis distances from this target to all background genes
        distances = []
        bg_indices = []
        
        for bg_idx, bg_row in enumerate(background_gene_rows):
            try:
                # Skip if any feature is missing
                if not all(col in bg_row for col in feature_cols):
                    continue
                    
                bg_features = np.array([bg_row[col] for col in feature_cols])
                
                # Reshape if only one feature
                if len(feature_cols) == 1:
                    bg_features = bg_features.reshape(1)
                    
                dist = mahalanobis(target_features, bg_features, inv_cov)
                distances.append(dist)
                bg_indices.append(bg_idx)
            except Exception as e:
                # Skip any problematic calculations
                continue
        
        if not distances:
            # No valid distances calculated
            continue
            
        distances = np.array(distances)
        bg_indices = np.array(bg_indices)
        
        # Find candidates within SD threshold
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # First try with initial threshold
        current_sd_threshold = sd_threshold
        threshold = mean_dist + current_sd_threshold * std_dist
        candidate_mask = distances <= threshold
        
        # If not enough candidates, try with double threshold
        if np.sum(candidate_mask) < max(10, n_matches):
            current_sd_threshold = 2.0 * sd_threshold
            threshold = mean_dist + current_sd_threshold * std_dist
            candidate_mask = distances <= threshold
        
        candidate_indices = bg_indices[candidate_mask]
        candidate_distances = distances[candidate_mask]
        
        # Save intermediate candidates if requested
        if save_intermediate and len(candidate_indices) > 0:
            candidate_rows = []
            for i, idx in enumerate(candidate_indices):
                bg_row = background_gene_rows[idx]
                bg_id = bg_row["gene_id"]
                candidate_rows.append({
                    "gene_id": bg_id,
                    "target_gene_id": target_id,
                    "initial_distance": float(candidate_distances[i])
                })
                
            if candidate_rows:
                candidate_df = pl.DataFrame(candidate_rows)
                candidate_file = Path(intermediate_dir) / f"candidates_{target_id}.csv"
                candidate_df.write_csv(candidate_file)
        
        # Recalculate distances with chromosome penalty
        final_distances = []
        final_indices = []
        
        for i, idx in enumerate(candidate_indices):
            bg_row = background_gene_rows[idx]
            bg_id = bg_row["gene_id"]
            bg_features = np.array([bg_row[col] for col in feature_cols])
            
            # Reshape if only one feature
            if len(feature_cols) == 1:
                bg_features = bg_features.reshape(1)
            
            # Apply chromosome penalty if chromosomes are different and both are defined
            pairwise_penalty = 0.0
            if has_chr and target_chr is not None:
                bg_chr = bg_row.get(chr_col, None)
                if bg_chr is not None and target_chr != bg_chr:
                    # Apply penalty for different chromosomes
                    pairwise_penalty = -1.0  # Simplified penalty
            
            # Create augmented feature vectors with penalty
            aug_target_features = np.append(target_features, [pairwise_penalty])
            aug_bg_features = np.append(bg_features, [0.0])
            
            # Create augmented covariance matrix
            aug_dim = cov_matrix.shape[0] + 1
            aug_cov = np.zeros((aug_dim, aug_dim))
            aug_cov[:cov_matrix.shape[0], :cov_matrix.shape[0]] = cov_matrix
            aug_cov[-1, -1] = 1.0
            
            # Calculate inverse
            try:
                aug_inv_cov = np.linalg.inv(aug_cov)
            except np.linalg.LinAlgError:
                aug_inv_cov = np.linalg.pinv(aug_cov)
            
            # Calculate final distance
            try:
                final_dist = mahalanobis(aug_bg_features, aug_target_features, aug_inv_cov)
                final_distances.append(final_dist)
                final_indices.append(idx)
            except Exception as e:
                # Skip problematic calculations
                continue
        
        if not final_distances:
            continue
            
        final_distances = np.array(final_distances)
        final_indices = np.array(final_indices)
        
        # Sort by final distance and get top matches
        sorted_indices = np.argsort(final_distances)
        top_n = min(n_matches, len(sorted_indices))
        top_indices = sorted_indices[:top_n]
        
        # Add matches to results
        for idx in top_indices:
            bg_row = background_gene_rows[final_indices[idx]]
            bg_id = bg_row["gene_id"]
            final_dist = final_distances[idx]
            
            # Find the initial distance for this gene
            candidate_idx = np.where(candidate_indices == final_indices[idx])[0]
            if len(candidate_idx) > 0:
                initial_dist = candidate_distances[candidate_idx[0]]
            else:
                initial_dist = float('nan')
            
            match_data = {
                "gene_id": bg_id,
                "target_gene_id": target_id,
                "mahalanobis_distance": float(final_dist),
                "initial_distance": float(initial_dist),
                "sd_threshold_used": float(current_sd_threshold)
            }
            
            # Add confounder values
            for col in confounders:
                if col in bg_row:
                    match_data[col] = bg_row[col]
            
            all_matched_data.append(match_data)
    
    end_time = time.time()
    print(f"Processed all {len(target_gene_rows)} target genes in {end_time - start_time:.2f} seconds")
    
    # Convert results to DataFrame
    if not all_matched_data:
        # Return empty DataFrame with expected columns if no matches found
        schema = {
            "gene_id": pl.Utf8,
            "target_gene_id": pl.Utf8, 
            "mahalanobis_distance": pl.Float64,
            "initial_distance": pl.Float64,
            "sd_threshold_used": pl.Float64
        }
        for col in confounders:
            if col in all_genes_data.columns:
                schema[col] = all_genes_data[col].dtype
            else:
                schema[col] = pl.Float64
        return pl.DataFrame(schema=schema)
    
    return pl.DataFrame(all_matched_data)
