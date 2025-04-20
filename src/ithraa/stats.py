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
    n_matches: int = 1
) -> pl.DataFrame:
    """
    Match each gene in gene_set with other genes using Mahalanobis distance.
    
    The distance is calculated in a hyperspace defined by the negative distance
    between genes and additional confounding factors. This allows finding control
    genes that are similar to target genes in terms of both physical distance and
    confounding factors.

    Args:
        gene_set_data: DataFrame containing genes to be matched (target genes)
        all_genes_data: DataFrame containing all genes to match against (background)
        distance_col: Column name containing the distance metric (will be negated)
        confounders: List of column names to use as confounding factors
        exclude_gene_ids: List of gene IDs to exclude from matching (e.g., the gene set itself)
        n_matches: Number of matching genes to return per gene in gene_set

    Returns:
        DataFrame containing matched genes with their matching target gene IDs
    """
    if gene_set_data.height == 0 or all_genes_data.height == 0:
        raise ValueError("Input data cannot be empty")

    # Set default empty list if confounders is None
    confounders = confounders or []
    exclude_gene_ids = exclude_gene_ids or []

    # Prepare features for distance calculation
    feature_cols = confounders.copy()
    
    # Negate the distance column (for proper Mahalanobis calculation)
    # We negate because smaller distances mean genes are closer/more similar
    target_genes = gene_set_data.select(
        [distance_col] + confounders
    ).with_columns([
        pl.col(distance_col).mul(-1).alias(f"neg_{distance_col}")
    ])
    
    background_genes = all_genes_data.select(
        ["gene_id", distance_col] + confounders
    ).with_columns([
        pl.col(distance_col).mul(-1).alias(f"neg_{distance_col}")
    ])
    
    # Replace distance column with negated version
    feature_cols.append(f"neg_{distance_col}")
    
    # Convert to numpy arrays for faster computation
    target_features = target_genes.select(feature_cols).to_numpy()
    background_features = background_genes.select(feature_cols).to_numpy()
    background_ids = background_genes["gene_id"].to_numpy()
    
    # Calculate covariance matrix for Mahalanobis distance
    # Use a regularised covariance estimation to ensure numerical stability
    # Combine target and background for better estimation of covariance
    all_features = np.vstack([target_features, background_features])
    
    # Calculate mean and covariance
    mean_vector = np.mean(all_features, axis=0)
    cov_matrix = np.cov(all_features, rowvar=False)
    
    # Add a small regularisation term to ensure the matrix is invertible
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    
    # Calculate inverse covariance matrix
    try:
        inv_cov = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if regular inverse fails
        inv_cov = np.linalg.pinv(cov_matrix)
    
    # Prepare to store results
    matched_genes = []
    
    # For each gene in gene_set, find n_matches closest genes by Mahalanobis distance
    for i, target_gene_features in enumerate(target_features):
        target_gene_id = gene_set_data["gene_id"][i]
        
        # Calculate Mahalanobis distances for all background genes
        distances = []
        valid_indices = []
        
        for j, bg_features in enumerate(background_features):
            bg_id = background_ids[j]
            
            # Skip if this gene is in the exclude list or is the same as target
            if bg_id in exclude_gene_ids or bg_id == target_gene_id:
                continue
                
            # Calculate Mahalanobis distance
            diff = bg_features - target_gene_features
            mahala_dist = np.sqrt(diff @ inv_cov @ diff.T)
            
            distances.append(mahala_dist)
            valid_indices.append(j)
        
        # Check if we found any valid matches
        if not distances:
            continue
            
        # Find indices of n_matches smallest distances
        distances = np.array(distances)
        valid_indices = np.array(valid_indices)
        
        if len(distances) <= n_matches:
            best_indices = np.argsort(distances)
        else:
            best_indices = np.argpartition(distances, n_matches)[:n_matches]
        
        # Get the actual indices in the background genes
        match_indices = valid_indices[best_indices]
        
        # Add matches to results
        for match_idx in match_indices:
            matched_gene_id = background_ids[match_idx]
            matched_gene_row = background_genes.filter(pl.col("gene_id") == matched_gene_id).to_dicts()[0]
            matched_gene_row["target_gene_id"] = target_gene_id
            matched_gene_row["mahalanobis_distance"] = float(distances[np.where(valid_indices == match_idx)[0][0]])
            matched_genes.append(matched_gene_row)
    
    # Convert results to DataFrame
    if not matched_genes:
        # Return empty DataFrame with expected columns if no matches found
        return pl.DataFrame(schema={
            "gene_id": pl.Utf8,
            "target_gene_id": pl.Utf8, 
            "mahalanobis_distance": pl.Float64,
            **{col: all_genes_data[col].dtype for col in confounders + [distance_col]},
            f"neg_{distance_col}": pl.Float64
        })
    
    return pl.DataFrame(matched_genes)

def visualise_gene_matching(
    gene_set_data: pl.DataFrame,
    matched_genes: pl.DataFrame,
    method: str = "pca",
    confounders: List[str] = None,
    distance_col: str = "distance",
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate visualisation data to compare gene set and matched genes.
    
    Args:
        gene_set_data: DataFrame containing the target gene set
        matched_genes: DataFrame containing matched genes (from match_genes_mahalanobis)
        method: Visualisation method, either "pca" or "tsne"
        confounders: List of confounder columns used for dimensionality reduction
        distance_col: Column name containing the distance metric
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with "coordinates", "labels", and "variance_explained" (for PCA only)
    """
    if gene_set_data.height == 0 or matched_genes.height == 0:
        raise ValueError("Input data cannot be empty")
        
    # Set default empty list if confounders is None
    confounders = confounders or []
    
    # Prepare features for dimensionality reduction
    feature_cols = confounders.copy()
    feature_cols.append(distance_col)
    
    # Create combined dataset with labels
    target_genes = gene_set_data.select(feature_cols).with_columns(
        pl.lit("target").alias("group")
    )
    
    matched = matched_genes.select(feature_cols).with_columns(
        pl.lit("matched").alias("group")
    )
    
    combined_data = pl.concat([target_genes, matched])
    
    # Extract features and labels
    features = combined_data.select(feature_cols).to_numpy()
    labels = combined_data["group"].to_numpy()
    
    # Apply dimensionality reduction
    if method.lower() == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=random_state)
        components = reducer.fit_transform(features)
        variance_explained = reducer.explained_variance_ratio_
        
        return {
            "coordinates": components,
            "labels": labels,
            "variance_explained": variance_explained
        }
    
    elif method.lower() == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state)
        components = reducer.fit_transform(features)
        
        return {
            "coordinates": components,
            "labels": labels,
            "variance_explained": None
        }
    
    else:
        raise ValueError(f"Unknown visualisation method: {method}. Use 'pca' or 'tsne'.")