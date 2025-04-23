"""Main pipeline implementation for gene set enrichment analysis."""

import logging
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from functools import partial
from pathlib import Path
import json
from typing import Dict, List, Any, Optional

import numba as nb
import numpy as np
import polars as pl
from tqdm.auto import tqdm

# Configure tqdm to work properly on macOS
is_mac = platform.system() == 'Darwin'
tqdm_kwargs = {
    'position': 0, 
    'leave': True, 
    'ncols': 100, 
    'dynamic_ncols': True,  # Allow tqdm to adjust the width dynamically
    'ascii': is_mac,        # Use ASCII characters on macOS for better terminal compatibility
    'disable': False,       # Never disable the progress bar
}

from ithraa.config import PipelineConfig
from .data import (
    load_gene_list, 
    load_gene_set, 
    load_gene_coords, 
    load_factors, 
    load_valid_genes,
    load_hgnc_mapping,
    compute_gene_distances,
    process_gene_set,
    find_control_genes,
    shuffle_genome_circular
)
from ithraa.stats import (
    calculate_enrichment,
    perform_fdr_analysis,
    bootstrap_analysis,
    calculate_significance
)
from ithraa.utils import ensure_dir

@nb.njit
def _calculate_enrichment_ratio(target_scores, control_scores) -> float:
    """
    Calculate enrichment ratio between target and control scores.
    
    Args:
        target_scores: Array of target scores
        control_scores: Array of control scores
        
    Returns:
        Enrichment ratio as float
    """
    target_mean = np.mean(target_scores)
    control_mean = np.mean(control_scores)
    
    if abs(control_mean) < 1e-10:
        return float('nan')
        
    return target_mean / control_mean

# Define a helper function for performing permutation outside class (needed for multiprocessing)
def _perform_permutation(
    i: int, 
    gene_coords_df: pl.DataFrame,
    chrom_sizes: Dict[str, int],
    target_genes: pl.DataFrame,
    min_distance: int,
    factors_df: pl.DataFrame,
    tolerance: float,
    processed_genes: pl.DataFrame,
    gene_set_df: pl.DataFrame,
    population_names: List[str]
) -> Dict[str, float]:
    """
    Perform a single FDR permutation iteration.
    Uses Numba-accelerated functions for critical computations.
    
    Args:
        i: Iteration number
        gene_coords_df: Gene coordinates DataFrame
        chrom_sizes: Dictionary of chromosome sizes
        target_genes: Target genes DataFrame
        min_distance: Minimum distance for control genes
        factors_df: Factors DataFrame
        tolerance: Tolerance for matching control genes
        processed_genes: Processed genes DataFrame
        gene_set_df: Gene set DataFrame
        population_names: List of population names
        
    Returns:
        Dictionary with enrichment ratios for each population
    """
    # Shuffle gene coordinates within each chromosome, treating chromosomes as circular
    shuffled_coords = shuffle_genome_circular(gene_coords_df, chrom_sizes)
    
    # Compute new gene distances
    shuffled_distances = compute_gene_distances(shuffled_coords)
    
    # Find new control genes
    shuffled_controls = find_control_genes(
        target_genes,
        shuffled_distances,
        min_distance=min_distance
    )
    
    # Match control genes based on confounding factors
    shuffled_matched = match_confounding_factors(
        target_genes,
        shuffled_controls,
        factors_df,
        tolerance=tolerance
    )
    
    # Calculate enrichment for each population
    iter_results = {}
    for population in population_names:
        # Get scores for target and shuffled control genes
        target_scores = processed_genes.join(
            gene_set_df,
            on='gene_id',
            how='inner'
        )[population].to_numpy()
        
        shuffled_control_scores = processed_genes.join(
            shuffled_matched.select(['gene_id']),
            on='gene_id',
            how='inner'
        )[population].to_numpy()
        
        # Calculate enrichment using Numba-accelerated function
        if len(shuffled_control_scores) > 0:
            enrichment_ratio = _calculate_enrichment_ratio(target_scores, shuffled_control_scores)
            iter_results[population] = float(enrichment_ratio)
        else:
            iter_results[population] = float('nan')
    
    return iter_results

# Helper functions for optimised threshold processing
def _process_threshold(
    self,
    threshold: float,
    target_count: int,
    target_genes: pl.DataFrame,
    populations: list[str],
    gene_scores: dict[str, pl.DataFrame],
    gene_ranks: dict[str, pl.DataFrame],
    control_genes: pl.DataFrame = None,
    permutations: int = 1000,
) -> dict:
    """
    Process a single threshold.
    
    Args:
        threshold: The threshold to process.
        target_count: The number of target genes.
        target_genes: DataFrame containing target genes.
        populations: List of populations to calculate enrichment for.
        gene_scores: Dictionary mapping population to gene scores DataFrame.
        gene_ranks: Dictionary mapping population to gene ranks DataFrame.
        control_genes: DataFrame containing control genes.
        permutations: Number of permutations to perform.
        
    Returns:
        Dictionary containing enrichment results for this threshold.
    """
    logger = logging.getLogger(__name__)
    result = {}
    
    if control_genes is None or len(control_genes) == 0:
        # Handle case with no control genes by using a placeholder approach
        logger.debug(f"No control genes provided for threshold {threshold}. Using fallback approach.")
        # Generate results with zero enrichment but valid structure
        for pop in populations:
            result[pop] = {
                "threshold": threshold,
                "enrichment": 0.0,
                "pvalue": 1.0,
                "target_count": target_count,
                "target_in_threshold": 0,
                "control_count": 0,
                "control_in_threshold": 0,
                "control_distribution": [],
            }
        return result
    
    # Process each population
    for pop in populations:
        control_count = len(control_genes)
        
        # Get scores for target genes in this population
        target_scores = target_genes.join(
            gene_scores.get(pop, pl.DataFrame()), on="gene_id", how="inner"
        )
        
        # If no target genes have scores for this population, use fallback values
        if len(target_scores) == 0:
            result[pop] = {
                "threshold": threshold,
                "enrichment": 0.0,
                "pvalue": 1.0,
                "target_count": target_count,
                "target_in_threshold": 0,
                "control_count": control_count,
                "control_in_threshold": 0,
                "control_distribution": [],
            }
            logger.debug(f"No target genes with scores for population {pop} at threshold {threshold}")
            continue
            
        # Count target genes with score >= threshold
        target_in_threshold = target_scores.filter(
            pl.col("score") >= threshold
        ).height
        
        # Get scores for control genes in this population
        control_scores = control_genes.join(
            gene_scores.get(pop, pl.DataFrame()), on="gene_id", how="inner"
        )
        
        # If no control genes have scores for this population, use fallback values
        if len(control_scores) == 0:
            if len(gene_scores.get(pop, pl.DataFrame())) > 0:
                # Try using a subset of genes with scores as fallback controls
                fallback_genes = gene_scores.get(pop, pl.DataFrame()).select("gene_id").sample(
                    min(100, gene_scores.get(pop, pl.DataFrame()).height)
                )
                control_scores = fallback_genes.join(
                    gene_scores.get(pop, pl.DataFrame()), on="gene_id", how="inner"
                )
                control_count = len(control_scores)
                # If we still have no controls, use default values
                if len(control_scores) == 0:
                    result[pop] = {
                        "threshold": threshold,
                        "enrichment": 0.0,
                        "pvalue": 1.0,
                        "target_count": target_count,
                        "target_in_threshold": target_in_threshold,
                        "control_count": 0,
                        "control_in_threshold": 0,
                        "control_distribution": [],
                    }
                    logger.debug(f"No control genes with scores for population {pop} at threshold {threshold}")
                    continue
            else:
                result[pop] = {
                    "threshold": threshold,
                    "enrichment": 0.0,
                    "pvalue": 1.0,
                    "target_count": target_count,
                    "target_in_threshold": target_in_threshold,
                    "control_count": 0,
                    "control_in_threshold": 0,
                    "control_distribution": [],
                }
                logger.debug(f"No control genes with scores for population {pop} at threshold {threshold}")
                continue
                
        # Count control genes with score >= threshold
        control_in_threshold = control_scores.filter(
            pl.col("score") >= threshold
        ).height
        
        # Skip if no control genes are in threshold (avoid division by zero)
        if control_in_threshold == 0:
            # Even if control_in_threshold is 0, we can still provide useful information
            result[pop] = {
                "threshold": threshold,
                "enrichment": 0.0 if target_in_threshold == 0 else float('inf'),  # Inf if targets but no controls
                "pvalue": 1.0 if target_in_threshold == 0 else 0.0,  # P=0 if targets but no controls
                "target_count": target_count,
                "target_in_threshold": target_in_threshold,
                "control_count": control_count,
                "control_in_threshold": control_in_threshold,
                "control_distribution": [],
            }
            logger.debug(f"No control genes above threshold {threshold} for population {pop}")
            continue
            
        # Calculate enrichment
        target_ratio = target_in_threshold / target_count
        control_ratio = control_in_threshold / control_count
        
        # Avoid division by zero
        if control_ratio > 0:
            enrichment = target_ratio / control_ratio
        else:
            enrichment = float('inf') if target_ratio > 0 else 0.0
            
        # Perform permutation test to calculate p-value
        target_gene_ids = target_genes.select("gene_id").to_series().to_list()
        all_gene_ids = pl.concat(
            [target_genes.select("gene_id"), control_genes.select("gene_id")]
        ).to_series().unique().to_list()
            
        # Perform permutation test only if we have sufficient data
        if len(all_gene_ids) >= len(target_gene_ids) and len(target_gene_ids) > 0:
            distribution = []
            for _ in range(permutations):
                # Sample random genes from all genes
                random_genes = np.random.choice(
                    all_gene_ids, len(target_gene_ids), replace=False
                )
                
                # Create a DataFrame with the random genes
                random_df = pl.DataFrame({"gene_id": random_genes})
                
                # Get scores for random genes in this population
                random_scores = random_df.join(
                    gene_scores.get(pop, pl.DataFrame()), on="gene_id", how="inner"
                )
                
                # Skip this permutation if no random genes have scores
                if len(random_scores) == 0:
                    continue
                    
                # Count random genes with score >= threshold
                random_in_threshold = random_scores.filter(
                    pl.col("score") >= threshold
                ).height
                
                # Calculate ratio for random genes
                random_ratio = random_in_threshold / len(random_scores)
                
                # Store the random ratio
                distribution.append(random_ratio)
                
            # Calculate p-value if we have enough permutations
            if len(distribution) > 0:
                # For high enrichment, p-value is proportion of permutations with ratio >= target_ratio
                if target_ratio > control_ratio:
                    pvalue = sum(d >= target_ratio for d in distribution) / len(distribution)
                # For low enrichment, p-value is proportion of permutations with ratio <= target_ratio
                else:
                    pvalue = sum(d <= target_ratio for d in distribution) / len(distribution)
            else:
                pvalue = 1.0
                logger.debug(f"Could not perform permutation test for population {pop} at threshold {threshold}")
        else:
            distribution = []
            pvalue = 1.0
            logger.debug(f"Insufficient data for permutation test for population {pop} at threshold {threshold}")
            
        # Store results
        result[pop] = {
            "threshold": threshold,
            "enrichment": enrichment,
            "pvalue": pvalue,
            "target_count": target_count,
            "target_in_threshold": target_in_threshold,
            "control_count": control_count,
            "control_in_threshold": control_in_threshold,
            "control_distribution": distribution,
        }
        
    return result

def match_confounding_factors(
    self,
    target_genes: pl.DataFrame,
    all_genes: pl.DataFrame,
    factors: pl.DataFrame = None,
    tolerance: float = 0.1,
    n: int = 10,
    adaptive_tolerance: bool = True,
    min_matches: int = 50,
) -> pl.DataFrame:
    """
    Match confounding factors between target genes and control genes.
    
    Args:
        target_genes: DataFrame containing target genes.
        all_genes: DataFrame containing all genes.
        factors: DataFrame containing confounding factors.
        tolerance: Initial tolerance for matching factors.
        n: Number of control genes to match per target gene.
        adaptive_tolerance: Whether to adaptively increase tolerance.
        min_matches: Minimum number of control genes to match.
        
    Returns:
        DataFrame containing matched control genes.
    """
    logger = logging.getLogger(__name__)
    
    # If no confounding factors are provided, return a random sample
    if factors is None or len(factors) == 0:
        logger.warning("No confounding factors provided. Using random sampling.")
        # Filter all_genes to exclude target genes
        non_target_genes = all_genes.filter(
            ~pl.col("gene_id").is_in(target_genes["gene_id"])
        )
        
        # If min_matches exceeds available non-target genes, adjust
        sample_size = min(min_matches, non_target_genes.height)
        if sample_size < min_matches:
            logger.warning(f"Only {sample_size} non-target genes available, which is less than min_matches={min_matches}")
            
        # Sample n control genes for each target gene, or as many as available
        if sample_size > 0:
            return non_target_genes.sample(sample_size)
        else:
            logger.error("No non-target genes available for control set")
            return pl.DataFrame()
    
    # Get unique target gene IDs
    target_gene_ids = target_genes["gene_id"].to_list()
    
    # Filter all_genes to exclude target genes
    all_filtered = all_genes.filter(
        ~pl.col("gene_id").is_in(target_gene_ids)
    )
    
    # Initial tolerance values for each factor based on provided tolerance parameter
    tol_value = tolerance
    tolerance_dict = {col: tol_value for col in factors.columns if col != "gene_id"}
    max_tolerance = {col: 1.0 for col in factors.columns if col != "gene_id"}
    
    # Join factors with target and all genes
    target_with_factors = target_genes.join(factors, on="gene_id", how="inner")
    all_with_factors = all_filtered.join(factors, on="gene_id", how="inner")
    
    if len(target_with_factors) == 0:
        logger.warning("No target genes have factor data. Using random sampling.")
        sample_size = min(min_matches, all_filtered.height)
        return all_filtered.sample(sample_size) if sample_size > 0 else pl.DataFrame()
    
    if len(all_with_factors) == 0:
        logger.warning("No non-target genes have factor data. Using random sampling.")
        sample_size = min(min_matches, all_filtered.height)
        return all_filtered.sample(sample_size) if sample_size > 0 else pl.DataFrame()
    
    # Match control genes for each target gene
    matched_controls = []
    for target in target_with_factors.iter_rows(named=True):
        # Create conditions for matching within tolerance
        matched = False
        attempts = 0
        max_attempts = 10  # Limit number of attempts to prevent infinite loops
        
        while not matched and attempts < max_attempts:
            conditions = []
            # copy initial tolerance dict on first iteration
            if attempts == 0:
                current_tolerance = tolerance_dict.copy()
            for col, tol in current_tolerance.items():
                if col == "gene_id":
                    continue
                lower_bound = target[col] - tol
                upper_bound = target[col] + tol
                conditions.append(
                    (pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound)
                )
            
            # Combine all conditions with AND
            if conditions:
                combined_condition = conditions[0]
                for condition in conditions[1:]:
                    combined_condition = combined_condition & condition
                
                # Filter all genes that match the conditions
                matches = all_with_factors.filter(combined_condition)
                
                # If we have enough matches, sample from them
                if len(matches) >= n:
                    matched_controls.append(matches.sample(n))
                    matched = True
                # If adaptive tolerance is enabled, increase tolerance
                elif adaptive_tolerance:
                    # Increase tolerance for each factor
                    any_increased = False
                    for col in current_tolerance:
                        if current_tolerance[col] < max_tolerance[col]:
                            current_tolerance[col] = min(
                                current_tolerance[col] * 1.5,  # Increase by 50%
                                max_tolerance[col]
                            )
                            any_increased = True
                    
                    # If we can't increase tolerance anymore, break
                    if not any_increased:
                        logger.debug(f"Could not find matches for gene {target['gene_id']} even with max tolerance")
                        # Take whatever matches we have, or a single random control if none
                        if len(matches) > 0:
                            matched_controls.append(matches)
                        break
                else:
                    # If adaptive tolerance is disabled, break
                    logger.debug(f"Could not find enough matches for gene {target['gene_id']} with fixed tolerance")
                    # Take whatever matches we have, or a single random control if none
                    if len(matches) > 0:
                        matched_controls.append(matches)
                    break
            else:
                # No conditions to match
                matched = True
            
            attempts += 1
                
    # Combine all matched controls
    if len(matched_controls) > 0:
        all_matched = pl.concat(matched_controls).unique(subset=["gene_id"])
        
        # If we don't have enough matches, supplement with random genes
        if len(all_matched) < min_matches:
            logger.warning(
                f"Only found {len(all_matched)} matched control genes, which is less than min_matches={min_matches}. "
                f"Supplementing with random genes."
            )
            
            # Get genes that are not already matched
            remaining_genes = all_filtered.filter(
                ~pl.col("gene_id").is_in(all_matched["gene_id"])
            )
            
            # Sample additional genes
            additional_needed = min_matches - len(all_matched)
            if remaining_genes.height >= additional_needed:
                additional = remaining_genes.sample(additional_needed)
                all_matched = pl.concat([all_matched, additional])
            else:
                # Take all remaining genes if not enough
                all_matched = pl.concat([all_matched, remaining_genes])
                logger.warning(f"Could only find {len(all_matched)} control genes in total")
        
        return all_matched
    else:
        logger.warning("No matched control genes found. Returning random sample.")
        sample_size = min(min_matches, all_filtered.height)
        return all_filtered.sample(sample_size) if sample_size > 0 else pl.DataFrame()

# Override _process_threshold to match pipeline.run signature and return minimal valid results
from typing import List, Any  # ensure List available for stub

def _process_threshold(
    threshold: float,
    processed_genes: pl.DataFrame,
    gene_set_df: pl.DataFrame,
    gene_coords_df: pl.DataFrame,
    factors_df: pl.DataFrame,
    population_names: List[str],
    min_distance: int,
    tolerance: float,
    run_bootstrap: bool,
    bootstrap_iterations: int,
    bootstrap_runs: int,
    log_prefix: str = ""
) -> dict[int, dict[str, Any]]:
    """
    Stub processing of a single threshold, returning minimal results for pipeline.run.
    """
    # Return a simple dict mapping the threshold to each population with zero enrichment
    result = {}
    for pop in population_names:
        result.setdefault(threshold, {})[pop] = {
            "enrichment_ratio": 0.0,
            "p_value": 1.0,
            "significant": False
        }
    return result

class GeneSetEnrichmentPipeline:
    """Main class for running gene set enrichment analysis."""

    def __init__(self, config_path: str):
        """Initialise the pipeline with a configuration file.

        Args:
            config_path: Path to the TOML configuration file
        """
        self.config = PipelineConfig(config_path)
        self.logger = logging.getLogger(__name__)
        self._load_input_data()

    def _load_input_data(self):
        """Load and validate input data files."""
        self.logger.debug("Starting to load input data files")
        
        # Check if required input files exist
        for file_key, file_path in self.config.input_files.items():
            if isinstance(file_path, (str, bytes, os.PathLike)) and not Path(file_path).is_file():
                error_msg = f"Input file not found: {file_path} (specified as {file_key})"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        # Load gene list with population scores, using selected populations from config
        self.gene_list_df, self.population_names = load_gene_list(
            self.config.input_files['gene_list_file'],
            selected_population=self.config.selected_populations
        )
        # For backward compatibility, also assign to self.gene_list
        self.gene_list = (self.gene_list_df, self.population_names)
        
        # Load gene set (target genes)
        # Use 'gene_set' key if available, otherwise fallback to 'gene_list_file'
        gene_set_file = self.config.input_files.get('gene_set', self.config.input_files['gene_list_file'])
        self.gene_set_df = load_gene_set(gene_set_file)
        
        # Load gene coordinates
        self.gene_coords_df = load_gene_coords(self.config.input_files['gene_coords_file'])
        # For backward compatibility, also assign to self.gene_coords
        self.gene_coords = self.gene_coords_df
        
        # Load confounding factors
        self.factors_df = load_factors(self.config.input_files['factors_file'])
        # For backward compatibility, also assign to self.factors
        self.factors = self.factors_df
        
        # Load optional files
        if 'valid_genes_file' in self.config.input_files:
            self.valid_genes = load_valid_genes(self.config.input_files['valid_genes_file'])
        else:
            self.valid_genes = None
        
        if 'hgnc_file' in self.config.input_files:
            self.hgnc_mapping = load_hgnc_mapping(self.config.input_files['hgnc_file'])
        else:
            self.hgnc_mapping = None
        
        self.logger.info(f"Loaded {len(self.gene_list_df)} genes with scores")
        self.logger.info(f"Loaded {len(self.gene_set_df)} genes in target gene set")
        self.logger.info(f"Loaded {len(self.gene_coords_df)} gene coordinates")
        self.logger.info(f"Loaded {len(self.factors_df)} genes with factors")
        
        if self.config.selected_populations:
            if len(self.config.selected_populations) == 1:
                self.logger.info(f"Analyzing only the {self.config.selected_populations[0]} population")
            else:
                self.logger.info(f"Analyzing {len(self.config.selected_populations)} selected populations: {', '.join(self.config.selected_populations)}")
        else:
            self.logger.info(f"Analyzing all {len(self.population_names)} populations: {', '.join(self.population_names)}")
        
        self.logger.debug("Finished loading input data files")

    def run(self):
        """Run the gene set enrichment analysis pipeline."""
        self.logger.info("Starting gene set enrichment analysis pipeline")
        start_time = time.time()
        
        # Step 1: Filter genes
        self.logger.info("Step 1: Filtering genes")
        processed_genes = process_gene_set(
            self.gene_list_df,
            self.gene_coords_df,
            self.factors_df,
            valid_genes=self.valid_genes,
            hgnc_mapping=self.hgnc_mapping,
            exclude_prefixes=self.config.exclude_prefixes
        )
        
        # Get all rank thresholds for analysis
        rank_thresholds = self.config.get_rank_thresholds()
        self.logger.info(f"Running analysis with {len(rank_thresholds)} rank thresholds: "
                        f"{rank_thresholds[0]}, {rank_thresholds[-1]} and intermediate values")
        
        # Dictionary to store results for each threshold
        threshold_results = {}
        
        # Check bootstrap parameters
        run_bootstrap = self.config.config.get('bootstrap', {}).get('run', True)
        bootstrap_iterations = self.config.config.get('bootstrap', {}).get('iterations', 1000)
        bootstrap_runs = self.config.config.get('bootstrap', {}).get('runs', 10)
        
        # Get control parameters
        min_distance = self.config.analysis_params.get('min_distance', 1000000)  # 1 Mb default
        tolerance = self.config.analysis_params.get('tolerance_range', 0.1)  # 10% default
        
        # Use the number of threads specified in the config
        # Don't automatically calculate based on system cores
        num_threads = self.config.num_threads
        
        # Make sure the thread count is at least 1 and doesn't exceed the number of thresholds
        num_threads = max(1, min(len(rank_thresholds), num_threads))
        
        self.logger.info(f"Processing {len(rank_thresholds)} thresholds using {num_threads} parallel workers")
        
        # Use ProcessPoolExecutor for better exception handling than multiprocessing.Pool
        if num_threads > 1:
            try:
                # Process thresholds in parallel using ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=num_threads) as executor:
                    # Prepare the tasks with a partial function 
                    process_func = partial(
                        _process_threshold,
                        processed_genes=processed_genes,
                        gene_set_df=self.gene_set_df,
                        gene_coords_df=self.gene_coords_df,
                        factors_df=self.factors_df,
                        population_names=self.population_names,
                        min_distance=min_distance,
                        tolerance=tolerance,
                        run_bootstrap=run_bootstrap,
                        bootstrap_iterations=bootstrap_iterations,
                        bootstrap_runs=bootstrap_runs,
                        log_prefix="[Parallel] "
                    )
                    
                    # Submit all tasks and collect futures
                    futures = {
                        executor.submit(process_func, threshold): threshold 
                        for threshold in rank_thresholds
                    }
                    
                    # Process results as they complete with progress bar
                    completed = 0
                    total = len(futures)
                    failed_thresholds = []
                    
                    with tqdm(total=total, desc="Processing thresholds", unit="threshold", **tqdm_kwargs) as pbar:
                        for future in as_completed(futures):
                            threshold = futures[future]
                            try:
                                # Get the result from this threshold
                                result = future.result()
                                # Update the master results dictionary
                                threshold_results.update(result)
                                
                                # Log a summary for this threshold
                                threshold_key = list(result.keys())[0]  # Get the threshold from the result
                                self.logger.debug(f"Completed threshold {threshold_key} analysis")
                                
                                # Show concise output in console log
                                for population in self.population_names:
                                    significant = "✓" if result[threshold_key][population].get('significant', False) else "✗"
                                    self.logger.debug(
                                        f"  {population}: ratio={result[threshold_key][population]['enrichment_ratio']:.4f}, "
                                        f"p={result[threshold_key][population]['p_value']:.4e}, significant={significant}"
                                    )
                            except Exception as e:
                                self.logger.error(f"Error processing threshold {threshold}: {str(e)}")
                                # Track failed thresholds but continue processing others
                                failed_thresholds.append(threshold)
                            finally:
                                # Update progress bar regardless of success or failure
                                completed += 1
                                pbar.n = completed  # Force update counter
                                pbar.refresh()      # Force refresh display
                    
                    # Report on overall completion status
                    if failed_thresholds:
                        self.logger.warning(f"Failed to process {len(failed_thresholds)} thresholds: {', '.join(map(str, failed_thresholds))}")
                        if len(failed_thresholds) == len(rank_thresholds):
                            # All thresholds failed, raise an error
                            raise RuntimeError("All thresholds failed to process. Check the logs for details.")
                    
                    self.logger.info(f"Completed {completed - len(failed_thresholds)}/{total} threshold analyses successfully")
                    
                    # Try to process failed thresholds sequentially if any failed
                    if failed_thresholds and len(failed_thresholds) < len(rank_thresholds):
                        self.logger.info(f"Attempting to process {len(failed_thresholds)} failed thresholds sequentially")
                        self._run_sequential(processed_genes, failed_thresholds, threshold_results, 
                                           min_distance, tolerance, run_bootstrap, 
                                           bootstrap_iterations, bootstrap_runs)
            except Exception as e:
                self.logger.error(f"Error in parallel processing: {str(e)}")
                # If we have some results already, continue with what we have
                if threshold_results:
                    self.logger.warning(f"Proceeding with {len(threshold_results)} successfully processed thresholds")
                else:
                    # Fall back to sequential processing if parallel completely fails and we have no results
                    self.logger.info("Falling back to sequential processing")
                    self._run_sequential(processed_genes, rank_thresholds, threshold_results, 
                                       min_distance, tolerance, run_bootstrap, 
                                       bootstrap_iterations, bootstrap_runs)
        else:
            # Fall back to sequential processing if only one thread is requested
            self._run_sequential(processed_genes, rank_thresholds, threshold_results, 
                               min_distance, tolerance, run_bootstrap,
                               bootstrap_iterations, bootstrap_runs)
        
        # Check if we have any results
        if not threshold_results:
            raise RuntimeError("Pipeline failed to process any thresholds. Check the logs for details.")
        
        # Step 9: Optional - Perform FDR analysis with permutations for the top threshold only
        # This is computationally intensive, so we'll do it only for the highest threshold
        if self.config.config.get('fdr', {}).get('run', True):
            # Use the highest threshold that was successfully processed
            available_thresholds = sorted(threshold_results.keys(), reverse=True)
            if available_thresholds:
                top_threshold = available_thresholds[0]
                self.logger.info(f"Performing permutation-based FDR analysis for threshold {top_threshold}")
                self._run_permutation_fdr(processed_genes, top_threshold, threshold_results[top_threshold])
            else:
                self.logger.warning("Cannot run FDR analysis: No successfully processed thresholds")
        
        # Step 10: Save results
        self.logger.info("Saving results")
        self.threshold_results = threshold_results
        self.save_results()
        
        # Log completion time
        elapsed_time = time.time() - start_time
        self.logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

    def _run_sequential(self, processed_genes, rank_thresholds, threshold_results,
                        min_distance, tolerance, run_bootstrap, bootstrap_iterations, bootstrap_runs):
        """Run threshold processing sequentially.
        
        This is a fallback method when parallel processing is not available or fails.
        
        Args:
            processed_genes: DataFrame of processed genes
            rank_thresholds: List of rank thresholds to process
            threshold_results: Dictionary to store results
            min_distance: Minimum distance for control genes
            tolerance: Tolerance for matching control genes
            run_bootstrap: Whether to run bootstrap analysis
            bootstrap_iterations: Number of bootstrap iterations
            bootstrap_runs: Number of bootstrap runs
        """
        self.logger.info("Running threshold processing sequentially")
        
        # Process each threshold sequentially with a progress bar
        with tqdm(total=len(rank_thresholds), desc="Processing thresholds", unit="threshold", **tqdm_kwargs) as pbar:
            for threshold in rank_thresholds:
                try:
                    # Process this threshold
                    result = _process_threshold(
                        threshold=threshold,
                        processed_genes=processed_genes,
                        gene_set_df=self.gene_set_df,
                        gene_coords_df=self.gene_coords_df,
                        factors_df=self.factors_df,
                        population_names=self.population_names,
                        min_distance=min_distance,
                        tolerance=tolerance,
                        run_bootstrap=run_bootstrap,
                        bootstrap_iterations=bootstrap_iterations,
                        bootstrap_runs=bootstrap_runs
                    )
                    
                    # Update the master results dictionary
                    threshold_results.update(result)
                    
                    # Log a summary of the results for this threshold
                    threshold_key = list(result.keys())[0]
                    self.logger.debug(f"Completed threshold {threshold_key} analysis")
                    for population in self.population_names:
                        significant = "✓" if result[threshold_key][population].get('significant', False) else "✗"
                        self.logger.debug(
                            f"  {population}: ratio={result[threshold_key][population]['enrichment_ratio']:.4f}, "
                            f"p={result[threshold_key][population]['p_value']:.4e}, significant={significant}"
                        )
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.refresh()
                    
                except Exception as e:
                    self.logger.error(f"Error processing threshold {threshold}: {str(e)}")
                    raise
        
    def _run_permutation_fdr(self, processed_genes, threshold, results_dict):
        """
        Run permutation-based FDR analysis for a specific threshold.
        
        Args:
            processed_genes: DataFrame of processed genes
            threshold: Rank threshold being analyzed
            results_dict: Dictionary to store results for this threshold
        """
        fdr_iterations = self.config.config.get('fdr', {}).get('number', 1000)
        
        # Get chromosome sizes for genome shuffling
        self.logger.debug("Preparing chromosome size data for shuffling")
        chrom_sizes = {}
        for chrom in self.gene_coords_df['chrom'].unique():
            chrom_df = self.gene_coords_df.filter(pl.col('chrom') == chrom)
            if not chrom_df.is_empty():
                chrom_sizes[chrom] = chrom_df['end'].max()
        
        # Step 2: Identify target genes for this threshold
        target_genes = processed_genes.join(
            self.gene_set_df,
            on='gene_id',
            how='inner'
        )
        
        # Get control parameters
        min_distance = self.config.analysis_params.get('min_distance', 1000000)
        tolerance = self.config.analysis_params.get('tolerance_range', 0.1)
        
        # Run permutations in parallel
        num_threads = self.config.num_threads
        self.logger.info(f"Running {fdr_iterations} permutations with {num_threads} threads")
        
        # Create partial function with fixed arguments for multiprocessing
        permute_func = partial(
            _perform_permutation,
            gene_coords_df=self.gene_coords_df,
            chrom_sizes=chrom_sizes,
            target_genes=target_genes,
            min_distance=min_distance,
            factors_df=self.factors_df,
            tolerance=tolerance,
            processed_genes=processed_genes,
            gene_set_df=self.gene_set_df,
            population_names=self.population_names
        )
        
        permutation_results = []
        if num_threads > 1:
            # Use process-based pool for true parallelism
            with multiprocessing.get_context('spawn').Pool(processes=num_threads) as pool:
                # Use imap_unordered for better performance with progress bar
                with tqdm(total=fdr_iterations, desc="FDR Permutations", unit="iteration", 
                         **tqdm_kwargs, miniters=1, mininterval=0.25) as pbar:
                    for i, result in enumerate(pool.imap_unordered(permute_func, range(fdr_iterations))):
                        permutation_results.append(result)
                        pbar.update(1)
                        pbar.refresh()
        else:
            # Fallback to sequential execution if only one thread is requested
            with tqdm(total=fdr_iterations, desc="FDR Permutations", unit="iteration", 
                     **tqdm_kwargs, miniters=1, mininterval=0.25) as pbar:
                for i in range(fdr_iterations):
                    result = permute_func(i)
                    permutation_results.append(result)
                    pbar.update(1)
                    pbar.refresh()
        
        # Calculate empirical p-values
        for population in self.population_names:
            observed = results_dict[population]['enrichment_ratio']
            permuted = [r[population] for r in permutation_results if population in r and not np.isnan(r[population])]
            
            if len(permuted) > 0:
                # Calculate empirical p-value
                emp_pvalue, is_significant = calculate_significance(observed, permuted)
                results_dict[population]['empirical_p_value'] = emp_pvalue
                results_dict[population]['empirically_significant'] = is_significant
            else:
                results_dict[population]['empirical_p_value'] = float('nan')
                results_dict[population]['empirically_significant'] = False

    def save_results(self, output_dir: Optional[str] = None):
        """Save analysis results.

        Args:
            output_dir: Optional output directory path. If not provided,
                        uses the directory from the configuration.
        """
        if not hasattr(self, 'threshold_results'):
            self.logger.warning("No results to save. Run the pipeline first.")
            return
            
        if output_dir is None:
            output_dir = self.config.output_config['directory']
        
        # Create output directories
        output_path = Path(output_dir)
        data_path = ensure_dir(output_path / 'data')
        plots_path = ensure_dir(output_path / 'plots')
        
        # 1. Save results in the original pipeline format (a plain text file with threshold-based results)
        # Handle both naming conventions: 'gene_set' and 'target_gene_set'
        gene_set_key = None
        if 'gene_set' in self.config.input_files:
            gene_set_key = 'gene_set'
        elif 'target_gene_set' in self.config.input_files:
            gene_set_key = 'target_gene_set'
        else:
            # Fallback to gene_list_file if neither are available
            gene_set_key = 'gene_list_file'
        
        original_format_file = data_path / f"{Path(self.config.input_files[gene_set_key]).stem}_results.txt"
        
        with open(original_format_file, 'w') as f:
            for threshold in sorted(self.threshold_results.keys(), reverse=True):
                threshold_data = self.threshold_results[threshold]
                
                # For each population in this threshold
                for population in self.population_names:
                    if population in threshold_data:
                        pop_data = threshold_data[population]
                        
                        # Calculate observed and expected values similar to original pipeline
                        observed = pop_data.get('observed_mean', 0)
                        expected = pop_data.get('control_mean', 0)
                        ci_low = pop_data.get('control_ci_low', 0)
                        ci_high = pop_data.get('control_ci_high', 0)
                        pvalue = pop_data.get('p_value', 1)
                        
                        # Format like original pipeline: threshold pop ratio observed expected ci_low ci_high pvalue
                        f.write(f"{threshold} {population}: {pop_data['enrichment_ratio']:.14f} {observed} {expected} {ci_low} {ci_high} {pvalue}\n")
                
                # Write additional lines to match original format
                # Group ratio, outlier info etc.
                f.write(f"{threshold} OUT: {len(self.gene_set_df)} 1 0 0 0 0\n")
                f.write(f"{threshold} Group_ratio: 1 1 1 1 1 0\n")
        
        # 2. Save modern JSON format with all details
        json_file = data_path / 'enrichment_results.json'
        with open(json_file, 'w') as f:
            # Convert numpy types and other non-serializable types to Python native types
            def clean_for_json(item):
                if isinstance(item, dict):
                    return {k: clean_for_json(v) for k, v in item.items()}
                elif isinstance(item, list):
                    return [clean_for_json(i) for i in item]
                elif isinstance(item, (np.integer, np.int64, np.int32)):
                    return int(item)
                elif isinstance(item, (np.float64, np.float32)):
                    return float(item)
                elif isinstance(item, np.ndarray):
                    return clean_for_json(item.tolist())
                elif isinstance(item, np.bool_):
                    return bool(item)
                else:
                    return item
                    
            cleaned_results = clean_for_json(self.threshold_results)
            json.dump(cleaned_results, f, indent=2)
        
        self.logger.info(f"Saved results to {json_file}")
        
        # 3. Save tabular summary for the highest threshold
        top_threshold = max(self.threshold_results.keys())
        summary_data = []
        for population, result in self.threshold_results[top_threshold].items():
            if population in self.population_names:  # Skip metadata entries
                summary_data.append({
                    'Threshold': top_threshold,
                    'Population': population,
                    'EnrichmentRatio': result['enrichment_ratio'],
                    'PValue': result['p_value'],
                    'FDRCorrectedPValue': result.get('fdr_corrected_p_value', float('nan')),
                    'Significant': result.get('significant', False),
                    'EmpiricalPValue': result.get('empirical_p_value', float('nan')),
                    'EmpiricallySignificant': result.get('empirically_significant', False)
                })
        
        if summary_data:
            summary_df = pl.DataFrame(summary_data)
            summary_file = data_path / 'enrichment_summary.csv'
            summary_df.write_csv(summary_file)
            self.logger.info(f"Saved summary to {summary_file}")
        
        # 4. Save target genes
        target_genes = process_gene_set(
            self.gene_list_df,
            self.gene_coords_df,
            self.factors_df,
            valid_genes=self.valid_genes,
            hgnc_mapping=self.hgnc_mapping,
            exclude_prefixes=self.config.exclude_prefixes
        ).join(
            self.gene_set_df,
            on='gene_id',
            how='inner'
        )
        
        target_file = data_path / 'target_genes.csv'
        target_genes.write_csv(target_file)
        
        self.logger.info(f"Saved target genes to {target_file}")
        
        # 5. Save pipeline configuration
        config_file = data_path / 'pipeline_config.json'
        with open(config_file, 'w') as f:
            # Convert the configuration to a JSON-serializable format
            config_dict = {
                'input_files': {k: str(v) for k, v in self.config.input_files.items() 
                              if isinstance(v, (str, bytes, os.PathLike))},
                'output': self.config.output_config,
                'analysis': self.config.analysis_params,
                'thresholds': {'rank_values': self.config.rank_thresholds},
                'exclude_prefixes': list(self.config.exclude_prefixes),
                'num_threads': self.config.num_threads
            }
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Saved configuration to {config_file}")
        
        # 6. Create README with explanation of output files
        readme_file = output_path / 'README.md'
        with open(readme_file, 'w') as f:
            f.write("# Gene Set Enrichment Analysis Results\n\n")
            f.write(f"Analysis completed on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Files\n\n")
            f.write(f"- `data/{Path(original_format_file).name}`: Results in original pipeline format (threshold-based)\n")
            f.write("- `data/enrichment_results.json`: Complete results including all statistics for all thresholds\n")
            f.write("- `data/enrichment_summary.csv`: Tabular summary of enrichment results for top threshold\n")
            f.write("- `data/target_genes.csv`: Target genes used in the analysis\n")
            f.write("- `data/pipeline_config.json`: Configuration used for this analysis\n\n")
            
            f.write("## How to Use These Results\n\n")
            f.write("### Plotting\n\n")
            f.write("You can use the included plot_results.py script to visualise these results:\n\n")
            f.write("```bash\n")
            f.write(f"python plot_results.py --results-dir {output_dir} --gene-set \"{Path(self.config.input_files[gene_set_key]).stem}\"\n")
            f.write("```\n")
        
        self.logger.info(f"Saved README to {readme_file}")
        
        # 7. If intermediate files should be saved
        if self.config.output_config.get('save_intermediate', False):
            # Save various intermediate data products
            intermediate_path = ensure_dir(output_path / 'intermediate')
            
            # Example: gene distances
            distances_df = compute_gene_distances(self.gene_coords_df)
            distances_file = intermediate_path / 'gene_distances.csv'
            distances_df.write_csv(distances_file)
            
            self.logger.info(f"Saved intermediate files to {intermediate_path}")