"""Main pipeline implementation for gene set enrichment analysis."""

import logging
import multiprocessing
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import json
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
import numba as nb
import numpy as np  # Use standard numpy instead of Numba's numpy API
from tqdm import tqdm

from .config import PipelineConfig
from .data import (
    load_gene_list, 
    load_gene_set, 
    load_gene_coords, 
    load_factors, 
    load_valid_genes,
    load_hgnc_mapping,
    filter_genes,
    compute_gene_distances,
    process_gene_set,
    find_control_genes,
    match_confounding_factors,
    shuffle_genome,
    shuffle_genome_circular
)
from .stats import (
    calculate_enrichment,
    perform_fdr_analysis,
    bootstrap_analysis,
    control_for_confounders,
    compute_enrichment_score,
    calculate_significance
)
from .utils import ensure_dir

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

# Helper functions for optimized threshold processing
def _process_threshold(
    threshold: int,
    processed_genes: pl.DataFrame,
    gene_set_df: pl.DataFrame,
    gene_coords_df: pl.DataFrame,
    factors_df: pl.DataFrame,
    population_names: List[str],
    min_distance: int = 1000000,
    tolerance: float = 0.1,
    run_bootstrap: bool = True,
    bootstrap_iterations: int = 1000,
    bootstrap_runs: int = 10,
    log_prefix: str = ""
) -> Dict[str, Dict[str, Any]]:
    """
    Process a single threshold to improve parallelization.
    
    Args:
        threshold: The rank threshold to process
        processed_genes: Processed genes DataFrame
        gene_set_df: Gene set DataFrame
        gene_coords_df: Gene coordinates DataFrame
        factors_df: Factors DataFrame
        population_names: List of population names
        min_distance: Minimum distance for control genes
        tolerance: Tolerance for matching control genes
        run_bootstrap: Whether to run bootstrap analysis
        bootstrap_iterations: Number of bootstrap iterations
        bootstrap_runs: Number of bootstrap runs
        log_prefix: Prefix for log messages (used by parallel processing)
        
    Returns:
        Dictionary with results for this threshold
    """
    # Create a separate logger for this process to avoid synchronization issues
    thread_logger = logging.getLogger(f"{__name__}.threshold_{threshold}")
    
    thread_logger.debug(f"{log_prefix}Processing threshold {threshold}")
    threshold_results = {}
    
    # Step 2: Identify target genes for this threshold
    target_genes = processed_genes.join(
        gene_set_df,
        on='gene_id',
        how='inner'
    )
    
    if len(target_genes) > threshold:
        thread_logger.debug(f"{log_prefix}Target gene set ({len(target_genes)}) exceeds threshold {threshold}. Using all available target genes.")
    
    thread_logger.debug(f"{log_prefix}Identified {len(target_genes)} target genes after filtering for threshold {threshold}")
    
    # Step 3: Compute gene distances
    distances_df = compute_gene_distances(gene_coords_df)
    
    # Step 4: Find control genes (genes far enough from target genes)
    control_genes = find_control_genes(
        target_genes,
        distances_df,
        min_distance=min_distance
    )
    thread_logger.debug(f"{log_prefix}Found {len(control_genes)} potential control genes for threshold {threshold}")
    
    # Step 5: Match control genes based on confounding factors
    matched_controls = match_confounding_factors(
        target_genes,
        control_genes,
        factors_df,
        tolerance=tolerance
    )
    thread_logger.debug(f"{log_prefix}Matched {len(matched_controls)} control genes for threshold {threshold}")
    
    # Step 6: Perform enrichment analysis for each population
    for population in population_names:
        thread_logger.debug(f"{log_prefix}Analyzing population: {population} for threshold {threshold}")
        
        # Get scores for target and control genes
        target_scores = processed_genes.join(
            gene_set_df,
            on='gene_id',
            how='inner'
        )[population].to_numpy()
        
        control_scores = processed_genes.join(
            matched_controls.select(['gene_id']),
            on='gene_id',
            how='inner'
        )[population].to_numpy()
        
        # Calculate enrichment
        enrichment_stats = calculate_enrichment(target_scores, control_scores)
        
        threshold_results[population] = enrichment_stats
        thread_logger.debug(
            f"{log_prefix}Population {population} at threshold {threshold}: "
            f"Enrichment ratio = {enrichment_stats['enrichment_ratio']:.4f}, "
            f"p-value = {enrichment_stats['p_value']:.6f}"
        )
    
    # Step 7: Perform FDR correction for this threshold
    p_values = np.array([threshold_results[pop]['p_value'] for pop in population_names])
    fdr_results = perform_fdr_analysis(p_values)
    
    # Add FDR results to population results
    for i, population in enumerate(population_names):
        threshold_results[population]['fdr_corrected_p_value'] = float(fdr_results['pvals_corrected'][i])
        threshold_results[population]['significant'] = bool(fdr_results['reject'][i])
    
    # Step 8: Perform bootstrap analysis for this threshold (if enabled)
    if run_bootstrap:
        for population in population_names:
            # Get scores for target and control genes
            target_scores = processed_genes.join(
                gene_set_df,
                on='gene_id',
                how='inner'
            )[population].to_numpy()
            
            control_scores = processed_genes.join(
                matched_controls.select(['gene_id']),
                on='gene_id',
                how='inner'
            )[population].to_numpy()
            
            # Run bootstrap analysis
            population_results = []
            for run_idx in range(bootstrap_runs):  # Configurable number of bootstrap runs
                # Resample target scores
                target_bootstrap = bootstrap_analysis(
                    target_scores,
                    n_iterations=bootstrap_iterations
                )
                
                # Resample control scores
                control_bootstrap = bootstrap_analysis(
                    control_scores,
                    n_iterations=bootstrap_iterations
                )
                
                population_results.append({
                    'target': target_bootstrap,
                    'control': control_bootstrap,
                    'enrichment_ratio': target_bootstrap['mean'] / control_bootstrap['mean'] 
                        if control_bootstrap['mean'] != 0 else float('nan'),
                    'target_minus_control': target_bootstrap['mean'] - control_bootstrap['mean'],
                })
            
            # Add bootstrap results to the threshold results
            threshold_results[population]['bootstrap'] = population_results
    
    # Return a tuple of threshold and results for easier tracking in parallel processing        
    return {threshold: threshold_results}

class GeneSetEnrichmentPipeline:
    """Main class for running gene set enrichment analysis."""

    def __init__(self, config_path: str):
        """Initialize the pipeline with a configuration file.

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
        
        # Load gene set (target genes)
        self.gene_set_df = load_gene_set(self.config.input_files['gene_set'])
        
        # Load gene coordinates
        self.gene_coords_df = load_gene_coords(self.config.input_files['gene_coords_file'])
        
        # Load confounding factors
        self.factors_df = load_factors(self.config.input_files['factors_file'])
        
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
                    
                    with tqdm(total=total, desc="Processing thresholds", unit="threshold", 
                            position=0, leave=True, ncols=100) as pbar:
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
        with tqdm(total=len(rank_thresholds), desc="Processing thresholds", unit="threshold",
                position=0, leave=True, ncols=100) as pbar:
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
        segments = self.config.config.get('fdr', {}).get('shuffling_segments', 1)
        
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
                with tqdm(total=fdr_iterations, desc="FDR Permutations", unit="iteration", position=0, leave=True, ncols=100, smoothing=0.1) as pbar:
                    for i, result in enumerate(pool.imap_unordered(permute_func, range(fdr_iterations))):
                        permutation_results.append(result)
                        pbar.update(1)
                        pbar.refresh()
        else:
            # Fallback to sequential execution if only one thread is requested
            with tqdm(total=fdr_iterations, desc="FDR Permutations", unit="iteration", position=0, leave=True, ncols=100, smoothing=0.1) as pbar:
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
        original_format_file = data_path / f"{Path(self.config.input_files['gene_set']).stem}_results.txt"
        
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
            f.write("You can use the included plot_results.py script to visualize these results:\n\n")
            f.write("```bash\n")
            f.write(f"python plot_results.py --results-dir {output_dir} --gene-set \"{Path(self.config.input_files['gene_set']).stem}\"\n")
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