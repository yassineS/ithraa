"""Main pipeline implementation for gene set enrichment analysis."""

import logging
import multiprocessing
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Set, Tuple, Any
import json
from functools import partial

import polars as pl
import numpy as np
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
    shuffle_genome
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
    # Shuffle gene coordinates
    shuffled_coords = shuffle_genome(gene_coords_df, chrom_sizes)
    
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
        
        # Calculate enrichment
        if len(shuffled_control_scores) > 0:
            enrichment_ratio = np.mean(target_scores) / np.mean(shuffled_control_scores) if np.mean(shuffled_control_scores) != 0 else float('nan')
            iter_results[population] = enrichment_ratio
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
    logger: logging.Logger,
    min_distance: int = 1000000,
    tolerance: float = 0.1,
    run_bootstrap: bool = True,
    bootstrap_iterations: int = 1000,
    bootstrap_runs: int = 10
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
        logger: Logger instance
        min_distance: Minimum distance for control genes
        tolerance: Tolerance for matching control genes
        run_bootstrap: Whether to run bootstrap analysis
        bootstrap_iterations: Number of bootstrap iterations
        bootstrap_runs: Number of bootstrap runs
        
    Returns:
        Dictionary with results for this threshold
    """
    logger.debug(f"Processing threshold {threshold}")
    threshold_results = {}
    
    # Step 2: Identify target genes for this threshold
    target_genes = processed_genes.join(
        gene_set_df,
        on='gene_id',
        how='inner'
    )
    
    if len(target_genes) > threshold:
        logger.debug(f"Target gene set ({len(target_genes)}) exceeds threshold {threshold}. Using all available target genes.")
    
    logger.debug(f"Identified {len(target_genes)} target genes after filtering for threshold {threshold}")
    
    # Step 3: Compute gene distances
    distances_df = compute_gene_distances(gene_coords_df)
    
    # Step 4: Find control genes (genes far enough from target genes)
    control_genes = find_control_genes(
        target_genes,
        distances_df,
        min_distance=min_distance
    )
    logger.debug(f"Found {len(control_genes)} potential control genes for threshold {threshold}")
    
    # Step 5: Match control genes based on confounding factors
    matched_controls = match_confounding_factors(
        target_genes,
        control_genes,
        factors_df,
        tolerance=tolerance
    )
    logger.debug(f"Matched {len(matched_controls)} control genes for threshold {threshold}")
    
    # Step 6: Perform enrichment analysis for each population
    for population in population_names:
        logger.debug(f"Analyzing population: {population} for threshold {threshold}")
        
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
        logger.debug(
            f"Population {population} at threshold {threshold}: "
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
            for run_idx in range(bootstrap_runs):
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
            
    return threshold_results

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
        
        # Load gene list with population scores
        self.gene_list_df, self.population_names = load_gene_list(self.config.input_files['gene_list_file'])
        
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
        
        # Process each threshold with a progress bar (only showing in terminal)
        for threshold in tqdm(rank_thresholds, desc="Processing thresholds", unit="threshold"):
            # Process this threshold and store results
            threshold_results[threshold] = _process_threshold(
                threshold=threshold,
                processed_genes=processed_genes,
                gene_set_df=self.gene_set_df,
                gene_coords_df=self.gene_coords_df,
                factors_df=self.factors_df,
                population_names=self.population_names,
                logger=self.logger,
                min_distance=min_distance,
                tolerance=tolerance,
                run_bootstrap=run_bootstrap,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_runs=bootstrap_runs
            )
            
            # Log a summary of the results for this threshold at INFO level
            # This will help track progress in the log file while keeping console clean
            self.logger.info(f"Completed threshold {threshold} analysis")
            for population in self.population_names:
                significant = "✓" if threshold_results[threshold][population].get('significant', False) else "✗"
                self.logger.info(
                    f"  {population}: ratio={threshold_results[threshold][population]['enrichment_ratio']:.4f}, "
                    f"p={threshold_results[threshold][population]['p_value']:.4e}, significant={significant}"
                )
        
        # Step 9: Optional - Perform FDR analysis with permutations for the top threshold only
        # This is computationally intensive, so we'll do it only for the highest threshold
        if self.config.config.get('fdr', {}).get('run', True):
            top_threshold = rank_thresholds[0]
            self.logger.info(f"Performing permutation-based FDR analysis for threshold {top_threshold}")
            
            self._run_permutation_fdr(processed_genes, top_threshold, threshold_results[top_threshold])
        
        # Step 10: Save results
        self.logger.info("Saving results")
        self.threshold_results = threshold_results
        self.save_results()
        
        # Log completion time
        elapsed_time = time.time() - start_time
        self.logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
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
                with tqdm(total=fdr_iterations, desc="FDR Permutations") as pbar:
                    for result in pool.imap_unordered(permute_func, range(fdr_iterations)):
                        permutation_results.append(result)
                        pbar.update(1)
        else:
            # Fallback to sequential execution if only one thread is requested
            for i in tqdm(range(fdr_iterations), desc="FDR Permutations"):
                permutation_results.append(permute_func(i))
        
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