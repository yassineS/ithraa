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
        
        # Step 2: Identify target genes
        self.logger.info("Step 2: Identifying target genes")
        target_genes = processed_genes.join(
            self.gene_set_df,
            on='gene_id',
            how='inner'
        )
        self.logger.info(f"Identified {len(target_genes)} target genes after filtering")
        
        # Step 3: Compute gene distances
        self.logger.info("Step 3: Computing gene distances")
        distances_df = compute_gene_distances(self.gene_coords_df)
        
        # Step 4: Find control genes (genes far enough from target genes)
        self.logger.info("Step 4: Finding control genes")
        min_distance = self.config.analysis_params.get('min_distance', 1000000)  # 1 Mb default
        control_genes = find_control_genes(
            target_genes,
            distances_df,
            min_distance=min_distance
        )
        self.logger.info(f"Found {len(control_genes)} potential control genes")
        
        # Step 5: Match control genes based on confounding factors
        self.logger.info("Step 5: Matching control genes based on confounding factors")
        tolerance = self.config.analysis_params.get('tolerance_range', 0.1)  # 10% default
        matched_controls = match_confounding_factors(
            target_genes,
            control_genes,
            self.factors_df,
            tolerance=tolerance
        )
        self.logger.info(f"Matched {len(matched_controls)} control genes")
        
        # Step 6: Perform enrichment analysis for each population
        self.logger.info("Step 6: Performing enrichment analysis")
        results = {}
        for population in self.population_names:
            self.logger.info(f"Analyzing population: {population}")
            
            # Get scores for target and control genes
            target_scores = processed_genes.join(
                self.gene_set_df,
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
            
            results[population] = enrichment_stats
            self.logger.info(
                f"Population {population}: Enrichment ratio = {enrichment_stats['enrichment_ratio']:.4f}, "
                f"p-value = {enrichment_stats['p_value']:.6f}"
            )
        
        # Step 7: Perform FDR correction
        self.logger.info("Step 7: Performing FDR correction")
        p_values = np.array([results[pop]['p_value'] for pop in self.population_names])
        fdr_results = perform_fdr_analysis(p_values)
        
        # Add FDR results to population results
        for i, population in enumerate(self.population_names):
            results[population]['fdr_corrected_p_value'] = float(fdr_results['pvals_corrected'][i])
            results[population]['significant'] = bool(fdr_results['reject'][i])
        
        # Step 8: Perform bootstrap analysis (if enabled)
        if self.config.config.get('bootstrap', {}).get('run', True):
            self.logger.info("Step 8: Performing bootstrap analysis")
            bootstrap_iterations = self.config.config.get('bootstrap', {}).get('iterations', 1000)
            bootstrap_runs = self.config.config.get('bootstrap', {}).get('runs', 10)
            
            # Run bootstrap analysis sequentially for each population
            bootstrap_results = {}
            
            for population in self.population_names:
                self.logger.info(f"Running bootstrap analysis for population: {population}")
                population_results = []
                
                # Get scores for target and control genes
                target_scores = processed_genes.join(
                    self.gene_set_df,
                    on='gene_id',
                    how='inner'
                )[population].to_numpy()
                
                control_scores = processed_genes.join(
                    matched_controls.select(['gene_id']),
                    on='gene_id',
                    how='inner'
                )[population].to_numpy()
                
                # Run multiple bootstrap analyses
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
                    
                bootstrap_results[population] = population_results
            
            # Add bootstrap results to the main results
            for population in self.population_names:
                results[population]['bootstrap'] = bootstrap_results[population]
        
        # Step 9: Perform FDR analysis with permutations (if enabled)
        if self.config.config.get('fdr', {}).get('run', True):
            self.logger.info("Step 9: Performing FDR analysis with permutations")
            fdr_iterations = self.config.config.get('fdr', {}).get('number', 1000)
            segments = self.config.config.get('fdr', {}).get('shuffling_segments', 1)
            
            # Get chromosome sizes for genome shuffling
            self.logger.info("Preparing chromosome size data for shuffling")
            chrom_sizes = {}
            for row in self.gene_coords_df.group_by('chrom').agg(
                pl.max('end').alias('size')
            ).to_dicts():
                chrom_sizes[row['chrom']] = row['size']
            
            # Run permutations in parallel using proper multiprocessing approach
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
                observed = results[population]['enrichment_ratio']
                permuted = [r[population] for r in permutation_results if population in r and not np.isnan(r[population])]
                
                if len(permuted) > 0:
                    # Calculate empirical p-value
                    emp_pvalue, is_significant = calculate_significance(observed, permuted)
                    results[population]['empirical_p_value'] = emp_pvalue
                    results[population]['empirically_significant'] = is_significant
                else:
                    results[population]['empirical_p_value'] = float('nan')
                    results[population]['empirically_significant'] = False
        
        # Step 10: Save results
        self.logger.info("Step 10: Saving results")
        self.results = results
        self.save_results()
        
        # Log completion time
        elapsed_time = time.time() - start_time
        self.logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")

    def save_results(self, output_dir: Optional[str] = None):
        """Save analysis results.

        Args:
            output_dir: Optional output directory path. If not provided,
                       uses the directory from the configuration.
        """
        if not hasattr(self, 'results'):
            self.logger.warning("No results to save. Run the pipeline first.")
            return
            
        if output_dir is None:
            output_dir = self.config.output_config['directory']
        
        # Create output directories
        output_path = Path(output_dir)
        data_path = ensure_dir(output_path / 'data')
        plots_path = ensure_dir(output_path / 'plots')
        
        # 1. Save main results as JSON
        results_file = data_path / 'enrichment_results.json'
        with open(results_file, 'w') as f:
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
                    
            cleaned_results = clean_for_json(self.results)
            json.dump(cleaned_results, f, indent=2)
        
        self.logger.info(f"Saved results to {results_file}")
        
        # 2. Save tabular summary
        summary_data = []
        for population, result in self.results.items():
            summary_data.append({
                'Population': population,
                'EnrichmentRatio': result['enrichment_ratio'],
                'PValue': result['p_value'],
                'FDRCorrectedPValue': result.get('fdr_corrected_p_value', float('nan')),
                'Significant': result.get('significant', False),
                'EmpiricalPValue': result.get('empirical_p_value', float('nan')),
                'EmpiricallySignificant': result.get('empirically_significant', False)
            })
        
        summary_df = pl.DataFrame(summary_data)
        summary_file = data_path / 'enrichment_summary.csv'
        summary_df.write_csv(summary_file)
        
        self.logger.info(f"Saved summary to {summary_file}")
        
        # 3. Save detailed data
        # Save target genes
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
        
        # Save pipeline configuration
        config_file = data_path / 'pipeline_config.json'
        with open(config_file, 'w') as f:
            # Convert the configuration to a JSON-serializable format
            config_dict = {
                'input_files': {k: str(v) for k, v in self.config.input_files.items() 
                               if isinstance(v, (str, bytes, os.PathLike))},
                'output': self.config.output_config,
                'analysis': self.config.analysis_params,
                'exclude_prefixes': list(self.config.exclude_prefixes),
                'num_threads': self.config.num_threads
            }
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Saved configuration to {config_file}")
        
        # Create README with explanation of output files
        readme_file = output_path / 'README.md'
        with open(readme_file, 'w') as f:
            f.write("# Gene Set Enrichment Analysis Results\n\n")
            f.write(f"Analysis completed on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Files\n\n")
            f.write("- `data/enrichment_results.json`: Complete results including all statistics\n")
            f.write("- `data/enrichment_summary.csv`: Tabular summary of enrichment results\n")
            f.write("- `data/target_genes.csv`: Target genes used in the analysis\n")
            f.write("- `data/pipeline_config.json`: Configuration used for this analysis\n\n")
            
            f.write("## How to Use These Results\n\n")
            f.write("### Plotting\n\n")
            f.write("You can use the CSV files with tools like Python (matplotlib, seaborn), R, or any other data visualization tool.\n\n")
            
            f.write("Example Python code for visualization:\n\n")
            f.write("```python\n")
            f.write("import pandas as pd\n")
            f.write("import matplotlib.pyplot as plt\n")
            f.write("import seaborn as sns\n\n")
            
            f.write("# Load summary\n")
            f.write("summary = pd.read_csv('data/enrichment_summary.csv')\n\n")
            
            f.write("# Create enrichment plot\n")
            f.write("plt.figure(figsize=(10, 6))\n")
            f.write("ax = sns.barplot(x='Population', y='EnrichmentRatio', data=summary)\n")
            f.write("ax.axhline(y=1, color='red', linestyle='--')\n")
            f.write("ax.set_title('Enrichment Results by Population')\n")
            f.write("plt.xticks(rotation=45, ha='right')\n")
            f.write("plt.tight_layout()\n")
            f.write("plt.savefig('plots/enrichment_plot.png', dpi=300)\n")
            f.write("```\n")
        
        self.logger.info(f"Saved README to {readme_file}")
        
        # If intermediate files should be saved
        if self.config.output_config.get('save_intermediate', False):
            # Save various intermediate data products
            intermediate_path = ensure_dir(output_path / 'intermediate')
            
            # Example: gene distances
            distances_df = compute_gene_distances(self.gene_coords_df)
            distances_file = intermediate_path / 'gene_distances.csv'
            distances_df.write_csv(distances_file)
            
            self.logger.info(f"Saved intermediate files to {intermediate_path}")