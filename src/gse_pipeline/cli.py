#!/usr/bin/env python3
"""
Command line interface for the gene set enrichment pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
import tomli
from tomli_w import dump
from .pipeline import GeneSetEnrichmentPipeline
from .config import PipelineConfig

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run gene set enrichment analysis pipeline"
    )
    
    # Required arguments
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to TOML configuration file"
    )
    
    # Input file overrides
    input_group = parser.add_argument_group("Input file overrides")
    input_group.add_argument(
        "--gene-list",
        type=str,
        help="Override gene list file path"
    )
    input_group.add_argument(
        "--gene-coords",
        type=str,
        help="Override gene coordinates file path"
    )
    input_group.add_argument(
        "--factors",
        type=str,
        help="Override confounding factors file path"
    )
    input_group.add_argument(
        "--valid-genes",
        type=str,
        help="Override valid genes file path"
    )
    input_group.add_argument(
        "--hgnc",
        type=str,
        help="Override HGNC file path"
    )
    
    # Output configuration overrides
    output_group = parser.add_argument_group("Output configuration overrides")
    output_group.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory"
    )
    output_group.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate files"
    )
    
    # Analysis parameter overrides
    analysis_group = parser.add_argument_group("Analysis parameter overrides")
    analysis_group.add_argument(
        "--min-distance",
        type=int,
        help="Override minimum distance"
    )
    analysis_group.add_argument(
        "--tolerance-range",
        type=float,
        help="Override tolerance range"
    )
    analysis_group.add_argument(
        "--max-rep",
        type=int,
        help="Override maximum repetitions"
    )
    analysis_group.add_argument(
        "--flip",
        action="store_true",
        help="Override flip parameter"
    )
    analysis_group.add_argument(
        "--cluster-distance",
        type=int,
        help="Override cluster distance"
    )
    analysis_group.add_argument(
        "--num-threads",
        type=int,
        help="Override number of threads for parallel processing"
    )
    
    # Bootstrap parameter overrides
    bootstrap_group = parser.add_argument_group("Bootstrap parameter overrides")
    bootstrap_group.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Disable bootstrap analysis"
    )
    bootstrap_group.add_argument(
        "--bootstrap-iterations",
        type=int,
        help="Override number of bootstrap iterations"
    )
    bootstrap_group.add_argument(
        "--bootstrap-runs",
        type=int,
        help="Override number of bootstrap runs"
    )
    bootstrap_group.add_argument(
        "--simultaneous-runs",
        type=int,
        help="Override number of simultaneous runs"
    )
    
    # FDR parameter overrides
    fdr_group = parser.add_argument_group("FDR parameter overrides")
    fdr_group.add_argument(
        "--no-fdr",
        action="store_true",
        help="Disable FDR analysis"
    )
    fdr_group.add_argument(
        "--fdr-iterations",
        type=int,
        help="Override number of FDR iterations"
    )
    fdr_group.add_argument(
        "--shuffling-segments",
        type=int,
        help="Override number of shuffling segments"
    )
    fdr_group.add_argument(
        "--interrupted",
        action="store_true",
        help="Override interrupted parameter"
    )
    
    # Sweep parameter overrides
    sweep_group = parser.add_argument_group("Sweep parameter overrides")
    sweep_group.add_argument(
        "--sweep-prefix",
        type=str,
        help="Override sweep prefix"
    )
    sweep_group.add_argument(
        "--sweep-sizes",
        type=str,
        nargs="+",
        help="Override sweep sizes"
    )
    sweep_group.add_argument(
        "--no-count",
        action="store_true",
        help="Disable sweep counting"
    )
    sweep_group.add_argument(
        "--no-run-count",
        action="store_true",
        help="Disable sweep run counting"
    )
    
    return parser.parse_args()

def update_config(config: dict, args: argparse.Namespace):
    """Update configuration with command line overrides."""
    # Input file overrides
    if args.gene_list:
        config['input']['gene_list_file'] = args.gene_list
    if args.gene_coords:
        config['input']['gene_coords_file'] = args.gene_coords
    if args.factors:
        config['input']['factors_file'] = args.factors
    if args.valid_genes:
        config['input']['valid_genes_file'] = args.valid_genes
    if args.hgnc:
        config['input']['hgnc_file'] = args.hgnc
    
    # Output configuration overrides
    if args.output_dir:
        config['output']['directory'] = args.output_dir
    if args.save_intermediate:
        config['output']['save_intermediate'] = True
    
    # Analysis parameter overrides
    if args.min_distance:
        config['analysis']['min_distance'] = args.min_distance
    if args.tolerance_range:
        config['analysis']['tolerance_range'] = args.tolerance_range
    if args.max_rep:
        config['analysis']['max_rep'] = args.max_rep
    if args.flip:
        config['analysis']['flip'] = True
    if args.cluster_distance:
        config['analysis']['cluster_distance'] = args.cluster_distance
    if args.num_threads:
        config['analysis']['num_threads'] = args.num_threads
    
    # Bootstrap parameter overrides
    if args.no_bootstrap:
        config['bootstrap']['run'] = False
    if args.bootstrap_iterations:
        config['bootstrap']['iterations'] = args.bootstrap_iterations
    if args.bootstrap_runs:
        config['bootstrap']['runs'] = args.bootstrap_runs
    if args.simultaneous_runs:
        config['bootstrap']['simultaneous_runs'] = args.simultaneous_runs
    
    # FDR parameter overrides
    if args.no_fdr:
        config['fdr']['run'] = False
    if args.fdr_iterations:
        config['fdr']['number'] = args.fdr_iterations
    if args.shuffling_segments:
        config['fdr']['shuffling_segments'] = args.shuffling_segments
    if args.interrupted:
        config['fdr']['interrupted'] = True
    
    # Sweep parameter overrides
    if args.sweep_prefix:
        config['sweep']['prefix'] = args.sweep_prefix
    if args.sweep_sizes:
        config['sweep']['sizes'] = args.sweep_sizes
    if args.no_count:
        config['sweep']['count'] = False
    if args.no_run_count:
        config['sweep']['run_count'] = False
    
    return config

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Load and validate config file
    try:
        with open(args.config_file, 'rb') as f:
            config = tomli.load(f)
    except Exception as e:
        logging.error(f"Error loading configuration file: {str(e)}")
        sys.exit(1)
    
    # Update config with command line overrides
    config = update_config(config, args)
    
    # Save updated config to a temporary file
    temp_config_path = Path(args.config_file).parent / "temp_config.toml"
    with open(temp_config_path, 'wb') as f:
        dump(config, f)
    
    try:
        # Initialize and run pipeline
        pipeline = GeneSetEnrichmentPipeline(str(temp_config_path))
        pipeline.run()
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up temporary config file
        temp_config_path.unlink()

if __name__ == "__main__":
    main()