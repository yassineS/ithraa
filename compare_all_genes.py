#!/usr/bin/env python3

"""
Script to compare traditional gene matching with Mahalanobis distance-based matching using all VIP genes.

This script:
1. Loads gene data and factor information
2. Loads a virus interacting protein (VIP) gene set (using all genes, not just a sample)
3. Matches genes using both traditional and Mahalanobis distance methods
4. Visualises the results using PCA and t-SNE, with connecting segments between target genes and matches
5. Generates distribution plots for Mahalanobis and Euclidean distances

Usage:
    python compare_all_genes.py --gene-set path/to/gene_set
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path

# Import functions from the ithraa package
from ithraa.data import (
    load_gene_list,
    load_gene_coords,
    load_factors,
    load_valid_genes,
    find_control_genes,
    match_confounding_factors,
    compute_gene_distances
)
from ithraa.stats import (
    match_genes_mahalanobis
)
from ithraa.visualise import (
    create_comparison_plots,
    load_vip_gene_set,
    calculate_distances,
    plot_distance_distributions
)
from ithraa.utils import ensure_dir

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare gene matching methods using ALL genes in a VIP gene set")
    parser.add_argument(
        "--gene-set", 
        type=str, 
        default="example/vips/temp_inter_coronaviruses_may2020",
        help="Path to the gene set file (e.g., example/vips/temp_inter_coronaviruses_may2020)"
    )
    parser.add_argument(
        "--min-distance", 
        type=int, 
        default=500000, 
        help="Minimum distance for traditional matching in bp (default: 500000)"
    )
    parser.add_argument(
        "--tolerance", 
        type=float, 
        default=0.5, 
        help="Tolerance for traditional factor matching (default: 0.5)"
    )
    return parser.parse_args()

def setup_paths(gene_set_path=None):
    """Set up paths to data files in the example directory."""
    base_dir = Path(__file__).parent
    example_dir = base_dir / "example"
    
    paths = {
        "gene_list": example_dir / "data" / "gene_ranks_ESN_only.txt",
        "gene_coords": example_dir / "data" / "gene_coords_ensembl_v69.txt",
        "factors": example_dir / "data" / "factors.txt",
        "valid_genes": example_dir / "data" / "valid_genes.txt",
        "output_dir": base_dir / "results" / "all_genes_comparison",
        "gene_set": Path(gene_set_path) if gene_set_path else None
    }
    
    # Create output directory if it doesn't exist
    ensure_dir(paths["output_dir"])
    
    return paths

def load_data(paths):
    """Load and process data from the specified files using ithraa functions."""
    # Load gene list with scores
    gene_list_df, populations = load_gene_list(paths["gene_list"])
    
    # Load gene coordinates
    gene_coords_df = load_gene_coords(paths["gene_coords"])
    
    # Load confounding factors
    factors_df = load_factors(paths["factors"])
    
    # Load valid genes list (if available)
    valid_genes = None
    if paths["valid_genes"].exists():
        valid_genes = load_valid_genes(paths["valid_genes"])
    
    # Filter to keep only common genes across all datasets
    genes_with_coords = set(gene_coords_df["gene_id"].to_list())
    genes_with_factors = set(factors_df["gene_id"].to_list())
    
    # Get common genes across all datasets
    common_genes = genes_with_coords.intersection(genes_with_factors)
    if valid_genes:
        common_genes = common_genes.intersection(valid_genes)
    
    # Filter dataframes to include only common genes
    gene_list_df = gene_list_df.filter(pl.col("gene_id").is_in(common_genes))
    gene_coords_df = gene_coords_df.filter(pl.col("gene_id").is_in(common_genes))
    factors_df = factors_df.filter(pl.col("gene_id").is_in(common_genes))
    
    # Load VIP gene set
    gene_set_df = None
    if paths["gene_set"] and paths["gene_set"].exists():
        gene_set_df = load_vip_gene_set(paths["gene_set"])
        # Filter gene set to keep only genes in common_genes
        gene_set_df = gene_set_df.filter(pl.col("gene_id").is_in(common_genes))
        print(f"Loaded {len(gene_set_df)} genes from VIP gene set {paths['gene_set'].name}")
    else:
        raise ValueError("Gene set file is required")
    
    return gene_list_df, gene_coords_df, factors_df, gene_set_df, populations


def main():
    """Main function to run the comparison with all VIP genes."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set up paths
    paths = setup_paths(args.gene_set)
    
    # Load data
    print("Loading data...")
    gene_list_df, gene_coords_df, factors_df, gene_set_df, populations = load_data(paths)
    print(f"Loaded {len(gene_list_df)} genes, {len(factors_df.columns)-1} factors, {len(gene_set_df)} genes in gene set")
    
    # Use all genes in the gene set
    print(f"Using all {len(gene_set_df)} genes from the VIP gene set")
    
    # Compute pairwise distances between genes
    distances_df = compute_gene_distances(gene_coords_df)
    
    # Run traditional matching
    print("Running traditional matching...")
    # Find control genes (genes at least min_distance away from target genes)
    control_genes_df = find_control_genes(gene_set_df, distances_df, min_distance=args.min_distance)
    
    # Match control genes based on confounding factors
    # Use n_matches=1 to ensure one match per target gene
    trad_matched_df = match_confounding_factors(
        gene_set_df,
        control_genes_df,
        factors_df,
        tolerance=args.tolerance,
        n_matches=1
    )
    print(f"Found {len(trad_matched_df)} traditional matches")
    
    # Run Mahalanobis matching
    print("Running Mahalanobis matching...")
    # Get factor column names for Mahalanobis matching
    factor_cols = [col for col in factors_df.columns if col != "gene_id"]
    
    # Add a default distance column to the factors dataframe
    # This is needed for the Mahalanobis function which expects a distance column
    factors_with_distance = factors_df.with_columns(pl.lit(0.0).alias("distance"))
    
    # Create a version of gene_set_df with the distance column and all factor columns
    gene_set_with_factors = gene_set_df.join(
        factors_df,
        on="gene_id",
        how="left"
    ).with_columns(pl.lit(0.0).alias("distance"))
    
    # Run Mahalanobis matching directly
    maha_matched_df = match_genes_mahalanobis(
        gene_set_with_factors,
        factors_with_distance,
        distance_col="distance",
        confounders=factor_cols,
        exclude_gene_ids=set(gene_set_df["gene_id"].to_list()),
        n_matches=1
    )
    print(f"Found {len(maha_matched_df)} Mahalanobis matches")
    
    # Create comparison plots using the visualise module
    print("Creating comparison plots...")
    create_comparison_plots(
        gene_set_df,
        trad_matched_df,
        maha_matched_df,
        factors_df,
        distances_df,
        paths["output_dir"],
        methods=["pca"]  # Focus on PCA for clarity when using all genes
    )
    
    # Calculate distances between targets and matches
    print("Calculating distances between target genes and their matches...")
    trad_distances = calculate_distances(gene_set_df, trad_matched_df, factors_df)
    maha_distances = calculate_distances(gene_set_df, maha_matched_df, factors_df)
    
    # Plot distance distributions
    print("Generating distance distribution plots...")
    plot_distance_distributions(
        trad_distances,
        maha_distances,
        paths["output_dir"]
    )
    
    print("Comparison complete! Results saved to", paths["output_dir"])


if __name__ == "__main__":
    main()