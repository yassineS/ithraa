#!/usr/bin/env python3

"""
Script to compare traditional gene matching with Mahalanobis distance-based matching.

This script:
1. Loads gene data and factor information
2. Loads a virus interacting protein (VIP) gene set
3. Matches genes using both traditional and Mahalanobis distance methods
4. Visualises the results using PCA and t-SNE

Usage:
    python compare_matching.py --gene-set path/to/gene_set
"""

import os
import random
import argparse
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Import functions from the ithraa package
from ithraa.data import (
    load_gene_list,
    load_gene_set,
    load_gene_coords,
    load_factors,
    load_valid_genes,
    find_control_genes,
    match_confounding_factors,
    compute_gene_distances
)
from ithraa.stats import (
    match_genes_mahalanobis,
)
from ithraa.visualise import (
    create_comparison_plots,
    load_vip_gene_set
)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare gene matching methods using a VIP gene set")
    parser.add_argument(
        "--gene-set", 
        type=str, 
        default=None,
        help="Path to the gene set file (e.g., example/vips/temp_inter_coronaviruses_may2020)"
    )
    parser.add_argument(
        "--n-genes", 
        type=int, 
        default=10, 
        help="Number of genes to sample from the gene set (default: 10)"
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
        default=0.2, 
        help="Tolerance for traditional factor matching (default: 0.2)"
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
        "output_dir": base_dir / "results" / "matching_comparison",
        "gene_set": Path(gene_set_path) if gene_set_path else None
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(paths["output_dir"], exist_ok=True)
    
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
    
    # Filter gene_list to keep only genes with coordinates and factors
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
    
    # Load VIP gene set if provided
    gene_set_df = None
    if paths["gene_set"] and paths["gene_set"].exists():
        gene_set_df = load_vip_gene_set(paths["gene_set"])
        # Filter gene set to keep only genes in common_genes
        gene_set_df = gene_set_df.filter(pl.col("gene_id").is_in(common_genes))
        print(f"Loaded {len(gene_set_df)} genes from VIP gene set {paths['gene_set'].name}")
    else:
        # Fallback to using top 100 genes from the first population as before
        print("No VIP gene set provided, using top 100 genes from the first population")
        if len(populations) > 0:
            population = populations[0]
            gene_set_df = gene_list_df.sort(population, descending=True).head(100).select(["gene_id"])
        else:
            # If no populations found, just take the first 100 genes
            gene_set_df = gene_list_df.head(100).select(["gene_id"])
    
    return gene_list_df, gene_coords_df, factors_df, gene_set_df, populations


def select_random_genes(gene_set_df, n=10):
    """Select n random genes from the gene set."""
    genes = gene_set_df["gene_id"].to_list()
    if len(genes) <= n:
        return gene_set_df
    
    # Select random genes
    selected_genes = random.sample(genes, n)
    return gene_set_df.filter(pl.col("gene_id").is_in(selected_genes))


def run_traditional_matching(gene_set_df, gene_coords_df, factors_df, min_distance=1000000, tolerance=0.1):
    """Run the traditional gene matching procedure using ithraa functions."""
    # Compute pairwise distances between genes
    distances_df = compute_gene_distances(gene_coords_df)
    
    # Find control genes (genes at least min_distance away from target genes)
    control_genes_df = find_control_genes(gene_set_df, distances_df, min_distance=min_distance)
    
    # Match control genes based on confounding factors
    matched_controls_df = match_confounding_factors(
        gene_set_df,
        control_genes_df,
        factors_df,
        tolerance=tolerance
    )
    
    return matched_controls_df, distances_df


def run_mahalanobis_matching(gene_set_df, all_genes_df, distances_df, factors_df, exclude_gene_ids=None, n_matches=1):
    """Run the Mahalanobis distance-based matching."""
    # Get the confounding factor column names
    factor_cols = [col for col in factors_df.columns if col != "gene_id"]
    
    # For each gene in the gene set, prepare a dataset with all other genes and their distances
    gene_set_data_list = []
    
    for target_gene_id in gene_set_df["gene_id"].to_list():
        # Get all distances for this target gene
        target_distances = distances_df.filter(
            (pl.col("gene_id") == target_gene_id) | (pl.col("target_gene") == target_gene_id)
        )
        
        # Get the confounding factors for this target gene
        target_factors = factors_df.filter(pl.col("gene_id") == target_gene_id)
        
        if target_factors.height == 0:
            print(f"Warning: No factor data found for gene {target_gene_id}")
            continue
            
        # For each other gene, create an entry with the distance and target gene id
        for _, row in enumerate(target_distances.to_dicts()):
            # Determine which gene is the other gene
            if row["gene_id"] == target_gene_id:
                other_gene_id = row["target_gene"]
            else:
                other_gene_id = row["gene_id"]
                
            # Skip if this other gene is in the exclude list
            if exclude_gene_ids and other_gene_id in exclude_gene_ids:
                continue
                
            # Get the other gene's factors
            other_factors = factors_df.filter(pl.col("gene_id") == other_gene_id)
            
            if other_factors.height == 0:
                continue
                
            # Create an entry with gene_id, distance, target_gene_id, and all factors
            entry = {
                "gene_id": other_gene_id,
                "target_gene_id": target_gene_id,
                "distance": row["distance"]
            }
            
            # Add all factor values
            for factor in factor_cols:
                entry[factor] = other_factors[factor][0]
                
            gene_set_data_list.append(entry)
    
    # Create the dataset for Mahalanobis matching
    if not gene_set_data_list:
        print("Warning: No valid gene pairs found for Mahalanobis matching")
        return pl.DataFrame()
        
    gene_set_data_df = pl.DataFrame(gene_set_data_list)
    
    # We need to add a dummy 'distance' column to the background data
    # since match_genes_mahalanobis expects it
    background_data = []
    for _, row in enumerate(all_genes_df.to_dicts()):
        gene_id = row["gene_id"]
        
        # Skip if this gene is in the exclude list
        if exclude_gene_ids and gene_id in exclude_gene_ids:
            continue
            
        # Get the factors for this gene
        gene_factors = factors_df.filter(pl.col("gene_id") == gene_id)
        
        if gene_factors.height == 0:
            continue
            
        # Create an entry with gene_id, dummy distance, and all factors
        entry = {
            "gene_id": gene_id,
            "distance": 0  # Dummy value, will be replaced during Mahalanobis calculation
        }
        
        # Add all factor values
        for factor in factor_cols:
            entry[factor] = gene_factors[factor][0]
            
        background_data.append(entry)
    
    background_data_df = pl.DataFrame(background_data)
    
    # Run Mahalanobis matching
    matched_genes_df = match_genes_mahalanobis(
        gene_set_data_df,
        background_data_df,
        distance_col="distance",
        confounders=factor_cols,
        exclude_gene_ids=exclude_gene_ids,
        n_matches=n_matches
    )
    
    return matched_genes_df


def main():
    """Main function to run the comparison."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Set up paths
    paths = setup_paths(args.gene_set)
    
    # Load data
    print("Loading data...")
    gene_list_df, gene_coords_df, factors_df, gene_set_df, populations = load_data(paths)
    print(f"Loaded {len(gene_list_df)} genes, {len(factors_df.columns)-1} factors, {len(gene_set_df)} genes in gene set")
    
    # Select random subset of genes if needed
    if args.n_genes < len(gene_set_df):
        random_gene_set_df = select_random_genes(gene_set_df, n=args.n_genes)
        print(f"Selected {len(random_gene_set_df)} random genes from the gene set")
    else:
        random_gene_set_df = gene_set_df
        print(f"Using all {len(random_gene_set_df)} genes from the gene set (fewer than requested {args.n_genes})")
    
    # Run traditional matching
    print("Running traditional matching...")
    trad_matched_df, distances_df = run_traditional_matching(
        random_gene_set_df,
        gene_coords_df,
        factors_df,
        min_distance=args.min_distance,
        tolerance=args.tolerance
    )
    print(f"Found {len(trad_matched_df)} traditional matches")
    
    # Run Mahalanobis matching
    print("Running Mahalanobis matching...")
    maha_matched_df = run_mahalanobis_matching(
        random_gene_set_df,
        gene_coords_df,
        distances_df,
        factors_df,
        exclude_gene_ids=set(random_gene_set_df["gene_id"].to_list()),
        n_matches=1  # Match 1 gene per target gene
    )
    print(f"Found {len(maha_matched_df)} Mahalanobis matches")
    
    # Create comparison plots using the visualise module
    print("Creating comparison plots...")
    create_comparison_plots(
        random_gene_set_df,
        trad_matched_df,
        maha_matched_df,
        factors_df,
        distances_df,
        paths["output_dir"],
        methods=["pca", "tsne"]
    )
    
    print("Comparison complete!")


if __name__ == "__main__":
    main()