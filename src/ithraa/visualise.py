"""
Collection of functions to visualise outputs and diagnostics 
of the ithraa package modules performance.
"""

from typing import List, Dict, Union, Optional, Set
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


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
        reducer = PCA(n_components=2, random_state=random_state)
        components = reducer.fit_transform(features)
        variance_explained = reducer.explained_variance_ratio_
        
        return {
            "coordinates": components,
            "labels": labels,
            "variance_explained": variance_explained
        }
    
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state)
        components = reducer.fit_transform(features)
        
        return {
            "coordinates": components,
            "labels": labels,
            "variance_explained": None
        }
    
    else:
        raise ValueError(f"Unknown visualisation method: {method}. Use 'pca' or 'tsne'.")


def create_comparison_plots(
    gene_set_df: pl.DataFrame,
    traditional_matches: pl.DataFrame,
    mahalanobis_matches: pl.DataFrame,
    factors_df: pl.DataFrame,
    distances_df: pl.DataFrame,
    output_dir: Path,
    methods: List[str] = ["pca", "tsne"]
) -> None:
    """
    Create comparison plots for different gene matching methods.

    Args:
        gene_set_df: DataFrame containing target genes
        traditional_matches: DataFrame containing genes matched using traditional method
        mahalanobis_matches: DataFrame containing genes matched using Mahalanobis distance
        factors_df: DataFrame with confounding factors
        distances_df: DataFrame with gene distances
        output_dir: Directory to save plots
        methods: List of dimensionality reduction methods to use ('pca', 'tsne')
    """
    # Get the confounding factor column names
    factor_cols = [col for col in factors_df.columns if col != "gene_id"]
    
    # Create a dataset for visualisation
    
    # First, get all genes with factors
    all_genes_with_factors = factors_df.clone()
    
    # Add a column for the matching type
    all_genes_with_factors = all_genes_with_factors.with_columns(
        pl.lit("All").alias("match_type")
    )
    
    # Extract factors for gene set
    gene_set_with_factors = gene_set_df.join(factors_df, on="gene_id", how="inner")
    
    # Extract factors for traditional matches
    if "target_gene_id" in traditional_matches.columns:
        # New format with target_gene_id column - just need gene_id for joining
        trad_matches_with_factors = traditional_matches.join(factors_df, on="gene_id", how="inner")
    else:
        # Old format without target_gene_id column
        trad_matches_with_factors = traditional_matches.join(factors_df, on="gene_id", how="inner")
    
    # Check if mahalanobis_matches has any rows before processing
    if mahalanobis_matches.height == 0:
        print("Warning: No Mahalanobis matches to plot")
        maha_matches_with_factors = pl.DataFrame(schema={"gene_id": pl.Utf8, "match_type": pl.Utf8, **{col: factors_df.schema[col] for col in factor_cols}})
    else:
        # Check if Mahalanobis matches already have factors
        missing_factors = [col for col in factor_cols if col not in mahalanobis_matches.columns]
        if missing_factors:
            # Some factors are missing, so join with factors_df
            maha_matches_with_factors = mahalanobis_matches.join(factors_df, on="gene_id", how="inner")
        else:
            # All factors already present
            maha_matches_with_factors = mahalanobis_matches.clone()
            
        # Ensure match_type column is added correctly
        if "match_type" in maha_matches_with_factors.columns:
            # Column already exists, drop it first
            maha_matches_with_factors = maha_matches_with_factors.drop("match_type")
        
        # Add the match_type column
        maha_matches_with_factors = maha_matches_with_factors.with_columns(
            pl.lit("Mahalanobis").alias("match_type")
        )
    
    # Add a column for the matching type for target genes
    gene_set_with_factors = gene_set_with_factors.with_columns(
        pl.lit("Target").alias("match_type")
    )
    
    trad_matches_with_factors = trad_matches_with_factors.with_columns(
        pl.lit("Traditional").alias("match_type")
    )
    
    # Make sure all datasets have the necessary columns with the same dtypes
    needed_cols = ["gene_id", "match_type"]
    for col in factor_cols:
        needed_cols.append(col)
        
        # Make sure the data type is consistent across all dataframes
        # Convert numeric columns to Float64 for consistency
        if all_genes_with_factors.height > 0 and col in all_genes_with_factors.columns:
            all_genes_with_factors = all_genes_with_factors.with_columns(
                pl.col(col).cast(pl.Float64)
            )
            
        if gene_set_with_factors.height > 0 and col in gene_set_with_factors.columns:
            gene_set_with_factors = gene_set_with_factors.with_columns(
                pl.col(col).cast(pl.Float64)
            )
            
        if trad_matches_with_factors.height > 0 and col in trad_matches_with_factors.columns:
            trad_matches_with_factors = trad_matches_with_factors.with_columns(
                pl.col(col).cast(pl.Float64)
            )
            
        if maha_matches_with_factors.height > 0 and col in maha_matches_with_factors.columns:
            maha_matches_with_factors = maha_matches_with_factors.with_columns(
                pl.col(col).cast(pl.Float64)
            )
    
    # Select only the needed columns with consistent data types
    all_genes_cols = [col for col in needed_cols if col in all_genes_with_factors.columns]
    gene_set_cols = [col for col in needed_cols if col in gene_set_with_factors.columns]
    trad_match_cols = [col for col in needed_cols if col in trad_matches_with_factors.columns]
    maha_match_cols = [col for col in needed_cols if col in maha_matches_with_factors.columns]
    
    # Remove genes that are in the gene set, traditional matches, or mahalanobis matches from the all_genes dataset
    # to avoid plotting them twice
    gene_set_ids = set(gene_set_df["gene_id"].to_list())
    trad_match_ids = set(traditional_matches["gene_id"].to_list()) if traditional_matches.height > 0 else set()
    maha_match_ids = set(mahalanobis_matches["gene_id"].to_list()) if mahalanobis_matches.height > 0 else set()
    
    # Combine all ids to exclude
    exclude_ids = gene_set_ids.union(trad_match_ids).union(maha_match_ids)
    
    # Filter all_genes to exclude genes that are already in other categories
    all_genes_with_factors = all_genes_with_factors.filter(
        ~pl.col("gene_id").is_in(list(exclude_ids))
    )
    print(f"Plotting {all_genes_with_factors.height} background genes in light grey")
    
    # Combine all datasets
    all_data_parts = []
    
    # Add all genes first so they appear at the bottom of the plot
    if all_genes_with_factors.height > 0 and len(all_genes_cols) > 0:
        all_data_parts.append(all_genes_with_factors.select(all_genes_cols))
    
    # Then add the matches and targets
    if trad_matches_with_factors.height > 0 and len(trad_match_cols) > 0:
        all_data_parts.append(trad_matches_with_factors.select(trad_match_cols))
        
    if maha_matches_with_factors.height > 0 and len(maha_match_cols) > 0:
        # Don't include mahalanobis_distance in the visualisation data
        all_data_parts.append(maha_matches_with_factors.select(maha_match_cols))
    
    if gene_set_with_factors.height > 0 and len(gene_set_cols) > 0:
        all_data_parts.append(gene_set_with_factors.select(gene_set_cols))
    
    # Check if we have any data to plot
    if not all_data_parts:
        print("No data to plot. Make sure at least one matching method returned results.")
        return
    
    # Convert data types to ensure all dataframes are compatible before concatenation
    for i in range(len(all_data_parts)):
        for col in all_data_parts[i].columns:
            if col != "gene_id" and col != "match_type":
                all_data_parts[i] = all_data_parts[i].with_columns(
                    pl.col(col).cast(pl.Float64)
                )
        
    all_data = pl.concat(all_data_parts)
    
    # Create visualisations for PCA and t-SNE
    for method in methods:
        print(f"Creating {method} visualisation...")
        # Get the feature columns for the plot
        feature_cols = factor_cols.copy()
        
        # Standardise the data
        features = all_data.select(feature_cols).to_numpy()
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get visualisation results
        if method == "pca":
            reducer = PCA(n_components=2, random_state=42)
            components = reducer.fit_transform(features_scaled)
            variance_explained = reducer.explained_variance_ratio_
            
            viz_title = f"PCA Visualisation (Explained variance: {variance_explained[0]:.2f}, {variance_explained[1]:.2f})"
        else:  # t-SNE
            # Use a larger perplexity for the larger dataset, but not more than n_samples / 3
            perplexity = min(50, features_scaled.shape[0] / 3)
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            components = reducer.fit_transform(features_scaled)
            
            viz_title = "t-SNE Visualisation"
        
        # Create the plot
        plt.figure(figsize=(14, 10))
        
        # Define the colour palette with all genes in light grey
        colors = {
            "All": "#DDDDDD",  # Light grey for all genes
            "Traditional": "#3366CC",  # Blue for traditional matches
            "Mahalanobis": "#33CC66",  # Green for Mahalanobis matches
            "Target": "#FF3333"  # Red for target genes
        }
        
        # Define the marker sizes and alphas
        sizes = {
            "All": 15,  # Small size for background genes
            "Traditional": 60,  # Medium size for traditional matches
            "Mahalanobis": 60,  # Medium size for Mahalanobis matches
            "Target": 100  # Large size for target genes
        }
        
        alphas = {
            "All": 0.3,  # More transparent for background genes
            "Traditional": 0.7,
            "Mahalanobis": 0.7,
            "Target": 1.0  # Fully opaque for target genes
        }
        
        # Create a DataFrame for easy plotting with seaborn
        viz_df = pd.DataFrame({
            "Component1": components[:, 0],
            "Component2": components[:, 1],
            "MatchType": all_data["match_type"].to_numpy(),
            "GeneID": all_data["gene_id"].to_numpy(),
        })
        
        # Plot each category separately to control size and alpha
        for match_type, color in colors.items():
            subset = viz_df[viz_df["MatchType"] == match_type]
            if len(subset) > 0:
                plt.scatter(
                    x=subset["Component1"],
                    y=subset["Component2"],
                    c=color,
                    s=sizes[match_type],
                    alpha=alphas[match_type],
                    label=f"{match_type} ({len(subset)} genes)",
                    edgecolors='none'
                )
        
        # Highlight target genes
        target_points = viz_df[viz_df["MatchType"] == "Target"]
        if len(target_points) > 0:
            plt.scatter(
                x=target_points["Component1"],
                y=target_points["Component2"],
                s=sizes["Target"] + 25,  # Slightly larger for the border
                color='none',
                edgecolor="black",
                linewidth=1.5
            )
            
            # Add gene labels for target genes
            for _, row in target_points.iterrows():
                plt.annotate(
                    row["GeneID"],
                    (row["Component1"], row["Component2"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )
                
            # Draw connecting segments between target genes and their matches
            # Only if target_gene_id column exists in the matches DataFrames
            if "target_gene_id" in traditional_matches.columns:
                trad_points = viz_df[viz_df["MatchType"] == "Traditional"]
                
                # For each target gene, find its traditional match and draw a line
                for _, target_row in target_points.iterrows():
                    target_id = target_row["GeneID"]
                    
                    # Find the traditional match for this target gene - using proper Polars filtering
                    trad_matches_for_target = traditional_matches.filter(pl.col("target_gene_id") == target_id)
                    if trad_matches_for_target.height > 0:
                        match_gene_ids = trad_matches_for_target["gene_id"].to_list()
                        trad_match = trad_points[trad_points["GeneID"].isin(match_gene_ids)]
                        
                        if not trad_match.empty:
                            # Draw a line from target to traditional match
                            plt.plot(
                                [target_row["Component1"], trad_match["Component1"].values[0]],
                                [target_row["Component2"], trad_match["Component2"].values[0]],
                                color=colors["Traditional"],
                                linestyle='-',
                                linewidth=0.7,
                                alpha=0.6,
                                zorder=1  # Draw lines below points
                            )
            
            if "target_gene_id" in mahalanobis_matches.columns:
                maha_points = viz_df[viz_df["MatchType"] == "Mahalanobis"]
                
                # For each target gene, find its Mahalanobis match and draw a line
                for _, target_row in target_points.iterrows():
                    target_id = target_row["GeneID"]
                    
                    # Find the Mahalanobis match for this target gene - using proper Polars filtering
                    maha_matches_for_target = mahalanobis_matches.filter(pl.col("target_gene_id") == target_id)
                    if maha_matches_for_target.height > 0:
                        match_gene_ids = maha_matches_for_target["gene_id"].to_list()
                        maha_match = maha_points[maha_points["GeneID"].isin(match_gene_ids)]
                        
                        if not maha_match.empty:
                            # Draw a line from target to Mahalanobis match
                            plt.plot(
                                [target_row["Component1"], maha_match["Component1"].values[0]],
                                [target_row["Component2"], maha_match["Component2"].values[0]],
                                color=colors["Mahalanobis"],
                                linestyle='-',
                                linewidth=0.7,
                                alpha=0.6,
                                zorder=1  # Draw lines below points
                            )
        
        plt.title(viz_title, fontsize=14)
        plt.xlabel("Component 1", fontsize=12)
        plt.ylabel("Component 2", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.legend(loc="upper right", markerscale=0.7)
        plt.tight_layout()
        
        # Save the plot
        output_path = output_dir / f"gene_matching_comparison_{method.lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved {method} visualisation to {output_path}")
        plt.close()


def load_vip_gene_set(gene_set_path: Path) -> pl.DataFrame:
    """
    Load a virus interacting protein gene set file.
    
    Args:
        gene_set_path: Path to the gene set file
        
    Returns:
        DataFrame with gene IDs of interacting proteins
    """
    if not gene_set_path or not Path(gene_set_path).exists():
        raise FileNotFoundError(f"Gene set file not found: {gene_set_path}")
    
    try:
        # Try to load the file as a tab-separated file with a header
        df = pl.read_csv(
            gene_set_path,
            separator='\t',
            has_header=True,
            ignore_errors=True
        )
        
        # If there are at least two columns, look for 'yes' values
        if len(df.columns) >= 2:
            # Identify potential interaction columns (those containing 'yes' values)
            potential_cols = []
            for col in df.columns[1:]:  # Skip the first column (assumed to be gene_id)
                if df[col].dtype == pl.Utf8 and 'yes' in df[col].to_list():
                    potential_cols.append(col)
            
            if potential_cols:
                # Use the first column with 'yes' values
                interaction_col = potential_cols[0]
                print(f"Found interaction column: {interaction_col}")
                
                # Filter to keep only genes with 'yes' in the interaction column
                df = df.filter(pl.col(interaction_col) == 'yes')
                
                # If there's a column named 'gene_id', use that, otherwise use the first column
                if 'gene_id' in df.columns:
                    return df.select(['gene_id'])
                else:
                    return df.rename({df.columns[0]: 'gene_id'}).select(['gene_id'])
        
        # If we couldn't find interaction columns or if the file has only one column
        # Assume the first column contains gene IDs
        return df.rename({df.columns[0]: 'gene_id'}).select(['gene_id'])
        
    except Exception as e:
        print(f"Error parsing the file as tab-separated: {e}")
        print("Falling back to loading as a simple list of gene IDs")
        
        # Fallback to loading as a simple list of gene IDs
        with open(gene_set_path, 'r') as f:
            gene_ids = [line.strip() for line in f if line.strip()]
        
        # Create a DataFrame with the gene IDs
        return pl.DataFrame({"gene_id": gene_ids})


def calculate_distances(
    gene_set_df: pl.DataFrame,
    matched_df: pl.DataFrame,
    factors_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate both Mahalanobis and Euclidean distances between target genes and their matches.
    
    Args:
        gene_set_df: DataFrame containing target genes
        matched_df: DataFrame containing matched genes with target_gene_id column
        factors_df: DataFrame with confounding factors
        
    Returns:
        DataFrame with gene_id, target_gene_id, mahalanobis_distance, euclidean_distance
    """
    from scipy.spatial.distance import euclidean, mahalanobis
    from numpy.linalg import inv
    
    if "target_gene_id" not in matched_df.columns:
        raise ValueError("matched_df must have a target_gene_id column")
    
    # Get factor columns
    factor_cols = [col for col in factors_df.columns if col != "gene_id"]
    
    # Add factors to target genes and matches
    gene_set_with_factors = gene_set_df.join(factors_df, on="gene_id", how="inner")
    matched_with_factors = matched_df.join(factors_df, on="gene_id", how="inner")
    
    # Calculate covariance matrix once for all calculations
    all_genes_features = factors_df.select(factor_cols).to_numpy()
    cov_matrix = np.cov(all_genes_features, rowvar=False)
    
    # Add a small regularization term to ensure the matrix is invertible
    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    
    # Calculate inverse covariance matrix
    try:
        inv_cov = inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if regular inverse fails
        inv_cov = np.linalg.pinv(cov_matrix)
    
    # Prepare result containers
    results = []
    
    # Calculate distances for each match
    for match_row in matched_with_factors.to_dicts():
        gene_id = match_row["gene_id"]
        target_gene_id = match_row["target_gene_id"]
        
        # Get target gene factors
        target_row = gene_set_with_factors.filter(pl.col("gene_id") == target_gene_id)
        
        if target_row.height == 0:
            continue
            
        # Extract feature vectors
        match_features = np.array([match_row[col] for col in factor_cols])
        target_features = np.array([target_row[0, col] for col in factor_cols])
        
        # Calculate Mahalanobis distance
        maha_dist = mahalanobis(match_features, target_features, inv_cov)
        
        # Calculate Euclidean distance
        eucl_dist = euclidean(match_features, target_features)
        
        # Store results
        results.append({
            "gene_id": gene_id,
            "target_gene_id": target_gene_id,
            "mahalanobis_distance": float(maha_dist),
            "euclidean_distance": float(eucl_dist)
        })
    
    # Create DataFrame from results
    if not results:
        return pl.DataFrame(schema={
            "gene_id": pl.Utf8, 
            "target_gene_id": pl.Utf8, 
            "mahalanobis_distance": pl.Float64, 
            "euclidean_distance": pl.Float64
        })
        
    return pl.DataFrame(results)


def plot_distance_distributions(
    traditional_distances: pl.DataFrame,
    mahalanobis_distances: pl.DataFrame,
    output_dir: Path
) -> None:
    """
    Plot distributions of Mahalanobis and Euclidean distances for both matching methods.
    
    Args:
        traditional_distances: DataFrame with distances for traditional matches
        mahalanobis_distances: DataFrame with distances for Mahalanobis matches
        output_dir: Directory to save plots
    """
    if traditional_distances.height == 0 and mahalanobis_distances.height == 0:
        print("No distances to plot")
        return
        
    # Plot Mahalanobis distances
    plt.figure(figsize=(10, 6))
    
    # Set up the plot
    plt.title("Distribution of Mahalanobis Distances", fontsize=14)
    plt.xlabel("Mahalanobis Distance", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    # Plot the distributions
    if traditional_distances.height > 0:
        trad_distances = traditional_distances["mahalanobis_distance"].to_numpy()
        plt.hist(trad_distances, alpha=0.6, label=f"Traditional (mean={np.mean(trad_distances):.2f})", 
                 color="#3366CC", edgecolor="black", bins=20)
    
    if mahalanobis_distances.height > 0:
        maha_distances = mahalanobis_distances["mahalanobis_distance"].to_numpy()
        plt.hist(maha_distances, alpha=0.6, label=f"Mahalanobis (mean={np.mean(maha_distances):.2f})", 
                 color="#33CC66", edgecolor="black", bins=20)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    mahalanobis_output_path = output_dir / "mahalanobis_distance_distribution.png"
    plt.savefig(mahalanobis_output_path, dpi=300)
    print(f"Saved Mahalanobis distance distribution to {mahalanobis_output_path}")
    plt.close()
    
    # Plot Euclidean distances
    plt.figure(figsize=(10, 6))
    
    # Set up the plot
    plt.title("Distribution of Euclidean Distances", fontsize=14)
    plt.xlabel("Euclidean Distance", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    
    # Plot the distributions
    if traditional_distances.height > 0:
        trad_distances = traditional_distances["euclidean_distance"].to_numpy()
        plt.hist(trad_distances, alpha=0.6, label=f"Traditional (mean={np.mean(trad_distances):.2f})", 
                 color="#3366CC", edgecolor="black", bins=20)
    
    if mahalanobis_distances.height > 0:
        maha_distances = mahalanobis_distances["euclidean_distance"].to_numpy()
        plt.hist(maha_distances, alpha=0.6, label=f"Mahalanobis (mean={np.mean(maha_distances):.2f})", 
                 color="#33CC66", edgecolor="black", bins=20)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    euclidean_output_path = output_dir / "euclidean_distance_distribution.png"
    plt.savefig(euclidean_output_path, dpi=300)
    print(f"Saved Euclidean distance distribution to {euclidean_output_path}")
    plt.close()