"""
Collection of functions to visualise outputs and diagnostics 
of the ithraa package modules performance.
"""

from typing import List, Dict

import numpy as np
import polars as pl

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