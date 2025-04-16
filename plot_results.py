#!/usr/bin/env python3
"""
Script to plot gene set enrichment results.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot gene set enrichment results"
    )
    
    # Required arguments
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory (default: results)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Path to output directory for plots (default: plots)"
    )
    
    parser.add_argument(
        "--gene-set",
        type=str,
        help="Name of the gene set (for plot titles)"
    )
    
    parser.add_argument(
        "--save-format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg", "jpg"],
        help="Format for saving plots (default: png)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saving plots (default: 300)"
    )
    
    return parser.parse_args()

def load_results(results_dir):
    """
    Load enrichment results from the results directory.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        dict: Enrichment results data
    """
    results_path = Path(results_dir) / "data" / "enrichment_results.json"
    summary_path = Path(results_dir) / "data" / "enrichment_summary.csv"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    # Load the full results JSON
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Load the summary CSV
    summary = pd.read_csv(summary_path)
    
    return {
        'full_results': results,
        'summary': summary
    }

def plot_enrichment_barplot(summary, output_path, gene_set_name=None, figsize=(10, 6), dpi=300, format="png"):
    """
    Create a bar plot of enrichment ratios for all populations.
    
    Args:
        summary: DataFrame with enrichment summary
        output_path: Path to save the plot
        gene_set_name: Name of gene set for the title
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the plot
        format: File format for saving the plot
    """
    plt.figure(figsize=figsize)
    
    # Create a custom color palette based on significance
    colors = summary.apply(
        lambda x: "#c03a2b" if x["Significant"] else "#7f8c8d",  # Red if significant, gray if not
        axis=1
    )
    
    # Create the bar plot
    ax = sns.barplot(
        x="Population",
        y="EnrichmentRatio",
        data=summary,
        palette=colors
    )
    
    # Add a horizontal line at y=1 (no enrichment)
    plt.axhline(y=1, color="black", linestyle="--", alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(summary["EnrichmentRatio"]):
        ax.text(
            i,
            v + 0.05,
            f"{v:.2f}",
            ha="center",
            fontsize=9
        )
    
    # Add significance stars
    for i, (_, row) in enumerate(summary.iterrows()):
        if row["Significant"]:
            ax.text(
                i,
                row["EnrichmentRatio"] + 0.15,
                "*",
                ha="center",
                fontsize=16,
                fontweight="bold"
            )
    
    # Customize the plot
    plt.xlabel("Population", fontsize=12)
    plt.ylabel("Enrichment Ratio", fontsize=12)
    
    title = "Gene Set Enrichment Results"
    if gene_set_name:
        title += f" - {gene_set_name}"
    plt.title(title, fontsize=14)
    
    # Add a legend
    red_patch = mpatches.Patch(color="#c03a2b", label="Significant")
    gray_patch = mpatches.Patch(color="#7f8c8d", label="Not Significant")
    plt.legend(handles=[red_patch, gray_patch], loc="best")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path / f"enrichment_barplot.{format}", dpi=dpi, format=format)
    plt.close()

def plot_pvalue_heatmap(summary, output_path, gene_set_name=None, figsize=(8, 6), dpi=300, format="png"):
    """
    Create a heatmap of p-values for all populations.
    
    Args:
        summary: DataFrame with enrichment summary
        output_path: Path to save the plot
        gene_set_name: Name of gene set for the title
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the plot
        format: File format for saving the plot
    """
    # Extract p-values to plot
    pvals = summary[["Population", "PValue", "FDRCorrectedPValue", "EmpiricalPValue"]]
    pvals = pvals.melt(
        id_vars=["Population"],
        var_name="P-value Type",
        value_name="P-value"
    )
    
    # Replace column names for better labels
    pvals["P-value Type"] = pvals["P-value Type"].replace({
        "PValue": "Nominal",
        "FDRCorrectedPValue": "FDR Corrected",
        "EmpiricalPValue": "Empirical"
    })
    
    # Create a pivot table for the heatmap
    pivot = pvals.pivot(
        index="Population",
        columns="P-value Type",
        values="P-value"
    )
    
    # Set up color map (white to dark blue)
    cmap = LinearSegmentedColormap.from_list("", ["white", "#2c3e50"])
    
    plt.figure(figsize=figsize)
    
    # Create the heatmap
    ax = sns.heatmap(
        pivot,
        cmap=cmap,
        annot=True,
        fmt=".3f",
        linewidths=.5,
        vmin=0,
        vmax=1
    )
    
    # Customize the plot
    plt.title(f"P-values {gene_set_name or ''}", fontsize=14)
    plt.tight_layout()
    
    plt.savefig(output_path / f"pvalue_heatmap.{format}", dpi=dpi, format=format)
    plt.close()

def plot_bootstrap_distributions(results, output_path, gene_set_name=None, figsize=(12, 8), dpi=300, format="png"):
    """
    Plot bootstrap distributions for each population.
    
    Args:
        results: Dictionary with full results data
        output_path: Path to save the plot
        gene_set_name: Name of gene set for the title
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the plot
        format: File format for saving the plot
    """
    # Check if bootstrap results exist
    has_bootstrap = False
    for pop, data in results['full_results'].items():
        if 'bootstrap' in data:
            has_bootstrap = True
            break
    
    if not has_bootstrap:
        print("No bootstrap results found, skipping bootstrap distribution plot")
        return
    
    # Get all populations
    populations = list(results['full_results'].keys())
    n_pops = len(populations)
    
    # Determine grid dimensions
    n_cols = min(3, n_pops)
    n_rows = (n_pops + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_pops == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, pop in enumerate(populations):
        if i >= len(axes):
            break
            
        ax = axes[i]
        pop_data = results['full_results'][pop]
        
        if 'bootstrap' not in pop_data or not pop_data['bootstrap']:
            continue
        
        # Get enrichment ratios from bootstrap results
        ratios = [b.get('enrichment_ratio', np.nan) for b in pop_data['bootstrap']]
        ratios = [r for r in ratios if not np.isnan(r)]
        
        if not ratios:
            continue
            
        # Plot distribution
        sns.histplot(ratios, kde=True, ax=ax)
        
        # Add vertical line for observed value
        observed = pop_data['enrichment_ratio']
        ax.axvline(observed, color='red', linestyle='--', 
                   label=f'Observed: {observed:.3f}')
        
        # Add p-value
        pvalue = pop_data.get('p_value', np.nan)
        if not np.isnan(pvalue):
            ax.text(0.05, 0.95, f'p-value: {pvalue:.3f}', 
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top')
        
        ax.set_title(pop)
        ax.set_xlabel('Enrichment Ratio')
        ax.set_ylabel('Frequency')
        ax.legend()
    
    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f"Bootstrap Distributions {gene_set_name or ''}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    
    plt.savefig(output_path / f"bootstrap_distributions.{format}", dpi=dpi, format=format)
    plt.close()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    try:
        results = load_results(args.results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Extract gene set name from config if not provided
    gene_set_name = args.gene_set
    if not gene_set_name:
        config_path = Path(args.results_dir) / "data" / "pipeline_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                if 'input_files' in config and 'gene_set' in config['input_files']:
                    gene_set = config['input_files']['gene_set']
                    # Extract just the filename without path or extension
                    gene_set_name = Path(gene_set).stem
    
    # Create plots
    plot_enrichment_barplot(
        results['summary'],
        output_path,
        gene_set_name,
        dpi=args.dpi,
        format=args.save_format
    )
    
    plot_pvalue_heatmap(
        results['summary'],
        output_path,
        gene_set_name,
        dpi=args.dpi,
        format=args.save_format
    )
    
    plot_bootstrap_distributions(
        results,
        output_path,
        gene_set_name,
        dpi=args.dpi,
        format=args.save_format
    )
    
    print(f"All plots saved to {output_path}")

if __name__ == "__main__":
    main()