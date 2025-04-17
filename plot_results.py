#!/usr/bin/env python3
"""
Script to plot gene set enrichment results with threshold rank analysis.
This version matches the style of the original Gene_Set_Enrichment_Pipeline plots.
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import numba as nb
import numba.np.unsafe.ndarray as np  # Use Numba's NumPy API
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Set the style to match the original plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot gene set enrichment results from thresholded data"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory (default: results)"
    )
    
    parser.add_argument(
        "--gene-set",
        type=str,
        help="Name of the gene set to visualize (without path or extension)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Path to output directory for plots (default: plots)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Format for saving plots (default: png)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for saving plots (default: 300)"
    )
    
    return parser.parse_args()

def parse_legacy_results(results_file_path):
    """
    Parse results file from the new pipeline format to match original pipeline data.
    
    Format: threshold population enrichment_ratio observed expected ci_low ci_high p-value
    
    Returns:
        DataFrame with parsed data
    """
    data = []
    with open(results_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:  # Proper format for threshold data line
                try:
                    threshold = int(parts[0])
                    pop_name = parts[1].strip(':')
                    
                    # Skip metadata rows
                    if pop_name in ["OUT", "Group_ratio"]:
                        continue
                        
                    enrichment = float(parts[2])
                    observed = float(parts[3])
                    expected = float(parts[4])
                    ci_low = float(parts[5])
                    ci_high = float(parts[6])
                    p_value = float(parts[7]) if len(parts) > 7 else None
                    
                    data.append({
                        'Threshold': threshold,
                        'Population': pop_name,
                        'EnrichmentRatio': enrichment,
                        'Observed': observed,
                        'Expected': expected,
                        'CI_Low': ci_low,
                        'CI_High': ci_high,
                        'P_Value': p_value,
                        'Significant': (p_value is not None and p_value <= 0.05)
                    })
                except (ValueError, IndexError):
                    continue  # Skip malformed lines
    
    return pd.DataFrame(data)

def find_results_file(results_dir, gene_set=None):
    """Find and identify the results file in the results directory."""
    data_dir = Path(results_dir) / 'data'
    
    # If gene_set is provided, look for that specific file
    if gene_set:
        filename = f"{gene_set}_results.txt"
        file_path = data_dir / filename
        if file_path.exists():
            return file_path
    
    # Otherwise, find all *_results.txt files and use the first one
    result_files = list(data_dir.glob('*_results.txt'))
    if result_files:
        return result_files[0]
    
    # If no results file found, raise error
    raise FileNotFoundError(f"No results file found in {data_dir}")

def plot_threshold_enrichment(results_df, output_path, gene_set=None, figsize=(14, 8), dpi=300, format='png'):
    """
    Create a plot of enrichment ratios across different thresholds for all populations.
    
    Args:
        results_df: DataFrame with parsed results
        output_path: Path to save the plot
        gene_set: Name of the gene set for the title (optional)
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the plot
        format: Format for saving the plot (png, pdf, svg)
    """
    # Sort by threshold (descending)
    results_df = results_df.sort_values('Threshold', ascending=False)
    
    # Get populations and thresholds
    populations = sorted(results_df['Population'].unique())
    thresholds = sorted(results_df['Threshold'].unique(), reverse=True)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot enrichment ratios for each population
    for i, pop in enumerate(populations):
        pop_data = results_df[results_df['Population'] == pop]
        
        # Sort by threshold (descending)
        pop_data = pop_data.sort_values('Threshold')
        
        plt.plot(
            pop_data['Threshold'], 
            pop_data['EnrichmentRatio'], 
            marker='o', 
            linestyle='-', 
            linewidth=2,
            label=pop
        )
        
        # Mark significant points with stars
        sig_points = pop_data[pop_data['Significant']]
        if not sig_points.empty:
            plt.plot(
                sig_points['Threshold'],
                sig_points['EnrichmentRatio'],
                'k*',
                markersize=10,
            )
    
    # Add a horizontal line at y=1 (no enrichment)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
    
    plt.xscale('log')  # Log scale for the x-axis (thresholds)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Add labels and title
    plt.xlabel('Rank Threshold', fontsize=14, fontweight='bold')
    plt.ylabel('Enrichment Ratio', fontsize=14, fontweight='bold')
    
    title = "Gene Set Enrichment by Rank Threshold"
    if gene_set:
        title += f" - {gene_set}"
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.legend(title="Population", loc='best', frameon=True)
    plt.tight_layout()
    
    # Save the figure
    output_file = output_path / f"threshold_enrichment.{format}"
    plt.savefig(output_file, dpi=dpi, format=format)
    plt.close()
    
    return output_file

def plot_population_enrichment(results_df, output_path, gene_set=None, figsize=(15, 10), dpi=300, format='png'):
    """
    Create individual plots for each population showing enrichment ratio across thresholds.
    
    Args:
        results_df: DataFrame with parsed results
        output_path: Path to save the plot
        gene_set: Name of the gene set for the title (optional)
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the plot
        format: Format for saving the plot (png, pdf, svg)
    """
    # Get populations
    populations = sorted(results_df['Population'].unique())
    n_pops = len(populations)
    
    # Determine grid dimensions
    n_cols = min(3, n_pops)
    n_rows = (n_pops + n_cols - 1) // n_cols  # Ceiling division
    
    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    
    # Handle case when there is only one population (resulting in a single subplot)
    if n_pops == 1:
        axes = [axes]
    else:
        # Convert to 1D array for easy indexing
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    # Sort by threshold (ascending order for plotting)
    results_df = results_df.sort_values('Threshold')
    
    # Plot each population in its own subplot
    for i, pop in enumerate(populations):
        if i >= len(axes):
            break
            
        ax = axes[i]
        pop_data = results_df[results_df['Population'] == pop]
        
        # Plot enrichment ratio
        ax.plot(
            pop_data['Threshold'], 
            pop_data['EnrichmentRatio'], 
            marker='o', 
            linestyle='-', 
            linewidth=2,
            color='#1f77b4'  # Blue color
        )
        
        # Add confidence interval range
        ax.fill_between(
            pop_data['Threshold'],
            pop_data['Expected'] - (pop_data['Expected'] - pop_data['CI_Low']),
            pop_data['Expected'] + (pop_data['CI_High'] - pop_data['Expected']),
            alpha=0.2,
            color='#1f77b4',
            label='95% CI'
        )
        
        # Add horizontal line at y=1 (no enrichment)
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Mark significant points with stars
        sig_points = pop_data[pop_data['Significant']]
        if not sig_points.empty:
            ax.plot(
                sig_points['Threshold'],
                sig_points['EnrichmentRatio'],
                'k*',
                markersize=10,
            )
        
        # Configure axis
        ax.set_xscale('log')  # Log scale for the x-axis (thresholds)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.set_title(f"{pop}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Rank Threshold', fontsize=12)
        
        if i % n_cols == 0:  # Only for leftmost plots
            ax.set_ylabel('Enrichment Ratio', fontsize=12)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Add a single title for the entire figure
    title = "Population Enrichment Patterns by Rank Threshold"
    if gene_set:
        title += f" - {gene_set}"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
    
    # Save the figure
    output_file = output_path / f"population_enrichment.{format}"
    plt.savefig(output_file, dpi=dpi, format=format)
    plt.close()
    
    return output_file

def plot_heatmap(results_df, output_path, gene_set=None, metric='EnrichmentRatio', 
                figsize=(12, 8), dpi=300, format='png'):
    """
    Create a heatmap of enrichment ratios or p-values across populations and thresholds.
    
    Args:
        results_df: DataFrame with parsed results
        output_path: Path to save the plot
        gene_set: Name of the gene set for the title (optional)
        metric: Metric to plot ('EnrichmentRatio' or 'P_Value')
        figsize: Figure size (width, height) in inches
        dpi: DPI for saving the plot
        format: Format for saving the plot (png, pdf, svg)
    """
    # Pivot the data to create a matrix with thresholds as rows and populations as columns
    pivot_df = results_df.pivot(index='Threshold', columns='Population', values=metric)
    
    # Sort thresholds in descending order
    pivot_df = pivot_df.sort_index(ascending=False)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Set up custom colormap based on the metric
    if metric == 'EnrichmentRatio':
        # Blue-white-red colormap for enrichment ratio (centered at 1.0)
        cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                               [(0, '#3b4cc0'), (0.5, 'white'), (1.0, '#b40426')])
        vmin = min(0.5, pivot_df.values.min())
        vmax = max(1.5, pivot_df.values.max())
        vcenter = 1.0
        cbar_label = "Enrichment Ratio"
        
        # For enrichment ratio, we also want to mark significant values
        sig_mask = results_df.pivot(index='Threshold', columns='Population', values='Significant')
        sig_mask = sig_mask.sort_index(ascending=False)
        
    else:  # P-value
        # White-blue colormap for p-values
        cmap = 'Blues_r'  # Reversed Blues colormap (darker = smaller p-value)
        vmin = 0
        vmax = 0.2  # Cap at 0.2 for better color resolution at significant values
        vcenter = None
        cbar_label = "P-value"
        sig_mask = None
    
    # Create the heatmap
    ax = sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap=cmap, 
                    vmin=vmin, vmax=vmax, center=vcenter,
                    cbar_kws={'label': cbar_label})
    
    # Optionally, mark significant values with a star marker
    if sig_mask is not None and metric == 'EnrichmentRatio':
        for i in range(sig_mask.shape[0]):
            for j in range(sig_mask.shape[1]):
                if sig_mask.iloc[i, j]:
                    ax.text(j + 0.5, i + 0.85, '*',
                            ha='center', va='center', color='black',
                            fontsize=15, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Population', fontsize=14, fontweight='bold')
    ax.set_ylabel('Rank Threshold', fontsize=14, fontweight='bold')
    
    title = f"{cbar_label} Heatmap by Rank Threshold"
    if gene_set:
        title += f" - {gene_set}"
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the figure
    metric_name = "enrichment" if metric == 'EnrichmentRatio' else "pvalue"
    output_file = output_path / f"{metric_name}_heatmap.{format}"
    plt.savefig(output_file, dpi=dpi, format=format)
    plt.close()
    
    return output_file

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Find results file
        results_file = find_results_file(args.results_dir, args.gene_set)
        print(f"Found results file: {results_file}")
        
        # Parse results
        results_df = parse_legacy_results(results_file)
        
        if results_df.empty:
            print("No valid data found in results file.")
            sys.exit(1)
        
        # Get gene set name from file if not provided
        gene_set_name = args.gene_set
        if not gene_set_name:
            gene_set_name = Path(results_file).stem.replace('_results', '')
        
        # Create plots
        threshold_plot = plot_threshold_enrichment(
            results_df, 
            output_path, 
            gene_set_name, 
            dpi=args.dpi, 
            format=args.format
        )
        print(f"Created threshold enrichment plot: {threshold_plot}")
        
        population_plot = plot_population_enrichment(
            results_df, 
            output_path, 
            gene_set_name, 
            dpi=args.dpi, 
            format=args.format
        )
        print(f"Created population enrichment plot: {population_plot}")
        
        # Create enrichment ratio heatmap
        enrichment_heatmap = plot_heatmap(
            results_df, 
            output_path, 
            gene_set_name, 
            metric='EnrichmentRatio', 
            dpi=args.dpi,
            format=args.format
        )
        print(f"Created enrichment ratio heatmap: {enrichment_heatmap}")
        
        # Create p-value heatmap
        pvalue_heatmap = plot_heatmap(
            results_df, 
            output_path, 
            gene_set_name, 
            metric='P_Value',
            dpi=args.dpi,
            format=args.format
        )
        print(f"Created p-value heatmap: {pvalue_heatmap}")
        
        print(f"All plots saved to {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()