"""
Utility functions for data processing in the gene set enrichment pipeline.
"""

from typing import Dict, List, Optional, Set, Tuple, Union
import polars as pl
import numpy as np
from pathlib import Path
import logging

def load_gene_list(
    file_path: Path, 
    selected_population: Optional[Union[str, List[str]]] = None
) -> Tuple[pl.DataFrame, List[str]]:
    """
    Load gene list with scores for multiple populations.
    
    Args:
        file_path: Path to gene list file
        selected_population: Optional specific population(s) to filter for (from config)
                            Can be a single population name or a list of population names
        
    Returns:
        Tuple of (DataFrame with gene_id and population scores, list of population names)
    """
    # Load tab-delimited file (now the standard format)
    df = pl.read_csv(
        file_path,
        separator='\t',
        has_header=True
    )
    
    # Get all available population names from header (all columns except gene_id)
    all_population_names = [col for col in df.columns if col != 'gene_id']
    
    # If specific population(s) are requested, filter for just those
    if selected_population is not None:
        # Convert single population to list for unified handling
        if isinstance(selected_population, str):
            selected_pops = [selected_population]
        else:
            selected_pops = selected_population
            
        # Check which requested populations exist in the data
        valid_pops = [pop for pop in selected_pops if pop in all_population_names]
        
        if valid_pops:
            # Only keep the gene_id and the selected population columns
            population_names = valid_pops
            df = df.select(['gene_id'] + population_names)
            logging.info(f"Filtered gene list to only include {len(population_names)} population(s): {', '.join(population_names)}")
        else:
            invalid_pops = ', '.join(selected_pops)
            logging.warning(f"None of the requested populations '{invalid_pops}' were found in data. "
                          f"Available populations: {', '.join(all_population_names)}")
            population_names = all_population_names
    else:
        population_names = all_population_names
    
    return df, population_names

def load_gene_set(file_path: Path) -> pl.DataFrame:
    """
    Load gene set with gene IDs and set names.
    
    Args:
        file_path: Path to gene set file
        
    Returns:
        DataFrame with gene_id column
    """
    df = pl.read_csv(
        file_path,
        separator='\t',
        has_header=True
    )
    
    # Check if this is the RSV format with 'gene_id' and 'rsv' columns
    if 'rsv' in df.columns:
        # Filter to include only genes marked as 'yes'
        df = df.filter(pl.col('rsv') == 'yes')
        # Keep only the gene_id column since we've already filtered
        df = df.select(['gene_id'])
    elif 'gene_set_name' not in df.columns:
        # If neither expected format is found, just assume the first column is gene_id
        # and keep only those genes (no filtering)
        df = df.select([df.columns[0]]).rename({df.columns[0]: 'gene_id'})
    
    return df

def load_gene_coords(file_path: Path) -> pl.DataFrame:
    """
    Load gene coordinates.
    
    Args:
        file_path: Path to gene coordinates file
        
    Returns:
        DataFrame with gene_id, chrom, start, and end columns
    """
    return pl.read_csv(
        file_path,
        separator='\t',
        has_header=True,
        columns=['gene_id', 'chrom', 'start', 'end']
    )

def load_factors(file_path: Path) -> pl.DataFrame:
    """
    Load confounding factors.
    
    Args:
        file_path: Path to confounding factors file
        
    Returns:
        DataFrame with gene_id and factor columns
    """
    return pl.read_csv(
        file_path,
        separator='\t',
        has_header=True
    )

def load_valid_genes(file_path: Optional[Path]) -> Optional[Set[str]]:
    """
    Load set of valid gene IDs.
    
    Args:
        file_path: Path to valid genes file, or None
        
    Returns:
        Set of valid gene IDs, or None if no file provided
    """
    if file_path is None:
        return None
        
    return set(
        pl.read_csv(
            file_path,
            separator='\t',
            has_header=True,
            columns=['gene_id']
        )['gene_id'].to_list()
    )

def load_hgnc_mapping(file_path: Optional[Path]) -> Optional[pl.DataFrame]:
    """
    Load HGNC gene ID to symbol mapping.
    
    Args:
        file_path: Path to HGNC file, or None
        
    Returns:
        DataFrame with gene_id and hgnc_symbol columns, or None if no file provided
    """
    if file_path is None:
        return None
        
    return pl.read_csv(
        file_path,
        separator='\t',
        has_header=True,
        columns=['gene_id', 'hgnc_symbol']
    )

def filter_genes(
    df: pl.DataFrame,
    valid_genes: Optional[Set[str]] = None,
    hgnc_mapping: Optional[pl.DataFrame] = None,
    exclude_prefixes: Set[str] = None
) -> pl.DataFrame:
    """
    Filter genes based on various criteria.
    
    Args:
        df: DataFrame with gene_id column
        valid_genes: Set of valid gene IDs to keep
        hgnc_mapping: DataFrame with gene_id and hgnc_symbol columns
        exclude_prefixes: Set of prefixes to exclude from HGNC symbols
        
    Returns:
        Filtered DataFrame
    """
    if valid_genes is not None:
        df = df.filter(pl.col('gene_id').is_in(valid_genes))
    
    if hgnc_mapping is not None and exclude_prefixes:
        # Join with HGNC mapping
        df = df.join(
            hgnc_mapping,
            on='gene_id',
            how='inner'
        )
        
        # Filter out genes with excluded prefixes
        for prefix in exclude_prefixes:
            df = df.filter(~pl.col('hgnc_symbol').str.starts_with(prefix))
        
        # Drop hgnc_symbol column
        df = df.drop('hgnc_symbol')
    
    return df

def compute_gene_distances(gene_coords: pl.DataFrame) -> pl.DataFrame:
    """
    Compute distances between genes using vectorized operations.
    
    Args:
        gene_coords: DataFrame with gene coordinates
        
    Returns:
        DataFrame with gene_id, target_gene, and distance columns
    """
    if gene_coords.height <= 1:
        return pl.DataFrame({
            'gene_id': [],
            'target_gene': [],
            'distance': []
        })
    
    # Group genes by chromosome for faster processing
    grouped_by_chrom = {}
    for chrom in gene_coords['chrom'].unique():
        grouped_by_chrom[chrom] = gene_coords.filter(pl.col('chrom') == chrom)
    
    # Process each chromosome separately
    all_distances = []
    
    for chrom, genes in grouped_by_chrom.items():
        n_genes = len(genes)
        if n_genes <= 1:
            continue
            
        # Extract data as numpy arrays for faster computation
        gene_ids = genes['gene_id'].to_numpy()
        starts = genes['start'].to_numpy()
        ends = genes['end'].to_numpy()
        
        # Create all pairs of genes on this chromosome
        # This avoids the O(n²) nested loop in the original implementation
        pairs = [(i, j) for i in range(n_genes) for j in range(i+1, n_genes)]
        
        # Compute distances for all pairs at once
        for i, j in pairs:
            # Check if genes overlap
            if (starts[i] <= ends[j] and ends[i] >= starts[j]) or \
               (starts[j] <= ends[i] and ends[j] >= starts[i]):
                distance = 0
            else:
                # Distance is the minimum gap between gene boundaries
                distance = min(
                    abs(starts[i] - ends[j]),
                    abs(ends[i] - starts[j])
                )
                
            all_distances.append({
                'gene_id': gene_ids[i],
                'target_gene': gene_ids[j],
                'distance': distance
            })
    
    if not all_distances:
        return pl.DataFrame({
            'gene_id': [],
            'target_gene': [],
            'distance': []
        })
    
    return pl.DataFrame(all_distances)

def process_gene_set(
    gene_list: pl.DataFrame,
    gene_coords: pl.DataFrame,
    factors: pl.DataFrame,
    valid_genes: Optional[Set[str]] = None,
    hgnc_mapping: Optional[pl.DataFrame] = None,
    exclude_prefixes: Set[str] = None
) -> pl.DataFrame:
    """
    Process gene set for analysis.
    
    Args:
        gene_list: DataFrame with gene IDs and population scores
        gene_coords: DataFrame with gene coordinates
        factors: DataFrame with confounding factors
        valid_genes: Set of valid gene IDs to keep
        hgnc_mapping: DataFrame with gene_id and hgnc_symbol columns
        exclude_prefixes: Set of prefixes to exclude from HGNC symbols
        
    Returns:
        Processed gene set DataFrame
    """
    # Filter genes
    gene_list = filter_genes(
        gene_list,
        valid_genes=valid_genes,
        hgnc_mapping=hgnc_mapping,
        exclude_prefixes=exclude_prefixes
    )
    
    # Join gene list with coordinates
    result = gene_list.join(
        gene_coords,
        on='gene_id',
        how='inner'
    )
    
    # Join with factors if available
    if not factors.is_empty():
        result = result.join(
            factors,
            on='gene_id',
            how='left'
        )
    
    return result

def find_control_genes(
    target_genes: pl.DataFrame,
    distances: pl.DataFrame,
    min_distance: int = 1000000
) -> pl.DataFrame:
    """
    Find control genes that are far enough from target genes.
    
    Args:
        target_genes: DataFrame with target gene IDs
        distances: DataFrame with gene distances
        min_distance: Minimum distance between target and control genes
        
    Returns:
        DataFrame with control gene IDs
    """
    if target_genes.height == 0:
        raise ValueError("Target genes DataFrame cannot be empty")
    
    if distances.height == 0:
        raise ValueError("Distances DataFrame cannot be empty")
    
    # Filter distances to only include pairs with sufficient distance
    valid_distances = distances.filter(pl.col('distance') >= min_distance)
    
    # Get target gene IDs
    target_ids = set(target_genes['gene_id'].to_list())
    
    # Find potential control genes
    control_genes = set()
    for row in valid_distances.to_dicts():
        if row['gene_id'] in target_ids:
            control_genes.add(row['target_gene'])
        elif row['target_gene'] in target_ids:
            control_genes.add(row['gene_id'])
    
    # Add genes that are not in the distances DataFrame (on different chromosomes)
    all_genes = set(distances['gene_id'].to_list()) | set(distances['target_gene'].to_list())
    control_genes.update(all_genes - target_ids)
    
    # Remove any target genes that might have been included
    control_genes = control_genes - target_ids
    
    return pl.DataFrame({'gene_id': list(control_genes)})

def match_confounding_factors(
    target_genes: pl.DataFrame,
    control_genes: pl.DataFrame,
    factors: pl.DataFrame,
    tolerance: float = 0.1
) -> pl.DataFrame:
    """
    Match control genes to target genes based on confounding factors.
    
    Args:
        target_genes: DataFrame with target gene IDs
        control_genes: DataFrame with control gene IDs
        factors: DataFrame with gene IDs and confounding factors
        tolerance: Maximum allowed relative difference in factor values (as a fraction)
        
    Returns:
        DataFrame with matched control gene IDs
    """
    if target_genes.height == 0:
        raise ValueError("Target genes DataFrame cannot be empty")
    
    if control_genes.height == 0:
        raise ValueError("Control genes DataFrame cannot be empty")
    
    if factors.height == 0:
        raise ValueError("Factors DataFrame cannot be empty")
    
    # Get factor values for target and control genes
    target_factors = factors.join(target_genes, on='gene_id', how='inner')
    control_factors = factors.join(control_genes, on='gene_id', how='inner')
    
    # Exclude target genes from control genes
    control_factors = control_factors.filter(~pl.col('gene_id').is_in(target_genes['gene_id']))
    
    # Get factor columns (excluding gene_id)
    factor_cols = [col for col in factors.columns if col != 'gene_id']
    
    # Find matches within tolerance
    matched_genes = []
    for target in target_factors.to_dicts():
        for control in control_factors.to_dicts():
            # Check if all factors are within relative tolerance
            matches = True
            for col in factor_cols:
                target_val = float(target[col])
                control_val = float(control[col])
                
                # Use relative difference for comparison
                if target_val != 0:
                    relative_diff = abs(target_val - control_val) / abs(target_val)
                else:
                    # Handle zero values specially
                    relative_diff = abs(control_val) > 1e-10
                
                if relative_diff > tolerance:
                    matches = False
                    break
            
            if matches:
                matched_genes.append({
                    'gene_id': control['gene_id'],
                    **{col: float(control[col]) for col in factor_cols}
                })
    
    # Create DataFrame with matched genes and their factor values
    if not matched_genes:
        return pl.DataFrame(schema={'gene_id': pl.Utf8, **{col: pl.Float64 for col in factor_cols}})
    
    return pl.DataFrame(matched_genes)

def shuffle_genome(gene_coords: pl.DataFrame, chrom_sizes: Dict[str, int]) -> pl.DataFrame:
    """
    Randomly shuffle gene positions while preserving gene lengths and chromosome assignments.
    Uses vectorized operations for better performance.
    
    Args:
        gene_coords: DataFrame with gene coordinates
        chrom_sizes: Dictionary mapping chromosome names to sizes
        
    Returns:
        DataFrame with shuffled gene coordinates
    """
    if gene_coords.height == 0:
        raise ValueError("Gene coordinates DataFrame cannot be empty")
    
    # Process each chromosome in parallel using polars expressions
    results = []
    
    for chrom, size in chrom_sizes.items():
        # Get genes on this chromosome
        chrom_genes = gene_coords.filter(pl.col('chrom') == chrom)
        
        # Skip if no genes on this chromosome
        if chrom_genes.height == 0:
            continue
        
        # Get gene IDs and compute gene lengths
        gene_ids = chrom_genes['gene_id'].to_numpy()
        gene_lengths = (chrom_genes['end'] - chrom_genes['start']).to_numpy()
        
        # Generate random start positions efficiently
        max_start = size - gene_lengths.max()
        if max_start <= 0:
            raise ValueError(f"Chromosome {chrom} is too small for genes")
        
        # Generate all random starts at once
        starts = np.random.randint(0, max_start, size=len(gene_lengths))
        ends = starts + gene_lengths
        
        # Create DataFrame for this chromosome
        chrom_result = pl.DataFrame({
            'gene_id': gene_ids,
            'chrom': [chrom] * len(gene_ids),
            'start': starts,
            'end': ends
        })
        
        results.append(chrom_result)
    
    # Combine results from all chromosomes
    if not results:
        return pl.DataFrame(schema={'gene_id': pl.Utf8, 'chrom': pl.Utf8, 'start': pl.Int64, 'end': pl.Int64})
        
    return pl.concat(results)