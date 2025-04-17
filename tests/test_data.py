"""Tests for data processing functions."""

import pytest
from pathlib import Path
import tempfile
import os
import shutil
import polars as pl
import numba as nb
import numba.np.unsafe.ndarray as np  # Use Numba's NumPy API

from gse_pipeline.data import (
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

# Setup common test data
@pytest.fixture
def gene_list_file():
    """Create a temporary gene list file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("gene_id\tpop1\tpop2\n")
        f.write("gene1\t1.0\t0.5\n")
        f.write("gene2\t0.8\t0.6\n")
        f.write("gene3\t0.6\t0.7\n")
        f.write("gene4\t0.4\t0.8\n")
        f.write("gene5\t0.2\t0.9\n")
    
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def gene_set_file():
    """Create a temporary gene set file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("gene_id\n")
        f.write("gene1\n")
        f.write("gene3\n")
        f.write("gene5\n")
    
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def gene_coords_file():
    """Create a temporary gene coordinates file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("gene_id\tchrom\tstart\tend\n")
        f.write("gene1\tchr1\t100\t200\n")
        f.write("gene2\tchr1\t300\t400\n")
        f.write("gene3\tchr1\t1500\t1600\n")
        f.write("gene4\tchr2\t100\t200\n")
        f.write("gene5\tchr2\t300\t400\n")
    
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def factors_file():
    """Create a temporary confounding factors file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("gene_id\tfactor1\tfactor2\n")
        f.write("gene1\t0.1\t0.2\n")
        f.write("gene2\t0.3\t0.4\n")
        f.write("gene3\t0.5\t0.6\n")
        f.write("gene4\t0.7\t0.8\n")
        f.write("gene5\t0.9\t1.0\n")
    
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def valid_genes_file():
    """Create a temporary valid genes file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("gene_id\n")
        f.write("gene1\n")
        f.write("gene2\n")
        f.write("gene3\n")
        f.write("gene4\n")
        f.write("gene5\n")
    
    yield Path(f.name)
    os.unlink(f.name)

@pytest.fixture
def hgnc_file():
    """Create a temporary HGNC mapping file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("gene_id\thgnc_symbol\n")
        f.write("gene1\tGENE1\n")
        f.write("gene2\tLINC_GENE2\n")
        f.write("gene3\tGENE3\n")
        f.write("gene4\tMIR_GENE4\n")
        f.write("gene5\tGENE5\n")
    
    yield Path(f.name)
    os.unlink(f.name)

def test_load_gene_list(gene_list_file):
    """Test loading gene list."""
    # Test without population filter
    df, populations = load_gene_list(gene_list_file)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 5
    assert len(df.columns) == 3  # gene_id, pop1, pop2
    assert populations == ['pop1', 'pop2']
    
    # Test with single population filter
    df, populations = load_gene_list(gene_list_file, selected_population='pop1')
    assert len(df.columns) == 2  # gene_id, pop1
    assert populations == ['pop1']
    
    # Test with multiple population filter
    df, populations = load_gene_list(gene_list_file, selected_population=['pop1', 'pop2'])
    assert len(df.columns) == 3  # gene_id, pop1, pop2
    assert populations == ['pop1', 'pop2']
    
    # Test with invalid population filter - should return the first population
    df, populations = load_gene_list(gene_list_file, selected_population='invalid')
    assert len(df.columns) == 2  # gene_id, pop1 (since it's the first available)
    assert populations == ['pop1']

def test_load_gene_set(gene_set_file):
    """Test loading gene set."""
    df = load_gene_set(gene_set_file)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 3
    assert 'gene_id' in df.columns
    assert set(df['gene_id'].to_list()) == {'gene1', 'gene3', 'gene5'}

def test_load_gene_coords(gene_coords_file):
    """Test loading gene coordinates."""
    df = load_gene_coords(gene_coords_file)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 5
    assert set(df.columns) == {'gene_id', 'chrom', 'start', 'end'}
    assert set(df['chrom'].to_list()) == {'chr1', 'chr2'}

def test_load_factors(factors_file):
    """Test loading confounding factors."""
    df = load_factors(factors_file)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 5
    assert set(df.columns) == {'gene_id', 'factor1', 'factor2'}
    assert df['factor1'].to_list() == [0.1, 0.3, 0.5, 0.7, 0.9]

def test_load_valid_genes(valid_genes_file):
    """Test loading valid genes."""
    valid_genes = load_valid_genes(valid_genes_file)
    assert isinstance(valid_genes, set)
    assert len(valid_genes) == 5
    assert valid_genes == {'gene1', 'gene2', 'gene3', 'gene4', 'gene5'}
    
    # Test with None
    assert load_valid_genes(None) is None

def test_load_hgnc_mapping(hgnc_file):
    """Test loading HGNC mapping."""
    df = load_hgnc_mapping(hgnc_file)
    assert isinstance(df, pl.DataFrame)
    assert len(df) == 5
    assert set(df.columns) == {'gene_id', 'hgnc_symbol'}
    
    # Test with None
    assert load_hgnc_mapping(None) is None

def test_filter_genes(gene_list_file, hgnc_file):
    """Test filtering genes."""
    gene_list = load_gene_list(gene_list_file)[0]
    hgnc_mapping = load_hgnc_mapping(hgnc_file)
    
    # Test with valid genes filter
    valid_genes = {'gene1', 'gene3', 'gene5'}
    filtered = filter_genes(gene_list, valid_genes=valid_genes)
    assert len(filtered) == 3
    assert set(filtered['gene_id'].to_list()) == valid_genes
    
    # Test with HGNC prefix filter
    exclude_prefixes = {'LINC_', 'MIR_'}
    filtered = filter_genes(gene_list, hgnc_mapping=hgnc_mapping, exclude_prefixes=exclude_prefixes)
    assert len(filtered) == 3
    assert set(filtered['gene_id'].to_list()) == {'gene1', 'gene3', 'gene5'}

def test_compute_gene_distances(gene_coords_file):
    """Test computing gene distances."""
    gene_coords = load_gene_coords(gene_coords_file)
    distances = compute_gene_distances(gene_coords)
    
    assert isinstance(distances, pl.DataFrame)
    assert set(distances.columns) == {'gene_id', 'target_gene', 'distance'}
    
    # Check distances for chr1 genes
    # gene1 to gene2: 300 - 200 = 100
    # gene1 to gene3: 1500 - 200 = 1300
    # gene2 to gene3: 1500 - 400 = 1100
    # Check that these distances are present in the results
    distances_dict = {(row['gene_id'], row['target_gene']): row['distance'] for row in distances.to_dicts()}
    
    assert distances_dict.get(('gene1', 'gene2'), None) == 100
    assert distances_dict.get(('gene1', 'gene3'), None) == 1300
    assert distances_dict.get(('gene2', 'gene3'), None) == 1100
    
    # Test with empty input
    empty_coords = pl.DataFrame(schema={'gene_id': pl.Utf8, 'chrom': pl.Utf8, 'start': pl.Int64, 'end': pl.Int64})
    empty_distances = compute_gene_distances(empty_coords)
    assert len(empty_distances) == 0
    assert set(empty_distances.columns) == {'gene_id', 'target_gene', 'distance'}

def test_process_gene_set(gene_list_file, gene_coords_file, factors_file, valid_genes_file, hgnc_file):
    """Test processing gene set."""
    gene_list = load_gene_list(gene_list_file)[0]
    gene_coords = load_gene_coords(gene_coords_file)
    factors = load_factors(factors_file)
    valid_genes = load_valid_genes(valid_genes_file)
    hgnc_mapping = load_hgnc_mapping(hgnc_file)
    
    # Test basic processing
    processed = process_gene_set(
        gene_list,
        gene_coords,
        factors,
        valid_genes=valid_genes,
        hgnc_mapping=hgnc_mapping,
        exclude_prefixes={'LINC_', 'MIR_'}
    )
    
    assert isinstance(processed, pl.DataFrame)
    # Should exclude gene2 (LINC_) and gene4 (MIR_)
    assert len(processed) == 3
    assert set(processed['gene_id'].to_list()) == {'gene1', 'gene3', 'gene5'}
    assert 'pop1' in processed.columns
    assert 'pop2' in processed.columns
    assert 'factor1' in processed.columns
    assert 'factor2' in processed.columns

def test_find_control_genes(gene_set_file, gene_coords_file):
    """Test finding control genes."""
    gene_set = load_gene_set(gene_set_file)
    gene_coords = load_gene_coords(gene_coords_file)
    
    # Compute distances
    distances = compute_gene_distances(gene_coords)
    
    # Find control genes with a minimum distance of 200bp
    controls = find_control_genes(gene_set, distances, min_distance=200)
    
    assert isinstance(controls, pl.DataFrame)
    assert len(controls) == 2  # gene2 and gene4 should be in the control set
    assert set(controls['gene_id'].to_list()) == {'gene2', 'gene4'}
    
    # Test with empty inputs
    empty_genes = pl.DataFrame(schema={'gene_id': pl.Utf8})
    empty_distances = pl.DataFrame(schema={'gene_id': pl.Utf8, 'target_gene': pl.Utf8, 'distance': pl.Int64})
    
    with pytest.raises(ValueError, match="Target genes DataFrame cannot be empty"):
        find_control_genes(empty_genes, distances)
    
    with pytest.raises(ValueError, match="Distances DataFrame cannot be empty"):
        find_control_genes(gene_set, empty_distances)

def test_match_confounding_factors(gene_set_file, gene_coords_file, factors_file):
    """Test matching confounding factors."""
    gene_set = load_gene_set(gene_set_file)
    gene_coords = load_gene_coords(gene_coords_file)
    factors = load_factors(factors_file)
    
    # Compute distances
    distances = compute_gene_distances(gene_coords)
    
    # Find control genes
    controls = find_control_genes(gene_set, distances, min_distance=200)
    
    # Match confounding factors with a tolerance of 0.5
    matched = match_confounding_factors(gene_set, controls, factors, tolerance=0.5)
    
    assert isinstance(matched, pl.DataFrame)
    assert 'gene_id' in matched.columns
    assert 'factor1' in matched.columns
    assert 'factor2' in matched.columns
    
    # Test with empty inputs
    empty_genes = pl.DataFrame(schema={'gene_id': pl.Utf8})
    empty_factors = pl.DataFrame(schema={'gene_id': pl.Utf8, 'factor1': pl.Float64})
    
    with pytest.raises(ValueError, match="Target genes DataFrame cannot be empty"):
        match_confounding_factors(empty_genes, controls, factors)
    
    with pytest.raises(ValueError, match="Control genes DataFrame cannot be empty"):
        match_confounding_factors(gene_set, empty_genes, factors)
    
    with pytest.raises(ValueError, match="Factors DataFrame cannot be empty"):
        match_confounding_factors(gene_set, controls, empty_factors)

def test_shuffle_genome(gene_coords_file):
    """Test shuffling genome."""
    gene_coords = load_gene_coords(gene_coords_file)
    
    # Define chromosome sizes
    chrom_sizes = {'chr1': 2000, 'chr2': 2000}
    
    # Shuffle genome
    shuffled = shuffle_genome(gene_coords, chrom_sizes)
    
    assert isinstance(shuffled, pl.DataFrame)
    assert len(shuffled) == len(gene_coords)
    assert set(shuffled.columns) == {'gene_id', 'chrom', 'start', 'end'}
    
    # Ensure gene lengths are preserved
    original_lengths = gene_coords.select(
        (pl.col('end') - pl.col('start')).alias('length')
    )['length'].to_list()
    
    shuffled_lengths = shuffled.select(
        (pl.col('end') - pl.col('start')).alias('length')
    )['length'].to_list()
    
    assert original_lengths == shuffled_lengths
    
    # Test with empty input
    empty_coords = pl.DataFrame(schema={'gene_id': pl.Utf8, 'chrom': pl.Utf8, 'start': pl.Int64, 'end': pl.Int64})
    with pytest.raises(ValueError, match="Gene coordinates DataFrame cannot be empty"):
        shuffle_genome(empty_coords, chrom_sizes)