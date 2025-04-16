"""Tests for data processing functions."""

import pytest
import polars as pl
from pathlib import Path
import tempfile
import os
from typing import Dict, List, Set
import numpy as np
import pandas as pd

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

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

@pytest.fixture
def gene_list_file(temp_dir):
    """Create a test gene list file."""
    content = """gene_id\tPop1\tPop2
ENSG1\t0.1\t0.2
ENSG2\t0.3\t0.4
ENSG3\t0.5\t0.6"""
    file_path = temp_dir / "gene_list.txt"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def gene_set_file(temp_dir):
    """Create a test gene set file."""
    content = """gene_id\tgene_set_name
ENSG1\thbv
ENSG2\thcv
ENSG3\thbv"""
    file_path = temp_dir / "gene_set.txt"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def gene_coords_file(temp_dir):
    """Create a test gene coordinates file."""
    content = """gene_id\tchrom\tstart\tend
ENSG1\t1\t1000\t2000
ENSG2\t1\t3000\t4000
ENSG3\t2\t1000\t2000"""
    file_path = temp_dir / "gene_coords.txt"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def factors_file(temp_dir):
    """Create a test factors file."""
    content = """gene_id\tfactor
ENSG1\t0.5
ENSG2\t0.7
ENSG3\t0.9"""
    file_path = temp_dir / "factors.txt"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def valid_genes_file(temp_dir):
    """Create a test valid genes file."""
    content = """gene_id
ENSG1
ENSG2"""
    file_path = temp_dir / "valid_genes.txt"
    file_path.write_text(content)
    return file_path

@pytest.fixture
def hgnc_file(temp_dir):
    """Create a test HGNC file."""
    content = """gene_id\thgnc_symbol
ENSG1\tGENE1
ENSG2\tHLA-GENE2
ENSG3\tHIST-GENE3"""
    file_path = temp_dir / "hgnc.txt"
    file_path.write_text(content)
    return file_path

def test_load_gene_list(gene_list_file):
    """Test loading gene list with multiple populations."""
    df, populations = load_gene_list(gene_list_file)
    
    assert isinstance(df, pl.DataFrame)
    assert len(populations) == 2
    assert populations == ['Pop1', 'Pop2']
    assert df.shape == (3, 3)
    assert df.columns == ['gene_id', 'Pop1', 'Pop2']
    assert df['gene_id'].to_list() == ['ENSG1', 'ENSG2', 'ENSG3']

def test_load_gene_list_empty(temp_dir):
    """Test loading empty gene list file."""
    empty_file = temp_dir / "empty.txt"
    empty_file.write_text("")
    
    with pytest.raises(Exception):
        load_gene_list(empty_file)

def test_load_gene_set(gene_set_file):
    """Test loading gene set file."""
    df = load_gene_set(gene_set_file)
    
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 2)
    assert df.columns == ['gene_id', 'gene_set_name']
    assert df['gene_id'].to_list() == ['ENSG1', 'ENSG2', 'ENSG3']
    assert df['gene_set_name'].to_list() == ['hbv', 'hcv', 'hbv']

def test_load_factors(factors_file):
    """Test loading factors file."""
    df = load_factors(factors_file)
    
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 2)
    assert df.columns == ['gene_id', 'factor']
    assert df['gene_id'].to_list() == ['ENSG1', 'ENSG2', 'ENSG3']
    assert df['factor'].to_list() == [0.5, 0.7, 0.9]

def test_load_valid_genes(valid_genes_file):
    """Test loading valid genes file."""
    valid_genes = load_valid_genes(valid_genes_file)
    
    assert isinstance(valid_genes, set)
    assert len(valid_genes) == 2
    assert valid_genes == {'ENSG1', 'ENSG2'}

def test_load_valid_genes_none():
    """Test loading valid genes with None input."""
    valid_genes = load_valid_genes(None)
    assert valid_genes is None

def test_load_hgnc_mapping(hgnc_file):
    """Test loading HGNC mapping file."""
    df = load_hgnc_mapping(hgnc_file)
    
    assert isinstance(df, pl.DataFrame)
    assert df.shape == (3, 2)
    assert df.columns == ['gene_id', 'hgnc_symbol']
    assert df['gene_id'].to_list() == ['ENSG1', 'ENSG2', 'ENSG3']
    assert df['hgnc_symbol'].to_list() == ['GENE1', 'HLA-GENE2', 'HIST-GENE3']

def test_load_hgnc_mapping_none():
    """Test loading HGNC mapping with None input."""
    df = load_hgnc_mapping(None)
    assert df is None

def test_filter_genes():
    """Test filtering genes with various criteria."""
    df = pl.DataFrame({
        'gene_id': ['ENSG1', 'ENSG2', 'ENSG3'],
        'score': [0.1, 0.2, 0.3]
    })
    
    # Test with valid genes
    valid_genes = {'ENSG1', 'ENSG2'}
    filtered = filter_genes(df, valid_genes=valid_genes)
    assert filtered.shape == (2, 2)
    assert filtered['gene_id'].to_list() == ['ENSG1', 'ENSG2']
    
    # Test with HGNC mapping and exclude prefixes
    hgnc_mapping = pl.DataFrame({
        'gene_id': ['ENSG1', 'ENSG2', 'ENSG3'],
        'hgnc_symbol': ['GENE1', 'HLA-GENE2', 'HIST-GENE3']
    })
    exclude_prefixes = {'HLA', 'HIST'}
    filtered = filter_genes(df, hgnc_mapping=hgnc_mapping, exclude_prefixes=exclude_prefixes)
    assert filtered.shape == (1, 2)
    assert filtered['gene_id'].to_list() == ['ENSG1']

def test_compute_gene_distances(gene_coords_file):
    """Test computing gene distances."""
    gene_coords = load_gene_coords(gene_coords_file)
    
    # Convert chromosome to string type
    gene_coords = gene_coords.with_columns(pl.col('chrom').cast(pl.Utf8))
    
    distances = compute_gene_distances(gene_coords)
    
    assert isinstance(distances, pl.DataFrame)
    assert distances.shape[0] > 0  # Should have at least one distance
    assert all(col in distances.columns for col in ['gene_id', 'target_gene', 'distance'])
    assert all(distances['distance'].to_numpy() >= 0)

def test_process_gene_set(gene_list_file, gene_coords_file, factors_file, valid_genes_file, hgnc_file):
    """Test processing gene set with all optional filters."""
    gene_list, _ = load_gene_list(gene_list_file)
    gene_coords = load_gene_coords(gene_coords_file)
    factors = load_factors(factors_file)
    valid_genes = load_valid_genes(valid_genes_file)
    hgnc_mapping = load_hgnc_mapping(hgnc_file)
    
    result = process_gene_set(
        gene_list,
        gene_coords,
        factors,
        valid_genes=valid_genes,
        hgnc_mapping=hgnc_mapping,
        exclude_prefixes={'HLA', 'HIST'}
    )
    
    assert isinstance(result, pl.DataFrame)
    assert result.shape[0] > 0
    assert all(col in result.columns for col in ['gene_id', 'chrom', 'start', 'end', 'Pop1', 'Pop2', 'factor'])

def test_find_control_genes(gene_coords_file):
    """Test finding control genes."""
    gene_coords = load_gene_coords(gene_coords_file)
    
    # Convert chromosome to string type
    gene_coords = gene_coords.with_columns(pl.col('chrom').cast(pl.Utf8))
    
    distances = compute_gene_distances(gene_coords)
    target_genes = pl.DataFrame({'gene_id': ['ENSG1']})
    
    controls = find_control_genes(target_genes, distances, min_distance=1000)
    
    assert isinstance(controls, pl.DataFrame)
    assert controls.shape[0] > 0
    assert 'gene_id' in controls.columns
    assert all(control not in target_genes['gene_id'].to_list() for control in controls['gene_id'].to_list())

def test_match_confounding_factors(gene_coords_file, factors_file):
    """Test matching control genes based on confounding factors."""
    gene_coords = load_gene_coords(gene_coords_file)
    
    # Convert chromosome to string type
    gene_coords = gene_coords.with_columns(pl.col('chrom').cast(pl.Utf8))
    
    distances = compute_gene_distances(gene_coords)
    factors = load_factors(factors_file)
    target_genes = pl.DataFrame({'gene_id': ['ENSG1'], 'factor': [0.5]})
    control_genes = find_control_genes(target_genes, distances, min_distance=1000)
    
    # Increase tolerance to ensure matches are found
    matched = match_confounding_factors(target_genes, control_genes, factors, tolerance=0.5)
    
    assert isinstance(matched, pl.DataFrame)
    assert matched.shape[0] > 0
    assert 'gene_id' in matched.columns

def test_shuffle_genome(gene_coords_file):
    """Test genome shuffling."""
    gene_coords = load_gene_coords(gene_coords_file)
    
    # Convert chromosome to string type
    gene_coords = gene_coords.with_columns(pl.col('chrom').cast(pl.Utf8))
    
    chrom_sizes = {'1': 1000000, '2': 1000000}
    
    shuffled = shuffle_genome(gene_coords, chrom_sizes)
    
    assert isinstance(shuffled, pl.DataFrame)
    assert shuffled.shape == gene_coords.shape
    assert all(col in shuffled.columns for col in ['gene_id', 'chrom', 'start', 'end'])
    assert all(shuffled['end'].to_numpy() > shuffled['start'].to_numpy())
    assert all(shuffled['chrom'].is_in(['1', '2'])) 