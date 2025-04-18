# Ithraa: A Gene Set Enrichment Pipeline
A gene set enrichment (GSE) pipeline, providing tools for gene set enrichment analysis, bootstrap testing, and control for confounding factors when gene matching to perform False Discovery Rate (FDR) estimation.

This is largely based on David Enard's pipeline: [https://github.com/DavidPierreEnard/Gene_Set_Enrichment_Pipeline/](https://github.com/DavidPierreEnard/Gene_Set_Enrichment_Pipeline/)

The name إثراء (ʾithrāʾ) means enrichment in arabic (inspired by tqdm).

## Features

- Gene set enrichment analysis with flexible input formats
- Bootstrap testing for statistical significance
- Control for confounding factors during FDR estimation using gene matching
- Modern Python implementation with type hints and comprehensive documentation
- Extensive test coverage via pytest
- Flexible column name handling
- Support for various gene ID formats (requires mapping file for non-HGNC IDs)
- Performance optimizations using Polars, NumPy, and Numba
- Parallel processing support

## Installation

We recommend using `uv` for faster dependency management.

```sh
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yourusername/gse_pipeline.git
cd gse_pipeline

# Create a virtual environment and install the package
uv venv
uv pip install -e .
```

### Development Installation

```bash
# Using the installation script (recommended)
python install.py --dev

# Using uv
uv pip install -e ".[dev]"

# Using pip
pip install -e ".[dev]"
```

## Usage

```python
from gse_pipeline import GeneSetEnrichmentPipeline

# Initialize the pipeline
pipeline = GeneSetEnrichmentPipeline(
    pipeline_dir="path/to/output",
    valid_genes_file="path/to/valid_genes.txt",
    gene_set_file="path/to/gene_set.txt",
    distance_file="path/to/distances.txt",
    factors_table="path/to/factors.txt",
    hgnc_file="path/to/hgnc_mapping.txt",
    gene_id_col="custom_gene_id",  # Custom column names
    distance_col="gene_distance",
    score_col="sweep_score"
)

# Run bootstrap analysis
pipeline.run_bootstrap(
    sweep_files=["path/to/sweep1.txt", "path/to/sweep2.txt"],
    iterations=1000
)

# Run FDR analysis
pipeline.run_fdr_analysis(fdr_number=1000)

# Control for confounding factors
pipeline.control_confounding_factors()
```

## Input File Formats

### Gene List with Scores
A tab-separated file containing gene IDs and their associated scores. The file must have a header row with column names.

Required columns:
- `gene_id`: Unique identifier for each gene
- `score`: Numeric value (float) associated with the gene

Example:
```
gene_id    score
GENE1      0.5
GENE2      0.8
```

### Gene Coordinates
A tab-separated file containing genomic coordinates for genes. This should be a superset of the genes in the gene list with scores. The file must have a header row with column names.

Required columns:
- `gene_id`: Unique identifier for each gene (must match gene_id in gene list)
- `chrom` or `chr` or `chromosome`: Chromosome identifier (string)
- `start`: Start position (positive integer)
- `end`: End position (positive integer)

Example:
```
gene_id    chrom    start    end
GENE1      chr1     1000     2000
GENE2      chr2     3000     4000
```

### Confounding Factors
A tab-separated file containing gene IDs and their associated confounding factors. The file must have a header row with column names.

Required columns:
- `gene_id`: Unique identifier for each gene (must match gene_id in gene list)
- Additional columns: The following confounding factors are recommended to be controlled for in the analysis:

1. **GTEx Expression**
   - `gtex_avg`: Average expression across 53 GTEx V8 tissues
   - `gtex_lymphocytes`: Expression in lymphocytes
   - `gtex_testis`: Expression in testis

2. **Protein Interactions**
   - `ppi_count`: Number of protein-protein interactions from the IntAct database

3. **Genomic Features**
   - `coding_density`: Ensembl (v83) coding sequence density in a 50kb window
   - `phastcons_density`: Density of conserved segments from PhastCons
   - `regulatory_density`: Density of ENCODE DNase I V3 Clusters in a 50kb window
   - `recombination_rate`: Recombination rate in a 200kb window
   - `gc_content`: GC content in a 50kb window

4. **Pathogen Interactions**
   - `bacteria_interactions`: Number of bacteria each gene interacts with (from IntAct database)

5. **Immune Function**
   - `immune_proportion`: Proportion of genes that are immune genes based on Gene Ontology annotations:
     - GO:0006952 (defense response)
     - GO:0006955 (immune response)
     - GO:0002376 (immune system process)

Example:
```
gene_id    gtex_avg    gtex_lymphocytes    gtex_testis    ppi_count    coding_density    phastcons_density    regulatory_density    recombination_rate    gc_content    bacteria_interactions    immune_proportion
GENE1      0.5         0.8                 0.3            15           0.45              0.67                 0.23                   0.0012              0.42          3                       0.1
GENE2      0.3         0.6                 0.4            8            0.38              0.59                 0.19                   0.0009              0.38          1                       0.0
```

Note: The distance file is computed automatically from the gene coordinates file and does not need to be provided separately. Control genes are selected at a minimum distance of 500kb from target genes to avoid overlapping sweeps.

### Gene Sets
A tab-separated file containing gene IDs and their classification. The file should have a header row with column names.

Example:
```
gene_id    is_target
GENE1            yes
GENE2            no
```

## Comparison with Original Perl Implementation

### Advantages of Python Implementation

1. **Modern Development Practices**
   - Type hints for better code maintainability
   - Comprehensive test suite
   - Modern packaging with pyproject.toml
   - Proper dependency management

2. **Flexibility**
   - Support for custom column names
   - No dependency on specific gene ID formats (e.g., ENSG)
   - More flexible input file handling

3. **Performance**
   - Vectorized operations using NumPy and Polars
   - Parallel processing capabilities
   - Memory-efficient data structures

4. **Usability**
   - Clear API documentation
   - Better error messages
   - Progress bars for long-running operations
   - Jupyter notebook support

5. **Maintainability**
   - Modular code structure
   - Comprehensive logging
   - Type checking with mypy
   - Code formatting with black


## Documentation

Full documentation is available at [https://gse-pipeline.readthedocs.io/](https://gse-pipeline.readthedocs.io/).

## Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{gse_pipeline,
  title = {Gene Set Enrichment Pipeline},
  author = {Yassine Souilmi},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ithraa}
}
```