"""
Gene Set Enrichment Pipeline
===========================

A Python package for performing gene set enrichment analysis.
"""

from .pipeline import GeneSetEnrichmentPipeline
from .config import PipelineConfig
from .data import (
    load_gene_list as load_gene_list,
    load_gene_set as load_gene_set,
    load_gene_coords as load_gene_coords,
    load_factors as load_factors,
    load_valid_genes as load_valid_genes,
)
from .stats import (
    calculate_enrichment as calculate_enrichment,
    perform_fdr_analysis as perform_fdr_analysis,
    bootstrap_analysis as bootstrap_analysis,
    control_for_confounders as control_for_confounders,
    compute_enrichment_score as compute_enrichment_score,
    calculate_significance as calculate_significance,
)
from .utils import setup_logging as setup_logging, ensure_dir as ensure_dir

__version__ = "0.1.0"

__all__ = [
    "GeneSetEnrichmentPipeline",
    "PipelineConfig",
    "load_gene_list",
    "load_gene_set",
    "load_gene_coords",
    "load_factors",
    "load_valid_genes",
    "calculate_enrichment",
    "perform_fdr_analysis",
    "bootstrap_analysis",
    "control_for_confounders",
    "compute_enrichment_score",
    "calculate_significance",
    "setup_logging",
    "ensure_dir",
] 