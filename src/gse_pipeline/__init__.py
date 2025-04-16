"""
Gene Set Enrichment Pipeline
===========================

A Python package for performing gene set enrichment analysis.
"""

from .pipeline import GeneSetEnrichmentPipeline
from .config import PipelineConfig
from .data import (
    load_gene_list,
    load_gene_set,
    load_gene_coords,
    load_factors,
    load_valid_genes,
)
from .stats import (
    calculate_enrichment,
    perform_fdr_analysis,
    bootstrap_analysis,
    control_for_confounders,
    compute_enrichment_score,
    calculate_significance,
)
from .utils import setup_logging, ensure_dir

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