"""Configuration management for gene set enrichment analysis."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import tomli
from tomli_w import dump as tomli_w_dump

from .utils import setup_logging, ensure_dir

class PipelineConfig:
    """Configuration manager for gene set enrichment analysis."""
    
    def __init__(self, config_path: str):
        """Initialize configuration from a TOML file.
        
        Args:
            config_path: Path to the TOML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup()
    
    def _load_config(self) -> dict:
        """Load configuration from TOML file."""
        try:
            with open(self.config_path, 'rb') as f:
                config = tomli.load(f)
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {e}")
        
        # Validate required sections
        required_sections = ['input', 'output', 'analysis']
        missing_sections = [section for section in required_sections if section not in config]
        if missing_sections:
            raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")
        
        # Validate required input files
        required_files = ['gene_list_file', 'gene_coords_file', 'factors_file']
        missing_files = [file for file in required_files if file not in config['input']]
        if missing_files:
            raise ValueError(f"Missing required input files: {', '.join(missing_files)}")
        
        # Handle optional files without adding them to config if not present
        if 'valid_genes_file' in config['input'] and config['input']['valid_genes_file'] is None:
            del config['input']['valid_genes_file']
        if 'hgnc_file' in config['input'] and config['input']['hgnc_file'] is None:
            del config['input']['hgnc_file']
            
        # Set default values for gene filtering
        if 'gene_filtering' not in config['analysis']:
            config['analysis']['gene_filtering'] = {
                'exclude_prefixes': ['HLA', 'HIST', 'OR'],
                'require_protein_coding': False
            }
        
        return config
    
    def _setup(self):
        """Set up logging and create output directories."""
        output_dir = Path(self.output_config['directory'])
        log_dir = output_dir / 'logs'
        ensure_dir(output_dir)
        setup_logging(log_dir)
    
    @property
    def input_files(self) -> Dict[str, str]:
        """Get input file paths."""
        return self.config['input']
    
    @property
    def output_config(self) -> Dict:
        """Get output configuration."""
        return self.config.get('output', {'directory': 'results', 'save_intermediate': False})
    
    @property
    def analysis_params(self) -> Dict:
        """Get analysis parameters."""
        return self.config.get('analysis', {})
    
    @property
    def gene_filtering(self) -> Dict:
        """Get gene filtering parameters."""
        return self.analysis_params.get('gene_filtering', {})
    
    @property
    def exclude_prefixes(self) -> Set[str]:
        """Get prefixes to exclude from analysis."""
        return set(self.gene_filtering.get('exclude_prefixes', []))
    
    @property
    def require_protein_coding(self) -> bool:
        """Get whether to require protein coding genes."""
        return self.gene_filtering.get('require_protein_coding', False)
    
    @property
    def num_threads(self) -> int:
        """Get number of threads for parallel processing."""
        return self.analysis_params.get('num_threads', 4)
    
    def get_output_path(self, category: str = None) -> Path:
        """Get path for output files.
        
        Args:
            category: Optional subdirectory within output directory
            
        Returns:
            Path to output directory or subdirectory
        """
        output_dir = Path(self.output_config['directory'])
        if category:
            return ensure_dir(output_dir / category)
        return output_dir
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to a TOML file.
        
        Args:
            output_path: Optional path to save configuration. If not provided,
                        saves to the original config file path.
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'wb') as f:
            tomli_w_dump(self.config, f)
    
    @classmethod
    def load_config(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from a TOML file.
        
        Args:
            config_path: Path to the TOML configuration file
            
        Returns:
            PipelineConfig instance
        """
        return cls(config_path)

    def copy(self) -> 'PipelineConfig':
        """Create a copy of the configuration.
        
        Returns:
            A new PipelineConfig instance with the same settings
        """
        new_config = PipelineConfig(self.config_path)
        new_config.config = self.config.copy()
        return new_config 