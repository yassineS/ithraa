"""Configuration handling for the gene set enrichment pipeline."""

import os
import tomli
import tomli_w  # Added import for writing TOML files
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union


class PipelineConfig:
    """Configuration class for the gene set enrichment pipeline."""

    def __init__(self, config_path: str):
        """Initialise the configuration from a TOML file.

        Args:
            config_path: Path to the TOML configuration file
        """
        self.config_path = config_path
        
        # Load configuration file
        try:
            with open(config_path, "rb") as f:
                self.config = tomli.load(f)
        except FileNotFoundError:
            raise ValueError(f"Error loading configuration file: {config_path} does not exist")
        except Exception as e:
            raise ValueError(f"Error loading configuration file: {str(e)}")
        
        # Validate required sections
        required_sections = ['input', 'output', 'analysis']
        missing_sections = [section for section in required_sections if section not in self.config]
        if missing_sections:
            raise ValueError(f"Missing required sections in configuration: {', '.join(missing_sections)}")
        
        # Extract input file paths
        self.input_files = self.config.get("input", {})
        
        # Validate required input files
        required_input_files = ['gene_list_file', 'gene_coords_file', 'factors_file']
        missing_files = [file for file in required_input_files if file not in self.input_files]
        if missing_files:
            raise ValueError(f"Missing required input files in configuration: {', '.join(missing_files)}")
        
        # Extract output configuration
        self.output_config = self.config.get("output", {})
        
        # Extract analysis parameters
        self.analysis_params = self.config.get("analysis", {})
        
        # Extract threshold parameters for rank-based analysis
        self.thresholds = self.config.get("thresholds", {})
        self.rank_thresholds = self.thresholds.get("rank_values", [5000, 4000, 3000, 2500, 2000, 
                                                               1500, 1000, 900, 800, 700, 
                                                               600, 500, 450, 400, 350, 
                                                               300, 250, 200, 150, 100,
                                                               90, 80, 70, 60, 50,
                                                               40, 30, 25, 20, 15, 10])
        
        # Extract gene prefixes to exclude
        self.exclude_prefixes = set(self.config.get("exclude_prefixes", []))
        
        # Extract number of threads from analysis section, default to 1 if not specified
        self.num_threads = self.analysis_params.get("num_threads", 1)
                                        
        # Extract population settings
        self.population_config = self.config.get("population", {})
        
        # Extract selected populations - support both single population and list of populations
        single_pop = self.population_config.get("population", None)
        multi_pop = self.population_config.get("populations", None)
        
        # Determine which populations to use (give priority to the list if both are specified)
        if multi_pop is not None:
            if isinstance(multi_pop, list):
                self.selected_populations = multi_pop
            else:
                self.selected_populations = [multi_pop]  # Convert single value to list
        elif single_pop is not None:
            self.selected_populations = [single_pop]
        else:
            self.selected_populations = None
            
        # For backward compatibility, maintain the selected_population attribute
        self.selected_population = self.selected_populations[0] if self.selected_populations else None
    
    def get_rank_thresholds(self) -> List[int]:
        """Get the rank thresholds for the analysis.
        
        Returns:
            List of rank threshold values in descending order
        """
        return sorted(self.rank_thresholds, reverse=True)
    
    def get_output_path(self, subdir: Optional[str] = None) -> Path:
        """Get the path to the output directory or a subdirectory within it.
        
        Args:
            subdir: Optional subdirectory name within the output directory
            
        Returns:
            Path object for the requested directory
        """
        # Get the output directory from the output configuration
        output_dir = self.output_config.get("output_dir", self.output_config.get("directory", "results"))
        base_path = Path(output_dir)
        
        # If a subdirectory is specified, append it to the base path
        if subdir:
            return base_path / subdir
        
        return base_path
        
    def save_config(self, output_path: Union[str, Path]) -> None:
        """Save the configuration to a TOML file.
        
        Args:
            output_path: Path to save the configuration file
        """
        with open(output_path, "wb") as f:
            tomli_w.dump(self.config, f)