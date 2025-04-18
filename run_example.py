import logging
import os
import multiprocessing
import time
from pathlib import Path
from ithraa import GeneSetEnrichmentPipeline
from ithraa.config import PipelineConfig

def run_pipeline():
    # Configure logging to file only for debug messages, console for info and above
    # This keeps the progress bar clean by not printing debug messages to console
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"pipeline_{time.strftime('%Y%m%d-%H%M%S')}.log")
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only INFO and above go to console
    
    # Format for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get the root logger and configure it
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all messages at the root
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Add the handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    config_path = Path("example/config.toml").absolute()
    print(f"Using config file: {config_path}")

    # Check if the config file exists
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        # Load the config first to debug
        config = PipelineConfig(str(config_path))
        
        # Debug: Print the input files configuration
        print("Input files configuration:")
        for key, value in config.input_files.items():
            print(f"  - {key}: {type(value)} = {value}")
        
        # Filter out non-file path items from the configuration
        # This fixes the issue with special_filtered_genes being a list
        input_files_copy = dict(config.config['input'])
        for key, value in list(input_files_copy.items()):
            if not isinstance(value, (str, bytes, os.PathLike)):
                print(f"Removing non-path item from input_files: {key}")
                special_value = input_files_copy.pop(key)
                # Store it elsewhere in the config
                if key == 'special_filtered_genes':
                    if 'gene_filtering' not in config.config['analysis']:
                        config.config['analysis']['gene_filtering'] = {}
                    config.config['analysis']['gene_filtering']['exclude_prefixes'] = special_value
        
        # Update the config with the filtered input files
        config.config['input'] = input_files_copy
        
        # Save modified config to a temporary file
        temp_config_path = config_path.parent / "temp_config.toml"
        config.save_config(temp_config_path)
        print(f"Saved filtered config to: {temp_config_path}")
        
        # Now run the pipeline with the modified config
        print("Initializing pipeline with filtered configuration...")
        pipeline = GeneSetEnrichmentPipeline(str(temp_config_path))
        
        # Run the pipeline
        print("Running pipeline...")
        pipeline.run()
        print("Pipeline execution completed successfully!")
        
        # Clean up temporary config file
        os.remove(temp_config_path)
    except Exception as e:
        logging.exception("Error running the pipeline")
        raise

if __name__ == "__main__":
    # This is required on macOS for multiprocessing to work properly
    multiprocessing.freeze_support()
    run_pipeline()
