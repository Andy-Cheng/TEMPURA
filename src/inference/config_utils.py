import json
import os
from argparse import ArgumentParser, Namespace

def load_config(config_path):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

def setup_parser_with_config_support(parser):
    """
    Add configuration file support to an existing argument parser.
    
    Args:
        parser: An ArgumentParser instance
        
    Returns:
        The modified ArgumentParser
    """
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    return parser

def parse_args_with_config(parser):
    """
    Parse command-line arguments with configuration file support.
    
    This function first checks for a config file option and loads parameters from it.
    Then it applies command-line arguments, which take precedence over config file values.
    
    Args:
        parser: An ArgumentParser instance
        
    Returns:
        Namespace object with all arguments
    """
    # Parse just the config argument first
    args, remaining_argv = parser.parse_known_args()
    
    # Start with default values
    config_values = {}
    
    # If config file is provided, load it
    if args.config and os.path.exists(args.config):
        config_values = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
        
        # Set defaults from config file
        parser.set_defaults(**config_values)
    
    # Parse remaining arguments (these will override config file values)
    final_args = parser.parse_args(remaining_argv)
    
    return final_args

def save_config(args, output_path):
    """
    Save the current arguments to a JSON configuration file.
    
    Args:
        args: Namespace object with arguments
        output_path: Path where to save the configuration file
    """
    # Convert Namespace to dictionary
    config = vars(args)
    
    # Remove any callable objects that can't be serialized
    config = {k: v for k, v in config.items() if not callable(v)}
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {output_path}") 