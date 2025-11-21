"""
Configuration management for membrane kymograph generator.
"""

import os
import configparser
from typing import Dict, Any, Optional


def load_config(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load configuration from INI file.
    
    Parameters
    ----------
    filepath : str
        Path to configuration file
        
    Returns
    -------
    dict or None
        Configuration dictionary or None if error
    """
    config = configparser.ConfigParser()
    
    try:
        config.read(filepath)
        
        if 'Parameters' not in config:
            return None
            
        return {
            'image_path': config.get('Parameters', 'image_path', fallback=''),
            'mask_path': config.get('Parameters', 'mask_path', fallback=''),
            'l_perp': config.getint('Parameters', 'l_perp', fallback=8),
            'n_channels': config.getint('Parameters', 'n_channels', fallback=1),
            'colormap': config.get('Parameters', 'colormap', fallback='Default')
        }
    except Exception as e:
        print(f"Error loading config: {e}")
        return None


def save_config(filepath: str, config_dict: Dict[str, Any]) -> bool:
    """
    Save configuration to INI file.
    
    Parameters
    ----------
    filepath : str
        Path to save configuration file
    config_dict : dict
        Configuration dictionary
        
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    config = configparser.ConfigParser()
    
    config['Parameters'] = {
        'image_path': str(config_dict.get('image_path', '')),
        'mask_path': str(config_dict.get('mask_path', '')),
        'l_perp': str(config_dict.get('l_perp', 8)),
        'n_channels': str(config_dict.get('n_channels', 1)),
        'colormap': str(config_dict.get('colormap', 'Default'))
    }
    
    try:
        with open(filepath, 'w') as f:
            config.write(f)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns
    -------
    dict
        Default configuration
    """
    return {
        'image_path': '',
        'mask_path': '',
        'l_perp': 8,
        'n_channels': 1,
        'colormap': 'Default'
    }