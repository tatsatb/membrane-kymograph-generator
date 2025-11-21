"""
Utility functions for membrane kymograph generator.
"""

import os
import shutil
import logging
from typing import Generator, Tuple
import numpy as np
import tifffile

logger = logging.getLogger(__name__)


def save_data_safely(filepath: str, data: np.ndarray) -> None:
    """
    Save data with backup and error handling.
    
    Parameters
    ----------
    filepath : str
        Path to save file
    data : np.ndarray
        Data to save
        
    Raises
    ------
    Exception
        If save fails
    """
    backup_path = None
    
    try:
        # Create backup if file exists
        if os.path.exists(filepath):
            backup_path = filepath + '.backup'
            shutil.copy2(filepath, backup_path)
            
        # Save data
        np.save(filepath, data)
        
        # Remove backup on success
        if backup_path and os.path.exists(backup_path):
            os.remove(backup_path)
            
    except Exception as e:
        # Restore from backup if available
        if backup_path and os.path.exists(backup_path):
            shutil.copy2(backup_path, filepath)
            os.remove(backup_path)
        raise Exception(f"Failed to save {filepath}: {str(e)}")


def load_tiff_stack_chunked(filepath: str, chunk_size: int = 100) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    Load TIFF stack in chunks to manage memory.
    
    Parameters
    ----------
    filepath : str
        Path to TIFF file
    chunk_size : int, optional
        Number of frames per chunk (default: 100)
        
    Yields
    ------
    tuple
        (chunk_data, start_index, end_index)
    """
    with tifffile.TiffFile(filepath) as tif:
        total_pages = len(tif.pages)
        
        if total_pages == 0:
            raise ValueError("TIFF file contains no images")
            
        shape = tif.pages[0].shape
        dtype = tif.pages[0].dtype
        
        for start_idx in range(0, total_pages, chunk_size):
            end_idx = min(start_idx + chunk_size, total_pages)
            chunk = np.zeros((shape[0], shape[1], end_idx - start_idx), dtype=dtype)
            
            for i, page_idx in enumerate(range(start_idx, end_idx)):
                chunk[:, :, i] = tif.pages[page_idx].asarray()
                
            yield chunk, start_idx, end_idx


def setup_logging(log_file: str = 'kymograph.log', level: int = logging.INFO) -> None:
    """
    Setup logging configuration.
    
    Parameters
    ----------
    log_file : str, optional
        Log file path (default: 'kymograph.log')
    level : int, optional
        Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


class ProgressReporter:
    """Helper class for progress reporting."""
    
    def __init__(self, total_items: int, callback=None):
        """
        Initialize progress reporter.
        
        Parameters
        ----------
        total_items : int
            Total number of items to process
        callback : callable, optional
            Callback function for progress updates
        """
        self.total_items = total_items
        self.current_item = 0
        self.callback = callback
        
    def update(self, message: str = "") -> None:
        """Update progress."""
        self.current_item += 1
        progress = self.current_item / self.total_items * 100
        
        full_message = f"Progress: {self.current_item}/{self.total_items} ({progress:.1f}%)"
        if message:
            full_message += f" - {message}"
            
        logger.info(full_message)
        
        if self.callback:
            self.callback(full_message)