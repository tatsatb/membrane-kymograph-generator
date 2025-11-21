"""
Tests for processor module.
"""

import pytest
import numpy as np
import tempfile
import os
import tifffile
from membrane_kymograph.processor import KymographProcessor


class TestKymographProcessor:
    """Tests for KymographProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return KymographProcessor()
        
    @pytest.fixture
    def test_data(self):
        """Create test image and mask data."""
        # Create synthetic test data
        frames = 10
        channels = 2
        size = 100
        
        # Create circular cell mask
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = 30
        mask = ((x - center)**2 + (y - center)**2) <= radius**2
        
        # Create image data with membrane signal
        image_data = np.zeros((size, size, frames * channels), dtype=np.uint16)
        
        for f in range(frames):
            for c in range(channels):
                idx = f * channels + c
                # Add membrane signal
                membrane = np.zeros((size, size))
                membrane[mask] = 100
                # Add edge enhancement
                from scipy.ndimage import binary_dilation, binary_erosion
                edge = binary_dilation(mask) & ~binary_erosion(mask)
                membrane[edge] = 1000 + 100 * c  # Different intensity per channel
                
                image_data[:, :, idx] = membrane
                
        return image_data, mask.astype(np.uint8)
    
    @pytest.fixture
    def temp_tiff_files(self, test_data):
        """Create temporary TIFF files for testing."""
        image_data, mask_data = test_data
        frames = 10
        
        # Create temporary TIFF files
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as img_file:
            image_path = img_file.name
            tifffile.imwrite(image_path, image_data.transpose(2, 0, 1))  # Save as (frames, height, width)
            
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as mask_file:
            mask_path = mask_file.name
            # Stack mask for each frame
            mask_stack = np.stack([mask_data] * frames, axis=0)
            tifffile.imwrite(mask_path, mask_stack)
            
        yield image_path, mask_path
        
        # Cleanup
        try:
            os.unlink(image_path)
            os.unlink(mask_path)
        except:
            pass
        
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.stop_requested == False
        assert processor.n_frames == 0
        assert processor.n_channels == 1
        
    def test_validate_inputs_valid(self, processor, temp_tiff_files):
        """Test input validation with valid TIFF files."""
        image_path, mask_path = temp_tiff_files
        # Should not raise any exceptions
        processor._validate_inputs(image_path, mask_path, 8, 2)
            
    def test_validate_inputs_missing_file(self, processor, temp_tiff_files):
        """Test validation with missing files."""
        _, mask_path = temp_tiff_files
        with pytest.raises(FileNotFoundError):
            processor._validate_inputs('nonexistent.tif', mask_path, 8, 2)
                
    def test_validate_inputs_non_tiff(self, processor):
        """Test validation rejects non-TIFF files."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            jpg_path = tmp.name
            
        try:
            with pytest.raises(ValueError, match="must be a TIFF file"):
                processor._validate_inputs(jpg_path, jpg_path, 8, 2)
        finally:
            try:
                os.unlink(jpg_path)
            except:
                pass
                
    def test_validate_inputs_invalid_l_perp(self, processor, temp_tiff_files):
        """Test validation with invalid l_perp."""
        image_path, mask_path = temp_tiff_files
        
        with pytest.raises(ValueError, match="Pixel width"):
            processor._validate_inputs(image_path, mask_path, 0, 2)
            
        with pytest.raises(ValueError, match="Pixel width"):
            processor._validate_inputs(image_path, mask_path, 100, 2)
                
    def test_validate_inputs_invalid_channels(self, processor, temp_tiff_files):
        """Test validation with invalid channel count."""
        image_path, mask_path = temp_tiff_files
        
        with pytest.raises(ValueError, match="Number of channels"):
            processor._validate_inputs(image_path, mask_path, 8, 0)
                
    def test_extract_boundary(self, processor, test_data):
        """Test boundary extraction."""
        _, mask = test_data
        
        boundary, centroid, smooth_boundary = processor._extract_boundary(mask)
        
        assert len(boundary) > 0
        assert len(centroid) == 2
        assert len(smooth_boundary) > 0
        assert not np.any(np.isnan(centroid))
        
    def test_extract_boundary_empty_mask(self, processor):
        """Test boundary extraction with empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        
        boundary, centroid, smooth_boundary = processor._extract_boundary(mask)
        
        assert len(boundary) == 0
        assert np.all(np.isnan(centroid))
        assert len(smooth_boundary) == 0
        
    def test_stop_processing(self, processor):
        """Test stop functionality."""
        assert processor.stop_requested == False
        processor.stop()
        assert processor.stop_requested == True
        
    def test_callbacks(self, processor):
        """Test callback setting."""
        def mock_progress(msg):
            pass
            
        def mock_image(idx, path):
            pass
            
        processor.set_callbacks(progress_callback=mock_progress, image_callback=mock_image)
        
        assert processor.progress_callback == mock_progress
        assert processor.image_callback == mock_image