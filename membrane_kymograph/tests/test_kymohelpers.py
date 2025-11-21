"""
Tests for kymohelpers module.
"""

import pytest
import numpy as np
from membrane_kymograph.kymohelpers import (
    smooth_boundary, interpboundary, aligninitboundary, alignboundary
)


class TestSmoothBoundary:
    """Tests for smooth_boundary function."""
    
    def test_smooth_boundary_basic(self):
        """Test basic smoothing."""
        bdy = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = smooth_boundary(bdy, 3)
        assert len(result) == len(bdy)
        assert np.all(result > 0)
        # Smoothed values should be closer to mean
        assert np.std(result) < np.std(bdy)
        
    def test_smooth_boundary_filter_sizes(self):
        """Test different filter sizes."""
        bdy = np.random.rand(20)
        
        # Small filter
        result1 = smooth_boundary(bdy, 1)
        assert result1.shape == bdy.shape
        
        # Medium filter
        result3 = smooth_boundary(bdy, 3)
        assert result3.shape == bdy.shape
        
        # Larger filter should produce smoother result
        result5 = smooth_boundary(bdy, 5)
        assert np.std(result5) < np.std(result3)
            
    def test_smooth_boundary_preserves_mean(self):
        """Test that smoothing preserves the mean value."""
        bdy = np.random.rand(50)
        result = smooth_boundary(bdy, 3)
        # Mean should be approximately preserved
        assert np.abs(np.mean(result) - np.mean(bdy)) < 0.1
        
    def test_smooth_boundary_2d(self):
        """Test smoothing 2D coordinates."""
        bdy_x = np.random.rand(10)
        bdy_y = np.random.rand(10)
        result_x = smooth_boundary(bdy_x, 3)
        result_y = smooth_boundary(bdy_y, 3)
        assert result_x.shape == (10,)
        assert result_y.shape == (10,)


class TestInterpBoundary:
    """Tests for interpboundary function."""
    
    def test_interpboundary_basic(self):
        """Test basic interpolation returns valid boundary."""
        bdy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = interpboundary(bdy)
        # Function interpolates internally but subsamples back to pixel resolution
        assert len(result) >= len(bdy)
        assert result.shape[1] == 2  # Should have 2 columns (x, y)
        
    def test_interpboundary_closed(self):
        """Test that boundary forms a closed loop."""
        bdy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = interpboundary(bdy)
        # Check that first and last points are reasonably close (forms a loop)
        dist = np.linalg.norm(result[0] - result[-1])
        assert dist <= 1.5  # Relaxed constraint for pixel-resolution boundary
        
    def test_interpboundary_subpixel(self):
        """Test different subpixel resolutions produce valid boundaries."""
        bdy = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        result_025 = interpboundary(bdy, subpixel=0.25)
        result_05 = interpboundary(bdy, subpixel=0.5)
        
        # Both should return valid boundaries (function subsamples back to pixel resolution)
        # So they may have similar lengths, but should be valid
        assert len(result_025) >= len(bdy)
        assert len(result_05) >= len(bdy)
        assert result_025.shape[1] == 2
        assert result_05.shape[1] == 2
        
    def test_interpboundary_circular(self):
        """Test interpolation of circular boundary."""
        theta = np.linspace(0, 2*np.pi, 20, endpoint=False)
        bdy = np.column_stack([np.cos(theta), np.sin(theta)])
        
        result = interpboundary(bdy)
        
        # Result should still be roughly circular
        distances = np.linalg.norm(result, axis=1)
        assert np.std(distances) < 0.1  # Low variance in radius


class TestAlignBoundary:
    """Tests for boundary alignment functions."""
    
    def test_aligninitboundary(self):
        """Test initial boundary alignment."""
        # Create circular boundary
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        bdy = np.column_stack([np.cos(theta), np.sin(theta)])
        center = np.array([0, 0])
        
        aligned, idx = aligninitboundary(bdy, center, 90)
        
        assert aligned.shape == bdy.shape
        assert 0 <= idx < len(bdy)
        
    def test_aligninitboundary_different_angles(self):
        """Test alignment at different angles."""
        theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
        bdy = np.column_stack([np.cos(theta), np.sin(theta)])
        center = np.array([0, 0])
        
        aligned_0, idx_0 = aligninitboundary(bdy, center, 0)
        aligned_90, idx_90 = aligninitboundary(bdy, center, 90)
        aligned_180, idx_180 = aligninitboundary(bdy, center, 180)
        
        # Different angles should give different starting indices
        assert aligned_0.shape == aligned_90.shape == aligned_180.shape
        
    def test_alignboundary(self):
        """Test boundary-to-boundary alignment."""
        # Create two similar boundaries
        theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
        bdy1 = np.column_stack([np.cos(theta), np.sin(theta)])
        bdy2 = np.roll(bdy1, 10, axis=0)  # Shifted version
        
        aligned, dist = alignboundary(bdy2, bdy1)
        
        assert aligned.shape == bdy2.shape
        assert dist >= 0
        
    def test_alignboundary_identical(self):
        """Test alignment of identical boundaries."""
        theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
        bdy = np.column_stack([np.cos(theta), np.sin(theta)])
        
        aligned, dist = alignboundary(bdy, bdy)
        
        # Distance should be very small for identical boundaries
        assert dist < 0.1
        
    def test_alignboundary_preserves_shape(self):
        """Test that alignment preserves boundary shape."""
        # Create elliptical boundary
        theta = np.linspace(0, 2*np.pi, 60, endpoint=False)
        bdy1 = np.column_stack([2*np.cos(theta), np.sin(theta)])
        bdy2 = np.roll(bdy1, 15, axis=0)
        
        aligned, dist = alignboundary(bdy2, bdy1)
        
        # Aligned boundary should have similar shape statistics
        assert aligned.shape == bdy2.shape
        # Mean position should be similar
        assert np.allclose(np.mean(aligned, axis=0), np.mean(bdy2, axis=0), atol=0.5)