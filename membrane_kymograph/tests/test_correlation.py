"""
Tests for correlation module.
"""

import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Set matplotlib backend before importing correlation module
# This is needed for headless CI environments (GitHub Actions)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from membrane_kymograph.correlation import KymographCorrelation


class TestKymographCorrelation:
    """Tests for kymograph correlation analysis."""
    
    @pytest.fixture
    def sample_kymographs(self, tmp_path):
        """Create sample kymograph data for testing."""
        np.random.seed(42)
        # Create two correlated kymographs (100 frames, 50 pixels)
        kymo1 = np.random.rand(100, 50) * 100
        # kymo2 is correlated with kymo1 plus some noise
        kymo2 = kymo1 * 0.8 + np.random.rand(100, 50) * 20
        
        # Save to temporary files
        kymo1_path = tmp_path / "kymo1_smoothed.npy"
        kymo2_path = tmp_path / "kymo2_smoothed.npy"
        np.save(kymo1_path, kymo1)
        np.save(kymo2_path, kymo2)
        
        return str(kymo1_path), str(kymo2_path), kymo1, kymo2
    
    @pytest.fixture
    def sample_kymographs_with_zeros(self, tmp_path):
        """Create sample kymographs with zero values."""
        np.random.seed(42)
        kymo1 = np.random.rand(50, 30) * 100
        kymo2 = kymo1 * 0.7 + np.random.rand(50, 30) * 30
        
        # Add some zeros
        kymo1[10:15, :] = 0
        kymo2[20:25, :] = 0
        
        kymo1_path = tmp_path / "kymo1_zeros.npy"
        kymo2_path = tmp_path / "kymo2_zeros.npy"
        np.save(kymo1_path, kymo1)
        np.save(kymo2_path, kymo2)
        
        return str(kymo1_path), str(kymo2_path)
    
    def test_initialization(self):
        """Test KymographCorrelation initialization."""
        corr = KymographCorrelation()
        assert corr.kymo1 is None
        assert corr.kymo2 is None
        assert corr.kymo1_path is None
        assert corr.kymo2_path is None
    
    def test_load_kymographs_valid(self, sample_kymographs):
        """Test loading valid kymograph files."""
        kymo1_path, kymo2_path, expected_kymo1, expected_kymo2 = sample_kymographs
        
        corr = KymographCorrelation()
        corr.load_kymographs(kymo1_path, kymo2_path)
        
        assert corr.kymo1 is not None
        assert corr.kymo2 is not None
        assert corr.kymo1.shape == expected_kymo1.shape
        assert corr.kymo2.shape == expected_kymo2.shape
        assert corr.kymo1_path == kymo1_path
        assert corr.kymo2_path == kymo2_path
    
    def test_load_kymographs_shape_mismatch(self, tmp_path):
        """Test that loading mismatched shapes raises ValueError."""
        kymo1 = np.random.rand(100, 50)
        kymo2 = np.random.rand(100, 60)  # Different width!
        
        kymo1_path = tmp_path / "kymo1.npy"
        kymo2_path = tmp_path / "kymo2.npy"
        np.save(kymo1_path, kymo1)
        np.save(kymo2_path, kymo2)
        
        corr = KymographCorrelation()
        with pytest.raises(ValueError, match="Kymograph shapes do not match"):
            corr.load_kymographs(str(kymo1_path), str(kymo2_path))
    
    def test_load_kymographs_missing_file(self, tmp_path):
        """Test that loading non-existent file raises FileNotFoundError."""
        kymo1_path = tmp_path / "kymo1.npy"
        kymo2_path = tmp_path / "nonexistent.npy"
        np.save(kymo1_path, np.random.rand(10, 10))
        
        corr = KymographCorrelation()
        with pytest.raises(FileNotFoundError):
            corr.load_kymographs(str(kymo1_path), str(kymo2_path))
    
    def test_compute_correlations_exclude_zeros(self, sample_kymographs_with_zeros):
        """Test correlation computation with zero exclusion."""
        kymo1_path, kymo2_path = sample_kymographs_with_zeros
        
        corr = KymographCorrelation()
        corr.load_kymographs(kymo1_path, kymo2_path)
        df = corr.compute_correlations(exclude_zeros=True)
        
        # Check DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30  # 30 frames (columns)
        assert 'Frame' in df.columns
        assert 'Pearson_R' in df.columns
        assert 'Pearson_P' in df.columns
        assert 'Spearman_R' in df.columns
        assert 'Spearman_P' in df.columns
        
        # Check that correlations are valid
        assert df['Pearson_R'].notna().all()  # No NaN values
        assert (df['Pearson_R'] >= -1).all() and (df['Pearson_R'] <= 1).all()
        assert (df['Spearman_R'] >= -1).all() and (df['Spearman_R'] <= 1).all()
    
    def test_compute_correlations_include_zeros(self, sample_kymographs_with_zeros):
        """Test correlation computation without zero exclusion."""
        kymo1_path, kymo2_path = sample_kymographs_with_zeros
        
        corr = KymographCorrelation()
        corr.load_kymographs(kymo1_path, kymo2_path)
        df = corr.compute_correlations(exclude_zeros=False)
        
        # Should still work but may have different correlations
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 30
        assert df['Pearson_R'].notna().all()
    
    def test_compute_correlations_without_loading(self):
        """Test that computing correlations without loading raises error."""
        corr = KymographCorrelation()
        with pytest.raises(ValueError, match="Kymographs not loaded"):
            corr.compute_correlations()
    
    def test_save_results(self, sample_kymographs, tmp_path):
        """Test saving correlation results to Excel."""
        kymo1_path, kymo2_path, _, _ = sample_kymographs
        
        corr = KymographCorrelation()
        corr.load_kymographs(kymo1_path, kymo2_path)
        df = corr.compute_correlations(exclude_zeros=False)
        
        output_file = tmp_path / "correlations.xlsx"
        corr.save_results(df, str(output_file))
        
        # Check file was created
        assert os.path.exists(output_file)
        assert str(output_file).endswith('.xlsx')
        
        # Check we can read it back
        df_loaded = pd.read_excel(output_file)
        assert len(df_loaded) == len(df)
        assert 'Frame' in df_loaded.columns
    
    def test_plot_correlations(self, sample_kymographs, tmp_path):
        """Test plotting correlation results."""
        kymo1_path, kymo2_path, _, _ = sample_kymographs
        
        corr = KymographCorrelation()
        corr.load_kymographs(kymo1_path, kymo2_path)
        df = corr.compute_correlations(exclude_zeros=False)
        
        corr_plot, pval_plot = corr.plot_correlations(
            df, str(tmp_path), "kymo1", "kymo2"
        )
        
        # Check both PNG files were created
        assert os.path.exists(corr_plot)
        assert os.path.exists(pval_plot)
        assert corr_plot.endswith('.png')
        assert pval_plot.endswith('.png')
        
        # Check that corresponding PDF files also exist
        corr_pdf = corr_plot.replace('.png', '.pdf')
        pval_pdf = pval_plot.replace('.png', '.pdf')
        assert os.path.exists(corr_pdf)
        assert os.path.exists(pval_pdf)
    
    def test_run_full_analysis(self, sample_kymographs, tmp_path):
        """Test complete analysis workflow."""
        kymo1_path, kymo2_path, _, _ = sample_kymographs
        
        corr = KymographCorrelation()
        
        results = corr.run_full_analysis(
            kymo1_path, kymo2_path,
            output_dir=str(tmp_path),
            exclude_zeros=True
        )
        
        # Check results dictionary
        assert 'excel_file' in results
        assert 'correlation_plot' in results
        assert 'pvalue_plot' in results
        assert 'dataframe' in results
        assert 'n_frames' in results
        assert 'exclude_zeros' in results
        
        # Check files were created
        assert os.path.exists(results['excel_file'])
        assert os.path.exists(results['correlation_plot'])
        assert os.path.exists(results['pvalue_plot'])
        
        # Check dataframe
        df = results['dataframe']
        assert len(df) == results['n_frames']
        assert 'Pearson_R' in df.columns
        assert 'Spearman_R' in df.columns
    
    def test_summary_statistics(self, sample_kymographs):
        """Test that summary statistics are calculated correctly."""
        kymo1_path, kymo2_path, _, _ = sample_kymographs
        
        corr = KymographCorrelation()
        corr.load_kymographs(kymo1_path, kymo2_path)
        df = corr.compute_correlations(exclude_zeros=False)
        
        mean_pearson = df['Pearson_R'].mean()
        median_spearman = df['Spearman_R'].median()
        
        # Should be positive correlation (since we created correlated data)
        assert mean_pearson > 0
        assert median_spearman > 0
    
    def test_correlation_consistency(self, sample_kymographs):
        """Test that running analysis twice gives same results."""
        kymo1_path, kymo2_path, _, _ = sample_kymographs
        
        corr1 = KymographCorrelation()
        corr1.load_kymographs(kymo1_path, kymo2_path)
        df1 = corr1.compute_correlations(exclude_zeros=False)
        
        corr2 = KymographCorrelation()
        corr2.load_kymographs(kymo1_path, kymo2_path)
        df2 = corr2.compute_correlations(exclude_zeros=False)
        
        # Results should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_edge_case_single_frame(self, tmp_path):
        """Test handling of kymographs with only one frame."""
        kymo1 = np.random.rand(100, 1)  # Single frame (column)
        kymo2 = np.random.rand(100, 1)
        
        kymo1_path = tmp_path / "kymo1_single.npy"
        kymo2_path = tmp_path / "kymo2_single.npy"
        np.save(kymo1_path, kymo1)
        np.save(kymo2_path, kymo2)
        
        corr = KymographCorrelation()
        corr.load_kymographs(str(kymo1_path), str(kymo2_path))
        df = corr.compute_correlations(exclude_zeros=False)
        
        # Should have 1 row
        assert len(df) == 1
        assert df['Frame'].iloc[0] == 1  # Frame numbering starts at 1
    
    def test_all_zeros_frame(self, tmp_path):
        """Test handling of frames that are all zeros."""
        kymo1 = np.zeros((50, 20))
        kymo2 = np.zeros((50, 20))
        
        kymo1_path = tmp_path / "kymo1_zeros.npy"
        kymo2_path = tmp_path / "kymo2_zeros.npy"
        np.save(kymo1_path, kymo1)
        np.save(kymo2_path, kymo2)
        
        corr = KymographCorrelation()
        corr.load_kymographs(str(kymo1_path), str(kymo2_path))
        
        # With exclude_zeros=True, should handle this gracefully
        df = corr.compute_correlations(exclude_zeros=True)
        
        # May have NaN for frames with no valid data
        assert len(df) == 20
