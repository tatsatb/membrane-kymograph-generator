"""
Correlation analysis module for kymograph data with two different channels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from typing import Tuple, Optional


class KymographCorrelation:
    """Class for computing correlations between two kymographs."""
    
    def __init__(self):
        """Initialize correlation analyzer."""
        self.kymo1 = None
        self.kymo2 = None
        self.kymo1_path = None
        self.kymo2_path = None
        
    def load_kymographs(self, kymo1_path: str, kymo2_path: str) -> None:
        """
        Load two kymograph .npy files.
        
        Parameters
        ----------
        kymo1_path : str
            Path to first kymograph file
        kymo2_path : str
            Path to second kymograph file
            
        Raises
        ------
        FileNotFoundError
            If files don't exist
        ValueError
            If kymographs have different shapes
        """
        if not os.path.exists(kymo1_path):
            raise FileNotFoundError(f"Kymograph 1 not found: {kymo1_path}")
        if not os.path.exists(kymo2_path):
            raise FileNotFoundError(f"Kymograph 2 not found: {kymo2_path}")
        
        self.kymo1 = np.load(kymo1_path)
        self.kymo2 = np.load(kymo2_path)
        self.kymo1_path = kymo1_path
        self.kymo2_path = kymo2_path
        
        # Validate shapes match
        if self.kymo1.shape != self.kymo2.shape:
            raise ValueError(
                f"Kymograph shapes do not match!\n"
                f"Kymograph 1: {self.kymo1.shape}\n"
                f"Kymograph 2: {self.kymo2.shape}"
            )
    
    def compute_correlations(self, exclude_zeros: bool = True) -> pd.DataFrame:
        """
        Compute Pearson and Spearman correlations for each frame.
        
        Parameters
        ----------
        exclude_zeros : bool, optional
            Whether to exclude zero values from correlation (default: True)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: Frame, Pearson_R, Pearson_P, Spearman_R, Spearman_P
        """
        if self.kymo1 is None or self.kymo2 is None:
            raise ValueError("Kymographs not loaded. Call load_kymographs() first.")
        
        n_positions, n_frames = self.kymo1.shape
        
        results = {
            'Frame': [],
            'Pearson_R': [],
            'Pearson_P': [],
            'Spearman_R': [],
            'Spearman_P': []
        }
        
        for frame_idx in range(n_frames):
            # Get data for this frame
            data1 = self.kymo1[:, frame_idx]
            data2 = self.kymo2[:, frame_idx]
            
            # Exclude zeros if requested
            if exclude_zeros:
                # Find indices where both values are non-zero
                valid_mask = (data1 != 0) & (data2 != 0) & ~np.isnan(data1) & ~np.isnan(data2)
                data1_filtered = data1[valid_mask]
                data2_filtered = data2[valid_mask]
            else:
                # Just remove NaN values
                valid_mask = ~np.isnan(data1) & ~np.isnan(data2)
                data1_filtered = data1[valid_mask]
                data2_filtered = data2[valid_mask]
            
            # Compute correlations if we have enough data points
            if len(data1_filtered) >= 3:
                try:
                    pearson_r, pearson_p = pearsonr(data1_filtered, data2_filtered)
                    spearman_r, spearman_p = spearmanr(data1_filtered, data2_filtered)
                except Exception as e:
                    print(f"Warning: Correlation failed for frame {frame_idx + 1}: {e}")
                    pearson_r, pearson_p = np.nan, np.nan
                    spearman_r, spearman_p = np.nan, np.nan
            else:
                pearson_r, pearson_p = np.nan, np.nan
                spearman_r, spearman_p = np.nan, np.nan
            
            results['Frame'].append(frame_idx + 1)
            results['Pearson_R'].append(pearson_r)
            results['Pearson_P'].append(pearson_p)
            results['Spearman_R'].append(spearman_r)
            results['Spearman_P'].append(spearman_p)
        
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Save correlation results to Excel file.
        
        Parameters
        ----------
        df : pd.DataFrame
            Correlation results
        output_path : str
            Path to save Excel file
        """
        df.to_excel(output_path, index=False)
        print(f"Results saved to: {output_path}")
    
    def plot_correlations(self, df: pd.DataFrame, output_dir: str,
                         kymo1_name: str = "Kymograph 1",
                         kymo2_name: str = "Kymograph 2") -> Tuple[str, str]:
        """
        Plot correlation time series using seaborn.
        
        Parameters
        ----------
        df : pd.DataFrame
            Correlation results
        output_dir : str
            Directory to save plots
        kymo1_name : str, optional
            Name for first kymograph (default: "Kymograph 1")
        kymo2_name : str, optional
            Name for second kymograph (default: "Kymograph 2")
            
        Returns
        -------
        tuple
            Paths to (correlation_plot, pvalue_plot)
        """
        os.makedirs(output_dir, exist_ok=True)
        

        sns.set_theme(style="ticks", palette="Set2")
        sns.set_context("paper", font_scale=1.2)
        

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        

        palette_colors = sns.color_palette()
        
        # Pearson correlation
        axes[0].plot(df['Frame'], df['Pearson_R'], 
                    linewidth=2, color=palette_colors[0], label='Pearson R', marker='o', markersize=3)
        axes[0].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].set_ylabel('Pearson Correlation (R)', fontsize=12)
        axes[0].set_title(f'Correlation between {kymo1_name} and {kymo2_name}', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].grid(False)  # Turn off grid
        axes[0].tick_params(which='both', direction='out', length=6)  # Keep ticks visible
        axes[0].set_ylim(-1.05, 1.05)
        
        # Spearman correlation
        axes[1].plot(df['Frame'], df['Spearman_R'], 
                    linewidth=2, color=palette_colors[1], label='Spearman R', marker='s', markersize=3)
        axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].set_xlabel('Frame', fontsize=12)
        axes[1].set_ylabel('Spearman Correlation (R)', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(False)  # Turn off grid
        axes[1].tick_params(which='both', direction='out', length=6)  # Keep ticks visible
        axes[1].set_ylim(-1.05, 1.05)
        
        plt.tight_layout()
        sns.despine() 
        
        # Save PNG
        corr_plot_png = os.path.join(output_dir, 'correlation_coefficients.png')
        plt.savefig(corr_plot_png, dpi=300, bbox_inches='tight')
        
        # Save PDF
        corr_plot_pdf = os.path.join(output_dir, 'correlation_coefficients.pdf')
        plt.savefig(corr_plot_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        # Plot of P-values over time
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        

        palette_colors = sns.color_palette()
        
        # Pearson p-values
        axes[0].plot(df['Frame'], df['Pearson_P'], 
                    linewidth=2, color=palette_colors[0], label='Pearson P-value', marker='o', markersize=3)
        axes[0].axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, 
                       alpha=0.7, label='p=0.05')
        axes[0].set_ylabel('Pearson P-value', fontsize=12)
        axes[0].set_title(f'Correlation P-values: {kymo1_name} vs {kymo2_name}', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].grid(False)  # Turn off grid
        axes[0].tick_params(which='both', direction='out', length=6)  # Keep ticks visible
        axes[0].set_yscale('log')
        
        # Spearman p-values
        axes[1].plot(df['Frame'], df['Spearman_P'], 
                    linewidth=2, color=palette_colors[1], label='Spearman P-value', marker='s', markersize=3)
        axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, 
                       alpha=0.7, label='p=0.05')
        axes[1].set_xlabel('Frame', fontsize=12)
        axes[1].set_ylabel('Spearman P-value', fontsize=12)
        axes[1].legend(loc='best')
        axes[1].grid(False)  # Turn off grid
        axes[1].tick_params(which='both', direction='out', length=6)  # Keep ticks visible
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        sns.despine()  
        
        # Save PNG
        pval_plot_png = os.path.join(output_dir, 'correlation_pvalues.png')
        plt.savefig(pval_plot_png, dpi=300, bbox_inches='tight')
        
        # Save PDF
        pval_plot_pdf = os.path.join(output_dir, 'correlation_pvalues.pdf')
        plt.savefig(pval_plot_pdf, format='pdf', bbox_inches='tight')
        plt.close()
        
        return corr_plot_png, pval_plot_png
    
    def run_full_analysis(self, kymo1_path: str, kymo2_path: str,
                         output_dir: str, exclude_zeros: bool = True,
                         kymo1_name: Optional[str] = None,
                         kymo2_name: Optional[str] = None) -> dict:
        """
        Run complete correlation analysis workflow.
        
        Parameters
        ----------
        kymo1_path : str
            Path to first kymograph
        kymo2_path : str
            Path to second kymograph
        output_dir : str
            Directory to save results
        exclude_zeros : bool, optional
            Whether to exclude zeros (default: True)
        kymo1_name : str, optional
            Name for kymograph 1
        kymo2_name : str, optional
            Name for kymograph 2
            
        Returns
        -------
        dict
            Results dictionary with paths to output files
        """
        # Load kymographs
        self.load_kymographs(kymo1_path, kymo2_path)
        
        # Use filenames if names not provided
        if kymo1_name is None:
            kymo1_name = os.path.basename(kymo1_path).replace('.npy', '')
        if kymo2_name is None:
            kymo2_name = os.path.basename(kymo2_path).replace('.npy', '')
        
        # Compute correlations
        df = self.compute_correlations(exclude_zeros=exclude_zeros)
        
        # Save to Excel
        os.makedirs(output_dir, exist_ok=True)
        excel_path = os.path.join(output_dir, 'correlation_analysis.xlsx')
        self.save_results(df, excel_path)
        
        # Generate plots
        corr_plot, pval_plot = self.plot_correlations(
            df, output_dir, kymo1_name, kymo2_name
        )
        
        return {
            'excel_file': excel_path,
            'correlation_plot': corr_plot,
            'pvalue_plot': pval_plot,
            'dataframe': df,
            'n_frames': len(df),
            'exclude_zeros': exclude_zeros
        }
