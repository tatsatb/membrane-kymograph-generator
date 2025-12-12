"""
Core processing module for membrane kymograph generation.
"""

import os
import time
import logging
import warnings
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed
import cv2
import tifffile
from skimage import measure, draw
from shapely.geometry import Polygon
from circle_fit import standardLSQ
import platform

# Configure matplotlib to use Agg backend BEFORE any matplotlib imports

import matplotlib
matplotlib.use('Agg', force=True)

# Additional safeguards to prevent GUI window creation
import os
os.environ['MPLBACKEND'] = 'Agg'  # Set environment variable
matplotlib.rcParams['interactive'] = False  # Disable interactive mode

import matplotlib.pyplot as plt
plt.ioff() 

from .kymohelpers import smooth_boundary, interpboundary, aligninitboundary, alignboundary
from .utils import save_data_safely, load_tiff_stack_chunked

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class KymographProcessor:
    """
    Main processor class for generating membrane kymographs.
    
    Attributes
    ----------
    stop_requested : bool
        Flag to stop processing
    n_frames : int
        Number of frames in the image stack
    n_channels : int
        Number of channels in the image
    boundary_abs : list
        Absolute boundaries for each frame
    centroid : np.ndarray
        Centroids for each frame
    """
    
    def __init__(self):
        self.stop_requested = False
        self.n_frames = 0
        self.n_channels = 1
        self.boundary_abs = None
        self.centroid = None
        self.smooth_boundary_final = None
        self.channels_data = []
        self.path_to_save = None
        self.path_to_temp_save = None
        self.progress_callback = None
        self.image_callback = None
        
    def set_callbacks(self, progress_callback=None, image_callback=None):
        """Set callback functions for progress and image updates."""
        self.progress_callback = progress_callback
        self.image_callback = image_callback
        
    def stop(self):
        """Request processing to stop."""
        self.stop_requested = True
        
    def _report_progress(self, message: str):
        """Report progress through callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
        logger.info(message)
        
    def _validate_inputs(self, image_path: str, mask_path: str, 
                        l_perp: int, n_channels: int) -> None:
        """Validate input parameters."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
            
        # Check if files are TIFF format
        valid_tiff_extensions = ('.tif', '.tiff', '.TIF', '.TIFF')
        if not image_path.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"Image file must be a TIFF file (.tif or .tiff). Got: {os.path.basename(image_path)}")
        if not mask_path.lower().endswith(('.tif', '.tiff')):
            raise ValueError(f"Mask file must be a TIFF file (.tif or .tiff). Got: {os.path.basename(mask_path)}")
            
        # Validate n_channels
        if n_channels <= 0:
            raise ValueError(f"Number of channels must be greater than 0. Got: {n_channels}")
            
        # Check dimensions compatibility
        try:
            with tifffile.TiffFile(image_path) as img_tif:
                n_image_pages = len(img_tif.pages)
                img_shape = img_tif.pages[0].shape  # (height, width)
                
            with tifffile.TiffFile(mask_path) as mask_tif:
                n_mask_pages = len(mask_tif.pages)
                mask_shape = mask_tif.pages[0].shape  # (height, width)
                
            # Calculate expected frames
            expected_frames_image = n_image_pages // n_channels
            expected_frames_mask = n_mask_pages
            
            # Check spatial dimensions match
            if img_shape != mask_shape:
                raise ValueError(
                    f"Image and mask spatial dimensions don't match!\n"
                    f"Image: {img_shape[1]}x{img_shape[0]} (width x height)\n"
                    f"Mask: {mask_shape[1]}x{mask_shape[0]} (width x height)"
                )
                
            # Check temporal dimensions match
            if expected_frames_image != expected_frames_mask:
                raise ValueError(
                    f"Image and mask temporal dimensions don't match!\n"
                    f"Image: {n_image_pages} pages / {n_channels} channels = {expected_frames_image} frames\n"
                    f"Mask: {n_mask_pages} frames\n"
                    f"Expected mask to have {expected_frames_image} frames."
                )
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise  
            else:
                raise ValueError(f"Error reading TIFF files: {str(e)}")
            
        if l_perp < 1 or l_perp > 50:
            raise ValueError("Pixel width (l_perp) must be between 1 and 50")
        if n_channels < 1:
            raise ValueError("Number of channels must be at least 1")
            
    def process(self, image_path: str, mask_path: str, l_perp: int = 8,
                colormap: str = "Default", n_channels: int = 1, 
                save_formats: List[str] = None) -> Dict[str, Any]:
        """
        Process microscopy images to generate membrane kymographs.
        
        Parameters
        ----------
        image_path : str
            Path to the multi-channel TIFF image stack
        mask_path : str
            Path to the binary mask TIFF stack
        l_perp : int, optional
            Perpendicular distance for intensity sampling (default: 8)
        colormap : str, optional
            Colormap for visualization (default: "Default")
        n_channels : int, optional
            Number of channels in the image (default: 1)
        save_formats : List[str], optional
            List of formats to save ('png', 'svg', 'pdf'). Defaults to ['png', 'svg']
            
        Returns
        -------
        dict
            Dictionary containing processed results
        """
        self.stop_requested = False
        self.n_channels = n_channels
        
        # Default to PNG and SVG if not specified
        if save_formats is None:
            save_formats = ['png', 'svg']
        
        try:
            # Validate inputs
            self._validate_inputs(image_path, mask_path, l_perp, n_channels)
            
            # Setup directories
            parent_dir = os.path.dirname(image_path)
            self.path_to_save = os.path.join(parent_dir, "Kymo_Processed_Data")
            self.path_to_temp_save = os.path.join(parent_dir, "Kymo_Temporary_Images")
            os.makedirs(self.path_to_save, exist_ok=True)
            os.makedirs(self.path_to_temp_save, exist_ok=True)
            
            # Load and prepare data
            self._report_progress("Loading image data...")
            self._load_and_prepare_data(image_path, mask_path)
            
            if self.stop_requested:
                return None
                
            # Process boundaries
            self._report_progress("Processing boundaries...")
            self._process_boundaries(mask_path)
            
            if self.stop_requested:
                return None
                
            # Generate kymographs
            self._report_progress("Generating kymographs...")
            kymo_results = self._generate_kymographs(l_perp)
            
            if self.stop_requested:
                return None
                
            # Save results
            self._report_progress("Saving results...Please wait...Kymographs will appear shortly.")
            self._save_results(kymo_results, colormap, save_formats)
            
            self._report_progress("Processing completed successfully!")
            
            return {
                'kymographs': kymo_results,
                'n_frames': self.n_frames,
                'n_channels': self.n_channels,
                'output_dir': parent_dir
            }
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}", exc_info=True)
            raise
            
    def _load_and_prepare_data(self, image_path: str, mask_path: str):
        """Load and prepare image data."""
        # Load image info
        with tifffile.TiffFile(image_path) as tif:
            n_images = len(tif.pages)
            shape = tif.pages[0].shape
            
        self.n_frames = n_images // self.n_channels
        self._report_progress(f"Found {self.n_frames} frames with {self.n_channels} channels")
        
        # Initialize storage
        self.channels_data = [np.zeros((shape[0], shape[1], self.n_frames), dtype=np.uint16) 
                             for _ in range(self.n_channels)]
        
        # Load data in chunks for memory efficiency
        chunk_size = min(100, n_images)
        for chunk_data, start_idx, end_idx in load_tiff_stack_chunked(image_path, chunk_size):
            for i in range(start_idx, end_idx):
                frame_idx = i // self.n_channels
                channel_idx = i % self.n_channels
                if frame_idx < self.n_frames:
                    self.channels_data[channel_idx][:, :, frame_idx] = chunk_data[:, :, i - start_idx]
                    
        # Save channel data 
        for c in range(self.n_channels):
            # Save as .npy
            save_data_safely(
                os.path.join(self.path_to_save, f'channel_{c+1}.npy'),
                self.channels_data[c]
            )
            
    def _process_boundaries(self, mask_path: str):
        """Process cell boundaries from mask."""
        # Load mask data
        with tifffile.TiffFile(mask_path) as tif:
            mask_frames = len(tif.pages)
            mask_data = tif.asarray()
        
        # Ensure mask_data is in (frames, height, width) format
        if len(mask_data.shape) == 2:
            # Single frame - add frame dimension
            mask_data = mask_data[np.newaxis, :, :]
        elif len(mask_data.shape) == 3:

            if mask_data.shape[0] == mask_frames:
                pass  # Already in (frames, height, width)
            else:
                mask_data = np.transpose(mask_data, (2, 0, 1))
            
        # Initialize storage
        boundary_final = [None] * self.n_frames
        smooth_boundary_final = [None] * self.n_frames
        centroid_final = np.full((self.n_frames, 2), np.nan)
        
        # Process each frame
        for frame_idx in range(min(self.n_frames, mask_frames)):
            if self.stop_requested:
                break
                
            self._report_progress(f"Processing boundary for frame {frame_idx + 1}/{self.n_frames}")
            
            # Extract mask for current frame (now in frames-first format)
            if frame_idx < mask_data.shape[0]:
                mask_frame = mask_data[frame_idx, :, :]
            else:
                mask_frame = mask_data[0, :, :]  # Use first frame if not enough masks
                
            # Process boundary
            boundary, centroid, smooth_boundary = self._extract_boundary(mask_frame)
            
            boundary_final[frame_idx] = boundary
            centroid_final[frame_idx, :] = centroid
            smooth_boundary_final[frame_idx] = smooth_boundary
            
        # Save results
        self.boundary_abs = smooth_boundary_final
        self.centroid = centroid_final.T
        
        # Save boundary data 
        save_data_safely(os.path.join(self.path_to_save, 'boundary.npy'), 
                        np.array(boundary_final, dtype=object))
        save_data_safely(os.path.join(self.path_to_save, 'smoothboundary.npy'), 
                        np.array(smooth_boundary_final, dtype=object))
        
        # Save centroid 
        save_data_safely(os.path.join(self.path_to_save, 'centroid.npy'), 
                        centroid_final)
        
        # Save Excel files
        try:
            df_centroid = pd.DataFrame(centroid_final, columns=['x', 'y'])
            df_centroid.to_excel(os.path.join(self.path_to_save, 'centroid.xlsx'), index=False)
        except Exception as e:
            logger.warning(f"Failed to save centroid.xlsx: {e}")
        
        
    def _extract_boundary(self, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract boundary from a single mask frame - EXACT legacy code replication."""

        image_mod_cell = np.array(mask, dtype=bool)
        image_mod_cell_label = image_mod_cell.astype(int)

        props = measure.regionprops_table(
            image_mod_cell_label,
            properties=('area', 'bbox', 'centroid', 'orientation', 'perimeter')
        )
        stats = pd.DataFrame(props)

        if stats.shape[0] == 0:
            return np.array([]), np.array([np.nan, np.nan]), np.array([])

        # Extract areas and centroids
        areas_tmp = np.zeros(stats.shape[0])
        centroid_tmp_0 = np.zeros(stats.shape[0])
        centroid_tmp_1 = np.zeros(stats.shape[0])

        for ij in range(stats.shape[0]):
            areas_tmp[ij] = stats['area'][ij]
            centroid_tmp_0[ij] = stats['centroid-0'][ij]
            centroid_tmp_1[ij] = stats['centroid-1'][ij]

        index1 = np.argmax(areas_tmp)
        centroid_0 = centroid_tmp_0[index1]
        centroid_1 = centroid_tmp_1[index1]

        BB0 = stats['bbox-0'][index1]
        BB1 = stats['bbox-1'][index1]
        BB2 = stats['bbox-2'][index1]
        BB3 = stats['bbox-3'][index1]

        BB_original = np.array([BB0, BB1, BB2, BB3])
        BB = BB_original + np.array([-1, -1, 2, 2])
        TMP_BW = image_mod_cell[BB[0]:BB[2], BB[1]:BB[3]]

        boundaries = measure.find_contours(TMP_BW, 0, fully_connected='high')

        if len(boundaries) == 0:
            return np.array([]), np.array([centroid_1, centroid_0]), np.array([])

        area_enclosed = np.zeros(len(boundaries))

        for jj in range(len(boundaries)):
            TMP_boundary = boundaries[jj]
            area_enclosed[jj] = Polygon(zip(TMP_boundary[:, 1], TMP_boundary[:, 0])).area

        index2 = np.argmax(area_enclosed)
        TMP_boundary = boundaries[index2]

        TMP_boundary_y = BB0 + TMP_boundary[:, 0]
        TMP_boundary_x = BB1 + TMP_boundary[:, 1]
        polygon_tmp = np.column_stack([TMP_boundary_y, TMP_boundary_x])

        # Interpolate
        interpolated_boundary = polygon_tmp
        interpolated_boundary = interpboundary(interpolated_boundary)

        interpolated_boundary_x = interpolated_boundary[:, 1]
        interpolated_boundary_y = interpolated_boundary[:, 0]

        interpolated_boundary_y_updated = smooth_boundary(interpolated_boundary_y, 3)
        interpolated_boundary_x_updated = smooth_boundary(interpolated_boundary_x, 3)
        

        polygon_tmp_smooth = np.column_stack([interpolated_boundary_y, interpolated_boundary_x])

        return np.array(polygon_tmp), np.array([centroid_1, centroid_0]), np.array(polygon_tmp_smooth)
        
    def _generate_kymographs(self, l_perp: int) -> List[np.ndarray]:
        """Generate kymograph matrices."""
        n_frames = self.n_frames
        kymo_smooth = np.zeros((self.n_channels, int(np.max([len(b) for b in self.boundary_abs if b is not None])), n_frames))
        channel_frames = [[] for _ in range(self.n_channels)]
        
        # Process each frame
        for frame_idx in range(n_frames):
            if self.stop_requested:
                break
                
            start_time = time.time()
            self._report_progress(f"Generating kymograph for frame {frame_idx + 1}/{n_frames}")
            
            # Get boundary for current frame
            boundary = self.boundary_abs[frame_idx]
            if boundary is None or len(boundary) == 0:
                continue
            
            tmpbdy = np.fliplr(boundary)
                
            # Get centroid for current frame
            x0 = self.centroid[0, frame_idx]
            y0 = self.centroid[1, frame_idx]

            if np.isnan(x0) or np.isnan(y0):
                fallback_centroid = np.nanmean(tmpbdy, axis=0)
                if np.any(np.isnan(fallback_centroid)):
                    self._report_progress(
                        f"Skipping frame {frame_idx + 1} due to invalid centroid values"
                    )
                    continue
                x0, y0 = fallback_centroid
                self.centroid[0, frame_idx] = x0
                self.centroid[1, frame_idx] = y0


            if frame_idx == 0:
                tmpbdy, dst = aligninitboundary(tmpbdy, np.array([x0, y0]), 180)
            else:
                x0old = self.centroid[0, frame_idx - 1]
                y0old = self.centroid[1, frame_idx - 1]
                
                prev_boundary = self.boundary_abs[frame_idx - 1]
                if prev_boundary is None or len(prev_boundary) == 0:
                    tmpbdy, dst = aligninitboundary(tmpbdy, np.array([x0, y0]), 180)
                else:
                    tmpbdy, dst = alignboundary(
                        tmpbdy - np.array([x0, y0]),
                        prev_boundary - np.array([x0old, y0old])
                    )
                    tmpbdy = tmpbdy + np.array([x0, y0])
            

            tmpbdy = tmpbdy[:-1, :]
            self.boundary_abs[frame_idx] = tmpbdy[:-1, :]
            aligned_boundary = tmpbdy
                
            # Sample intensities
            channel_values, perp_lines = self._sample_intensities(
                aligned_boundary, frame_idx, l_perp
            )
            
            # Store results
            for c in range(self.n_channels):
                channel_frames[c].append(channel_values[c])
                
            # Generate preview image if callback provided
            if self.image_callback:
                self._generate_preview(frame_idx, aligned_boundary, perp_lines)
                
            elapsed = time.time() - start_time
            self._report_progress(f"Frame {frame_idx + 1} processed in {elapsed:.2f}s")
            
        # Smooth and normalize kymographs
        kymo_results = self._smooth_and_normalize(channel_frames, kymo_smooth)
        
        return kymo_results
        
    def _sample_intensities(self, boundary: np.ndarray, frame_idx: int, 
                           l_perp: int) -> Tuple[List[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """Sample intensities perpendicular to boundary and return perpendicular line coords."""
        n_points = len(boundary)
        channel_values = [np.zeros(n_points) for _ in range(self.n_channels)]
        delta = 4
        
        def process_point(pt):
            if self.stop_requested:
                return [np.nan] * self.n_channels, None, None
                
            x0, y0 = boundary[pt]
            
            # Get local boundary segment
            cb = np.roll(boundary, delta - pt, axis=0)
            seg = cb[0:2 * delta, :]
            
            # Fit circle to local segment
            try:
                xc, yc, rc, _ = standardLSQ(seg)
                if np.isnan(rc):
                    raise ValueError("Circle fit failed")
            except:
                # Use simple perpendicular if circle fit fails (not accurate)
                if pt > 0:
                    dx = boundary[pt, 0] - boundary[pt-1, 0]
                    dy = boundary[pt, 1] - boundary[pt-1, 1]
                else:
                    dx = boundary[1, 0] - boundary[0, 0]
                    dy = boundary[1, 1] - boundary[0, 1]
                norm = np.sqrt(dx**2 + dy**2)
                if norm > 0:
                    dx, dy = -dy/norm, dx/norm
                else:
                    dx, dy = 0, 1
                xc = x0 - 1000*dx
                yc = y0 - 1000*dy
            
            # Calculate perpendicular direction
            theta = np.degrees(np.arctan2((x0 - xc), (y0 - yc)))
            d = np.arange(-l_perp, l_perp + 0.5, 0.5)
            
            x = x0 + d * np.sin(np.deg2rad(theta))
            y = y0 + d * np.cos(np.deg2rad(theta))
            
            # Sample intensities for each channel
            values = []
            for c in range(self.n_channels):
                coords = np.vstack([y, x]).T
                try:
                    tmp_vals = map_coordinates(
                        self.channels_data[c][:, :, frame_idx],
                        [coords[:, 0], coords[:, 1]],
                        order=2, mode='nearest'
                    )
                    # Take mean of top 5 values
                    sorted_vals = np.sort(tmp_vals)[::-1]
                    values.append(np.mean(sorted_vals[:5]))
                except:
                    values.append(np.nan)
                    
            return values, x, y
        
        # Process points in parallel using joblib
        results = Parallel(n_jobs=-1, prefer="threads")(
            delayed(process_point)(pt) for pt in range(n_points)
        )
        
        # Organize results by channel and collect perpendicular lines
        x_array = []
        y_array = []
        for pt, result in enumerate(results):
            if result is not None and len(result) == 3:
                values, x, y = result
                if values is not None:
                    for c in range(self.n_channels):
                        channel_values[c][pt] = values[c]
                if x is not None and y is not None:
                    x_array.append(x)
                    y_array.append(y)
        
        # Convert to arrays and subsample every 3rd line like legacy code
        if x_array and y_array:
            x_lines = np.array(x_array)
            y_lines = np.array(y_array)
            # Subsample every 3rd perpendicular line and transpose
            x_lines = x_lines[::3, :].T
            y_lines = y_lines[::3, :].T
            perp_lines = (x_lines, y_lines)
        else:
            perp_lines = (np.array([]), np.array([]))
                    
        return channel_values, perp_lines
    
    def _smooth_and_normalize(self, channel_frames: List[List[np.ndarray]], 
                             kymo_smooth: np.ndarray) -> List[np.ndarray]:
        """Smooth and normalize kymograph data."""
        max_len = kymo_smooth.shape[1]
        
        channel_all = [[] for _ in range(self.n_channels)]
        
        # Smooth data
        for c in range(self.n_channels):
            for i, frame_data in enumerate(channel_frames[c]):
                if len(frame_data) == 0:
                    continue
                
                # Collect raw data for normalization (BEFORE smoothing)
                channel_all[c].extend(frame_data)
                    
                # Apply LOWESS smoothing
                try:
                    smoothed = lowess(frame_data, np.arange(len(frame_data)), frac=0.1)
                    smoothed_values = smoothed[:, 1]
                except:
                    smoothed_values = frame_data
                    
                # Center in kymograph matrix
                offset = (max_len - len(smoothed_values)) // 2
                end_idx = offset + len(smoothed_values)
                kymo_smooth[c, offset:end_idx, i] = smoothed_values
                
        # Normalize each channel using RAW data statistics 
        kymo_normalized = []
        for c in range(self.n_channels):

            if len(channel_all[c]) > 0:

                s_channel = np.sort(np.array(channel_all[c]))
                M_c = s_channel[int(np.round(0.98 * len(s_channel)))]
                m_c = s_channel[int(np.round(0.02 * len(s_channel)))]
                

                kymo_norm = (kymo_smooth[c] - 0.02 * m_c) / (M_c - 0.02 * m_c)
                kymo_norm = np.clip(kymo_norm, 0, 1)
            else:
                kymo_norm = kymo_smooth[c]
                
            kymo_normalized.append(kymo_norm)
            
        return kymo_normalized
    
    def _generate_preview(self, frame_idx: int, boundary: np.ndarray, 
                          perp_lines: Tuple[np.ndarray, np.ndarray]):
        """Generate preview image for GUI display."""
        # Import Figure from matplotlib.figure to avoid pyplot
        # Agg backend is already set at module level to prevent Tk window creation
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        fig = Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        # Display first channel
        ax.imshow(self.channels_data[0][:, :, frame_idx], cmap='viridis')
        
        # Plot boundary in blue
        ax.plot(boundary[:, 0], boundary[:, 1], 'b-', linewidth=2)
        
        # Plot perpendicular lines in yellow
        x_lines, y_lines = perp_lines
        if x_lines.size > 0 and y_lines.size > 0:
            ax.plot(x_lines, y_lines, 'y-', linewidth=2)
        
        ax.set_title(f'Frame {frame_idx + 1}')
        ax.axis('off')
        
        fig.tight_layout()
        
        # Save using Agg backend (non-interactive) - backend already set at module level
        preview_path = os.path.join(self.path_to_temp_save, f'frame_{frame_idx + 1}.png')
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(preview_path, dpi=100, bbox_inches='tight')
        
        # Clean up figure to prevent memory leaks and window creation
        plt_close_needed = False
        try:
            # Only import pyplot if absolutely necessary
            import matplotlib.pyplot as plt
            plt_close_needed = True
        except:
            pass
        
        fig.clf()
        del fig, canvas
        
        # Close any pyplot figures if pyplot was imported
        if plt_close_needed:
            try:
                plt.close('all')
            except:
                pass
        
        if self.image_callback:
            self.image_callback(frame_idx, preview_path)
            
    def _save_results(self, kymo_results: List[np.ndarray], colormap: str, 
                     save_formats: List[str]):
        """Save kymograph results."""
        # Agg backend is already set at module level
        # No need to re-configure, but ensure environment variable is set
        import os
        import matplotlib
        
        # Set environment variable to prevent Tk window creation (defensive)
        os.environ['MPLBACKEND'] = 'Agg'
        
        # Backend is already Agg from module-level import, but force again to be safe
        matplotlib.use('Agg', force=True)
        
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from .parulamap import cm_data
        
        # Ensure interactive mode is off
        plt.ioff()
        
        # Additional safeguard: tell matplotlib not to show plots
        matplotlib.rcParams['interactive'] = False
        
        # Setup colormap
        if colormap == "Default":
            cmap = LinearSegmentedColormap.from_list('parula', cm_data)
        else:
            cmap = plt.get_cmap(colormap)
            
        # Save data and generate visualizations
        for c in range(self.n_channels):
            # Save raw kymograph data as .npy
            save_data_safely(
                os.path.join(self.path_to_save, f'kymo_channel_{c+1}_smooth.npy'),
                kymo_results[c]
            )
            
            # Save kymograph data as Excel in Kymo_Processed_Data folder
            try:
                df = pd.DataFrame(kymo_results[c])
                excel_path = os.path.join(self.path_to_save, f'kymo_channel_{c+1}_smooth.xlsx')
                df.to_excel(excel_path, index=False)
                
            except Exception as e:
                logger.warning(f"Failed to save kymograph Excel files for channel {c+1}: {e}")
            
            # Generate visualization using Figure
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)
            im = ax.imshow(kymo_results[c], aspect='auto', cmap=cmap, vmin=0, vmax=1)
            ax.set_xlabel('Frame')
            
            # Set y-axis to show angles from -π to π (positive at top, negative at bottom)
            n_positions = kymo_results[c].shape[0]
            # Create angle ticks at convenient intervals
            angle_ticks_positions = np.linspace(0, n_positions - 1, 9)  # 9 ticks from 0 to n-1
            angle_ticks_labels = ['π', '3π/4', 'π/2', 'π/4', '0', '-π/4', '-π/2', '-3π/4', '-π']
            ax.set_yticks(angle_ticks_positions)
            ax.set_yticklabels(angle_ticks_labels)
            ax.set_ylabel('Angle')
            
            ax.set_title(f'Channel {c+1} Kymograph')
            fig.colorbar(im, ax=ax, label='Normalized Intensity')
            
            # Save in requested formats
            base_path = os.path.dirname(self.path_to_save)
            
            if 'png' in save_formats:
                png_path = os.path.join(base_path, f'kymo_channel_{c+1}.png')
                fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
                
            if 'svg' in save_formats:
                svg_path = os.path.join(base_path, f'kymo_channel_{c+1}.svg')
                fig.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
                
            if 'pdf' in save_formats:
                pdf_path = os.path.join(base_path, f'kymo_channel_{c+1}.pdf')
                fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
                
            # Close the specific figure to free memory
            plt.close(fig)
            # Also clear the figure to release resources
            fig.clf()
            del fig, canvas
        
        # Close all remaining figures to ensure cleanup
        plt.close('all')