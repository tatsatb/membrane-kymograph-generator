"""
GUI module for membrane kymograph generator.
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import platform

# Import version from package
try:
    from . import __version__
except ImportError:
    __version__ = "unknown"

# CRITICAL: Configure matplotlib backend BEFORE any pyplot imports
# This prevents extra window creation on macOS
import matplotlib

# Force Agg backend globally and configure to never create windows
os.environ['MPLBACKEND'] = 'Agg'
matplotlib.use('Agg', force=True)
matplotlib.rcParams['interactive'] = False
matplotlib.rcParams['figure.max_open_warning'] = 0

# Prevent matplotlib from creating new Tk instances
if platform.system() == 'Darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    matplotlib.rcParams['backend'] = 'Agg'

import matplotlib.pyplot as plt
plt.ioff()
plt.close('all')

# Import Figure and TkAgg backend for GUI embedding ONLY

import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .processor import KymographProcessor
from .config import load_config, save_config
from .correlation import KymographCorrelation
import sys
import platform


class KymographGUI:
    """Main GUI class for the membrane kymograph generator."""
    
    _instance = None
    _instance_created = False

    def __new__(cls):
        """Ensure only one instance of the GUI can exist."""
        if cls._instance is None:
            cls._instance = super(KymographGUI, cls).__new__(cls)
            cls._instance_created = False
        return cls._instance

    def __init__(self):
        # Prevent re-initialization if already created
        if self._instance_created:
            return
        
        self._instance_created = True
        KymographGUI._instance_created = True  # Set on class too
        
        # CRITICAL: Destroy any existing Tk root windows before creating new one
        try:
            if hasattr(tk, '_default_root') and tk._default_root is not None:
                tk._default_root.destroy()
                tk._default_root = None
        except:
            pass
        
        self.root = tk.Tk()
        
        # macOS specific: bring window to front
        if platform.system() == 'Darwin':
            self.root.lift()
            self.root.call('wm', 'attributes', '.', '-topmost', '1')
            self.root.after_idle(self.root.call, 'wm', 'attributes', '.', '-topmost', '0')
        
        # Thread-safe messagebox flag for macOS
        self._messagebox_queue = []
        
        self.style = ttk.Style("superhero")
        self.root.title(f"Membrane Kymograph Generator v{__version__}")

        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        # Calculate window size (90% of screen height, maintaining reasonable width)
        # Minimum: 800x600, Maximum: 1200x1000
        # Target: 800 width, up to 90% of screen height
        max_height = int(screen_height * 0.9)
        window_width = min(max(800, screen_width // 2), 1200)  # 800-1200px wide
        window_height = min(max(600, max_height), 1000)  # 600-1000px tall

        # Store dimensions for later use
        self.window_width = window_width
        self.window_height = window_height

        # Center window on screen
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        self.root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")


        self.root.minsize(750, 550)


        self.root.resizable(True, True)

        # Set custom font with fallbacks
        import tkinter.font as tkfont

        try:
            default_font = tkfont.nametofont("TkDefaultFont")
            default_font.configure(family="Arial", size=10)
            text_font = tkfont.nametofont("TkTextFont")
            text_font.configure(family="Arial", size=10)
        except:
            try:
                default_font = tkfont.nametofont("TkDefaultFont")
                default_font.configure(family="Ubuntu", size=10)
                text_font = tkfont.nametofont("TkTextFont")
                text_font.configure(family="Ubuntu", size=10)
            except:
                try:
                    default_font = tkfont.nametofont("TkDefaultFont")
                    default_font.configure(family="Liberation Sans", size=10)
                    text_font = tkfont.nametofont("TkTextFont")
                    text_font.configure(family="Liberation Sans", size=10)
                except:
                    # Use system default
                    pass

        self.processor = KymographProcessor()
        self.processing_thread = None
        self.current_frame = 0
        self.n_frames = 1
        self.slider_step = 1
        self._updating_slider = False 
        self.kymo_canvas = None  

        self.setup_gui()
        self.load_default_config()
    
    def _safe_messagebox(self, func, *args, **kwargs):
        """Thread-safe wrapper for messagebox calls (macOS compatible)."""
        if platform.system() == 'Darwin':
            # On macOS, ensure we're on the main thread
            try:
                # Schedule on main thread if not already there
                self.root.after(0, lambda: func(*args, **kwargs))
            except:
                # Fallback: direct call if after() fails
                func(*args, **kwargs)
        else:
            # On other platforms, call directly
            func(*args, **kwargs)
    
    def safe_showinfo(self, title, message):
        """Thread-safe showinfo."""
        self._safe_messagebox(messagebox.showinfo, title, message)
    
    def safe_showerror(self, title, message):
        """Thread-safe showerror."""
        self._safe_messagebox(messagebox.showerror, title, message)
    
    def safe_showwarning(self, title, message):
        """Thread-safe showwarning."""
        self._safe_messagebox(messagebox.showwarning, title, message)
    
    def safe_askyesno(self, title, message):
        """Thread-safe askyesno - returns result via callback on macOS."""
        if platform.system() == 'Darwin':
            # On macOS, we need to be extra careful
            result = [None]  # Use list to store result from nested function
            def _ask():
                result[0] = messagebox.askyesno(title, message)
            self.root.after(0, _ask)
            self.root.update()  # Process the event
            return result[0]
        else:
            return messagebox.askyesno(title, message)

    def setup_gui(self):
        """Setup the GUI components."""

        input_frame = ttk.LabelFrame(self.root, text="Input Parameters", padding=10)
        input_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=5, sticky="ew")


        ttk.Label(input_frame, text="Image Path:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_image_path = ttk.Entry(input_frame, width=50)
        self.entry_image_path.grid(row=0, column=1, padx=5, pady=5)
        self.browse_image_button = ttk.Button(
            input_frame, text="Browse", command=self.browse_image
        )
        self.browse_image_button.grid(row=0, column=2, padx=5, pady=5)


        ttk.Label(input_frame, text="Mask Path:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.entry_mask_path = ttk.Entry(input_frame, width=50)
        self.entry_mask_path.grid(row=1, column=1, padx=5, pady=5)
        self.browse_mask_button = ttk.Button(
            input_frame, text="Browse", command=self.browse_mask
        )
        self.browse_mask_button.grid(row=1, column=2, padx=5, pady=5)


        params_frame = ttk.LabelFrame(
            self.root, text="Processing Parameters", padding=10
        )
        params_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="ew")

        ttk.Label(params_frame, text="Membrane Width (pixels):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.spinbox_l_perp = ttk.Spinbox(params_frame, from_=3, to=15, width=10)
        self.spinbox_l_perp.set(8)
        self.spinbox_l_perp.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(params_frame, text="Number of Channels:").grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )
        self.spinbox_channels = ttk.Spinbox(params_frame, from_=1, to=10, width=10)
        self.spinbox_channels.set(1)
        self.spinbox_channels.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        ttk.Label(params_frame, text="Colormap:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        self.colormap_var = tk.StringVar(value="Default")
        colormaps = [
            "Default",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Greys",
            "jet",
            "turbo",
            "gnuplot",
            "gnuplot2",
        ]
        self.colormap_menu = ttk.OptionMenu(
            params_frame, self.colormap_var, "Default", *colormaps
        )
        self.colormap_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(params_frame, text="Output Format:").grid(
            row=1, column=2, padx=5, pady=5, sticky="w"
        )
        format_frame = ttk.Frame(params_frame)
        format_frame.grid(row=1, column=3, padx=5, pady=5, sticky="w")

        self.save_png_var = tk.BooleanVar(value=True)
        self.save_svg_var = tk.BooleanVar(value=True)
        self.save_pdf_var = tk.BooleanVar(value=False)

        ttk.Checkbutton(format_frame, text="PNG", variable=self.save_png_var).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Checkbutton(format_frame, text="SVG", variable=self.save_svg_var).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Checkbutton(format_frame, text="PDF", variable=self.save_pdf_var).pack(
            side=tk.LEFT, padx=2
        )

        control_frame = ttk.Frame(self.root)
        control_frame.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        self.run_button = ttk.Button(
            control_frame,
            text="Generate Kymograph",
            command=self.run_kymograph,
            style="success.TButton",
        )
        self.run_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            control_frame,
            text="Stop",
            command=self.stop_processing,
            style="danger.TButton",
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.stop_button.config(state=tk.DISABLED)

        ttk.Button(
            control_frame,
            text="Save Config",
            command=self.save_config,
            style="info.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            control_frame,
            text="Load Config",
            command=self.load_config,
            style="info.TButton",
        ).pack(side=tk.LEFT, padx=5)

        self.adjust_limits_button = ttk.Button(
            control_frame,
            text="Adjust Existing Kymograph",
            command=self.open_adjust_limits_dialog,
            bootstyle="warning-outline",
        )
        self.adjust_limits_button.pack(side=tk.LEFT, padx=5)

        self.adjust_limits_button.config(state=tk.NORMAL)

        # Correlation analysis button
        self.correlation_button = ttk.Button(
            control_frame,
            text="Correlation Analysis",
            command=self.open_correlation_dialog,
            bootstyle="info-outline",
        )
        self.correlation_button.pack(side=tk.LEFT, padx=5)

        # Progress text
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=5)
        progress_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=5, sticky="ew")

        # Adjust text widget height based on window size
        text_height = max(
            4, min(8, self.window_height // 120)
        )  
        self.text_widget = tk.Text(progress_frame, height=text_height, width=80)
        self.text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(progress_frame, command=self.text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget.config(yscrollcommand=scrollbar.set)

        # Preview frame
        preview_frame = ttk.LabelFrame(self.root, text="Preview", padding=5)
        preview_frame.grid(
            row=4, column=0, columnspan=3, padx=10, pady=5, sticky="nsew"
        )

        # Frame slider 
        self.slider = tk.Scale(
            preview_frame,
            from_=1,
            to=1,
            orient=tk.HORIZONTAL,
            command=self.on_slider_change,
            showvalue=True,
            label="Frame Indicator [Once processing is completed, use the slider to rewind]",
        )
        self.slider.pack(fill=tk.X, padx=5, pady=5)

        self.canvas_frame = ttk.Frame(preview_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.root.grid_rowconfigure(4, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def browse_image(self):
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if filename:
            self.entry_image_path.delete(0, tk.END)
            self.entry_image_path.insert(0, filename)
            # Force GUI update on macOS to prevent freezing
            if platform.system() == 'Darwin':
                self.root.update()

    def browse_mask(self):
        """Browse for mask file."""
        filename = filedialog.askopenfilename(
            title="Select Mask File",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")],
        )
        if filename:
            self.entry_mask_path.delete(0, tk.END)
            self.entry_mask_path.insert(0, filename)
            # Force GUI update on macOS to prevent freezing
            if platform.system() == 'Darwin':
                self.root.update()

    def run_kymograph(self):
        """Start kymograph processing."""
        # Validate inputs
        if not self.validate_inputs():
            return

        # Get parameters
        image_path = self.entry_image_path.get()
        mask_path = self.entry_mask_path.get()
        l_perp = int(self.spinbox_l_perp.get())
        n_channels = int(self.spinbox_channels.get())
        colormap = self.colormap_var.get()

        # Get output format options
        save_formats = []
        if self.save_png_var.get():
            save_formats.append("png")
        if self.save_svg_var.get():
            save_formats.append("svg")
        if self.save_pdf_var.get():
            save_formats.append("pdf")

        if not save_formats:
            self.safe_showwarning(
                "No Format Selected",
                "Please select at least one output format (PNG, SVG, or PDF).",
            )
            return

        # Change to image directory
        os.chdir(os.path.dirname(image_path))

        # Update UI state
        self.set_processing_state(True)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, "Starting processing...\n")

        # Setup processor callbacks
        self.processor.set_callbacks(
            progress_callback=self.update_progress, image_callback=self.update_preview
        )

        # Start processing in thread
        self.processing_thread = threading.Thread(
            target=self._process_worker,
            args=(image_path, mask_path, l_perp, colormap, n_channels, save_formats),
            daemon=True,
        )
        self.processing_thread.start()

        # Start monitoring thread
        self.root.after(100, self.check_processing_thread)

    def _process_worker(
        self, image_path, mask_path, l_perp, colormap, n_channels, save_formats
    ):
        """Worker function for processing thread."""
        try:
            results = self.processor.process(
                image_path, mask_path, l_perp, colormap, n_channels, save_formats
            )
            if results:
                self.n_frames = results["n_frames"]
                self.root.after(0, self.processing_complete, results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, self.processing_error, str(e))

    def check_processing_thread(self):
        """Check if processing thread is still running."""
        if self.processing_thread and self.processing_thread.is_alive():
            self.root.after(100, self.check_processing_thread)
        else:
            self.set_processing_state(False)

    def processing_complete(self, results):
        """Handle processing completion."""
        self.update_progress("\nProcessing completed successfully!")
        display_output_dir = os.path.normpath(results["output_dir"])
        self.update_progress(f"Results saved to: {display_output_dir}")

        self.results = results

        self.n_frames = results["n_frames"]
        self.slider.config(from_=1, to=max(1, self.n_frames))

        if self.n_frames < 10:
            self.slider_step = 1
        elif self.n_frames < 30:
            self.slider_step = max(1, self.n_frames // 5)
        else:
            self.slider_step = max(1, self.n_frames // 10)

        self.update_progress(
            f"Slider updated: {self.n_frames} frames available (range 1-{self.n_frames})"
        )
        self.update_progress(
            "Use the slider above the preview to view different frames."
        )

        self._display_kymographs(results)


        self.safe_showinfo(
            "Success",
            "Kymograph generation completed!\n\n"
            f"Total frames: {self.n_frames}\n"
            "Use the slider to browse through frames.\n"
            "Use 'Adjust Existing Kymograph' to modify visualization.",
        )

    def processing_error(self, error_msg):
        """Handle processing error."""
        self.update_progress(f"\nERROR: {error_msg}")
        self.safe_showerror("Processing Error", f"An error occurred:\n{error_msg}")

    def stop_processing(self):
        """Stop current processing."""
        if self.processor:
            self.processor.stop()
        self.update_progress("\nProcessing stopped by user.")

    def update_progress(self, message):
        """Update progress text (thread-safe)."""
        self.root.after(0, self._update_progress_text, message)
    
    def _update_progress_text(self, message):
        """Actually update the progress text (must run on main thread)."""
        self.text_widget.insert(tk.END, message + "\n")
        self.text_widget.see(tk.END)

    def update_preview(self, frame_idx, image_path):
        """Update preview image (thread-safe)."""
        self.current_frame = frame_idx
        self.root.after(0, self._update_canvas, image_path)

    def _update_canvas(self, image_path):
        """Update canvas with new image."""
        # Prevent recursion
        if self._updating_slider:
            return

        # Clear previous content
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        try:
            img = mpimg.imread(image_path)

            fig = Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            if platform.system() == 'Darwin':
                self.root.update()

            self._updating_slider = True
            self.slider.set(self.current_frame + 1)
            self._updating_slider = False

            if hasattr(self.processor, "n_frames") and self.processor.n_frames > 0:
                self.n_frames = self.processor.n_frames

            self.root.title(
                f"Membrane Kymograph Generator - Frame {self.current_frame + 1}/{self.n_frames}"
            )

        except Exception as e:
            print(f"Error updating canvas: {e}")
            import traceback

            traceback.print_exc()
            self._updating_slider = False

    def _display_kymographs(self, results):
        """Display generated kymographs in the preview area."""
        try:
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            
            plt.close('all')

            import numpy as np
            from matplotlib.colors import LinearSegmentedColormap
            from .parulamap import cm_data

            n_channels = results["n_channels"]
            output_dir = results["output_dir"]

            # Load kymograph images (PNG files)
            kymo_images = []
            for c in range(n_channels):
                img_path = os.path.join(output_dir, f"kymo_channel_{c+1}.png")
                if os.path.exists(img_path):
                    kymo_images.append(img_path)

            if not kymo_images:
                # Fallback: no images found, show message
                label = ttk.Label(
                    self.canvas_frame,
                    text="Kymographs generated successfully!\nImage files saved to output directory.",
                )
                label.pack(pady=20)
                return

            # Create figure with subplots for each channel
            if n_channels == 1:
                fig = Figure(figsize=(8, 5), dpi=100)
                ax = fig.add_subplot(111)
                img = mpimg.imread(kymo_images[0])
                ax.imshow(img)
                ax.axis("off")
                ax.set_title("Channel 1 Kymograph")
            else:
                # Multiple channels - show side by side
                fig = Figure(figsize=(12, 5), dpi=100)
                for idx, img_path in enumerate(kymo_images):
                    ax = fig.add_subplot(1, n_channels, idx + 1)
                    img = mpimg.imread(img_path)
                    ax.imshow(img)
                    ax.axis("off")
                    ax.set_title(f"Channel {idx + 1}")

            fig.tight_layout()

            # Display in canvas - store reference to prevent garbage collection
            self.kymo_canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            self.kymo_canvas.draw()
            self.kymo_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.update_progress("\nKymographs displayed in preview area.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Show error message in canvas
            self.root.after(0, lambda: self._show_display_error(str(e)))
    
    def _show_display_error(self, error_msg):
        """Show error message in canvas frame (must be called on main thread)."""
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        label = ttk.Label(
            self.canvas_frame, text=f"Error displaying kymographs:\n{error_msg}"
        )
        label.pack(pady=20)

    def on_slider_change(self, value):
        """Handle slider change."""

        if self._updating_slider:
            return

        try:
            frame_idx = int(float(value)) - 1

            if (
                hasattr(self.processor, "path_to_temp_save")
                and self.processor.path_to_temp_save
            ):
                image_path = os.path.join(
                    self.processor.path_to_temp_save, f"frame_{frame_idx + 1}.png"
                )
                if os.path.exists(image_path):
                    self.current_frame = frame_idx
                    self._update_canvas(image_path)
                else:
                    print(f"Frame image not found: {image_path}")
        except Exception as e:
            print(f"Error in slider change: {e}")

    def set_processing_state(self, is_processing):
        """Update UI state during processing."""
        state = tk.DISABLED if is_processing else tk.NORMAL
        self.run_button.config(state=state)
        self.browse_image_button.config(state=state)
        self.browse_mask_button.config(state=state)
        self.stop_button.config(state=tk.NORMAL if is_processing else tk.DISABLED)

    def validate_inputs(self):
        """Validate user inputs."""
        errors = []

        # Check file paths
        image_path = self.entry_image_path.get()
        mask_path = self.entry_mask_path.get()

        if not image_path:
            errors.append("Please select an image file")
        elif not os.path.exists(image_path):
            errors.append("Image file does not exist")

        if not mask_path:
            errors.append("Please select a mask file")
        elif not os.path.exists(mask_path):
            errors.append("Mask file does not exist")

        # Check numeric inputs
        try:
            l_perp = int(self.spinbox_l_perp.get())
            if l_perp < 1 or l_perp > 50:
                errors.append("Membrane width must be between 1 and 50 pixels")
        except ValueError:
            errors.append("Membrane width must be a number")

        try:
            n_channels = int(self.spinbox_channels.get())
            if n_channels < 1:
                errors.append("Number of channels must be at least 1")
        except ValueError:
            errors.append("Number of channels must be a number")

        if errors:
            self.safe_showerror("Validation Error", "\n".join(errors))
            return False
        return True

    def save_config(self):
        """Save current configuration."""
        filename = filedialog.asksaveasfilename(
            defaultextension=".ini",
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")],
        )
        if platform.system() == 'Darwin':
            self.root.update()
            
        if filename:
            config = {
                "image_path": self.entry_image_path.get(),
                "mask_path": self.entry_mask_path.get(),
                "l_perp": self.spinbox_l_perp.get(),
                "n_channels": self.spinbox_channels.get(),
                "colormap": self.colormap_var.get(),
                "save_png": str(self.save_png_var.get()),
                "save_svg": str(self.save_svg_var.get()),
                "save_pdf": str(self.save_pdf_var.get()),
            }
            save_config(filename, config)
            self.safe_showinfo("Success", "Configuration saved!")

    def load_config(self):
        """Load configuration from file."""
        filename = filedialog.askopenfilename(
            filetypes=[("INI files", "*.ini"), ("All files", "*.*")]
        )

        if platform.system() == 'Darwin':
            self.root.update()
            
        if filename:
            config = load_config(filename)
            if config:
                self.entry_image_path.delete(0, tk.END)
                self.entry_image_path.insert(0, config.get("image_path", ""))
                self.entry_mask_path.delete(0, tk.END)
                self.entry_mask_path.insert(0, config.get("mask_path", ""))
                self.spinbox_l_perp.set(config.get("l_perp", 8))
                self.spinbox_channels.set(config.get("n_channels", 1))
                self.colormap_var.set(config.get("colormap", "Default"))
                self.save_png_var.set(config.get("save_png", "True") == "True")
                self.save_svg_var.set(config.get("save_svg", "True") == "True")
                self.save_pdf_var.set(config.get("save_pdf", "False") == "True")

    def load_default_config(self):
        """Load default configuration if exists."""
        default_config_path = os.path.join(
            os.path.expanduser("~"), ".membrane_kymograph.ini"
        )
        if os.path.exists(default_config_path):
            config = load_config(default_config_path)
            if config:
                self.spinbox_l_perp.set(config.get("l_perp", 8))
                self.spinbox_channels.set(config.get("n_channels", 1))
                self.colormap_var.set(config.get("colormap", "Default"))
                self.save_png_var.set(config.get("save_png", "True") == "True")
                self.save_svg_var.set(config.get("save_svg", "True") == "True")
                self.save_pdf_var.set(config.get("save_pdf", "False") == "True")

    def on_closing(self):
        """Handle window closing."""
        # Check if processing is running before showing messagebox
        is_processing = self.processing_thread and self.processing_thread.is_alive()

        if is_processing:
            # Ask confirmation before destroying window
            try:
                response = self.safe_askyesno(
                    "Confirm Exit",
                    "Processing is still running. Do you want to stop and exit?",
                )
                if not response:
                    return

                # Stop processing
                self.processor.stop()
                if self.processing_thread:
                    self.processing_thread.join(timeout=2.0)
            except Exception as e:
                # If messagebox fails, just stop processing
                print(f"Error showing messagebox: {e}")
                self.processor.stop()

        # Destroy the window
        try:
            self.root.quit()
            self.root.destroy()
        except Exception as e:
            print(f"Error closing window: {e}")

    def open_adjust_limits_dialog(self):
        """Open a dialog to adjust color limits for existing kymograph files."""
        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Adjust Kymograph Color Limits")

        # Get screen dimensions for dialog sizing
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()

        # Calculate dialog size (80% of screen height, max 700x700)
        dialog_width = min(700, int(screen_width * 0.6))
        dialog_height = min(700, int(screen_height * 0.8))

        # Center dialog on screen
        x_pos = (screen_width - dialog_width) // 2
        y_pos = (screen_height - dialog_height) // 2

        dialog.geometry(f"{dialog_width}x{dialog_height}+{x_pos}+{y_pos}")
        dialog.minsize(600, 500)  # Set minimum size

        # Variables to store loaded data
        kymo_data = None
        current_kymo_path = tk.StringVar(value="")

        # Frame for file loading
        file_frame = ttk.LabelFrame(dialog, text="Load Kymograph File", padding=10)
        file_frame.pack(padx=10, pady=10, fill=tk.X)

        # Kymograph smooth file
        ttk.Label(file_frame, text="Kymograph (.npy):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        kymo_entry = ttk.Entry(file_frame, textvariable=current_kymo_path, width=50)
        kymo_entry.grid(row=0, column=1, padx=5, pady=5)

        def browse_kymo():
            filename = filedialog.askopenfilename(
                title="Select Kymograph File (kymo_channel_X_smooth.npy)",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            )
            if filename:
                current_kymo_path.set(filename)
                load_data()

        ttk.Button(file_frame, text="Browse", command=browse_kymo).grid(
            row=0, column=2, padx=5, pady=5
        )

        # Info label
        info_label = ttk.Label(
            file_frame,
            text="Load a kymograph file to adjust visualization limits",
            foreground="gray",
        )
        info_label.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        # Frame for controls
        control_frame = ttk.LabelFrame(dialog, text="Color Limit Controls", padding=10)
        control_frame.pack(padx=10, pady=10, fill=tk.X)

        # Auto-calculate button
        def auto_calculate_limits():
            """
            Auto-calculate reasonable visualization limits from kymograph data.
            Uses 5th-95th percentiles for better contrast.
            """
            if kymo_data is None:
                self.safe_showwarning("No Data", "Please load a kymograph file first.")
                return

            try:
                import numpy as np

                kymo_flat = kymo_data.flatten()
                kymo_flat = kymo_flat[~np.isnan(kymo_flat)]
                kymo_flat = kymo_flat[kymo_flat > 0]  # Exclude zeros

                if len(kymo_flat) > 0:

                    lower_pct = np.percentile(kymo_flat, 5)
                    upper_pct = np.percentile(kymo_flat, 95)

                    lower_limit_var.set(max(0.0, lower_pct))
                    upper_limit_var.set(min(1.0, upper_pct))

                    info_label.config(
                        text=f"Suggested from kymograph: 5%={lower_pct:.3f}, 95%={upper_pct:.3f}",
                        foreground="green",
                    )
                else:
                    self.safe_showerror(
                        "Error", "Kymograph data is empty or all zeros."
                    )
                    return

                update_preview()

            except Exception as e:
                self.safe_showerror("Error", f"Failed to calculate limits:\n{e}")

        ttk.Button(
            control_frame,
            text="Auto-Suggest Limits",
            command=auto_calculate_limits,
            style="info.TButton",
        ).grid(row=0, column=0, columnspan=4, pady=5)

        # Lower limit slider
        ttk.Label(control_frame, text="Lower Limit:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        lower_limit_var = tk.DoubleVar(value=0.0)
        lower_limit_scale = tk.Scale(
            control_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=lower_limit_var,
            length=300,
        )
        lower_limit_scale.grid(
            row=1, column=1, columnspan=2, padx=5, pady=5, sticky="ew"
        )
        lower_limit_label = ttk.Label(control_frame, text="0.00")
        lower_limit_label.grid(row=1, column=3, padx=5, pady=5)

        # Upper limit slider
        ttk.Label(control_frame, text="Upper Limit:").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        upper_limit_var = tk.DoubleVar(value=1.0)
        upper_limit_scale = tk.Scale(
            control_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=upper_limit_var,
            length=300,
        )
        upper_limit_scale.grid(
            row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew"
        )
        upper_limit_label = ttk.Label(control_frame, text="1.00")
        upper_limit_label.grid(row=2, column=3, padx=5, pady=5)

        # Colormap selection
        ttk.Label(control_frame, text="Colormap:").grid(
            row=3, column=0, padx=5, pady=5, sticky="w"
        )
        colormap_var = tk.StringVar(value="Default")
        colormaps = [
            "Default",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "Greys",
            "jet",
            "turbo",
            "gnuplot",
            "gnuplot2",
        ]
        colormap_menu = ttk.OptionMenu(
            control_frame, colormap_var, "Default", *colormaps
        )
        colormap_menu.grid(row=3, column=1, padx=5, pady=5, sticky="w")



        def update_labels(*args):
            lower_limit_label.config(text=f"{lower_limit_var.get():.2f}")
            upper_limit_label.config(text=f"{upper_limit_var.get():.2f}")

        lower_limit_var.trace("w", update_labels)
        upper_limit_var.trace("w", update_labels)

        preview_frame = ttk.LabelFrame(dialog, text="Kymograph Preview", padding=5)
        preview_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        preview_canvas_frame = ttk.Frame(preview_frame)
        preview_canvas_frame.pack(fill=tk.BOTH, expand=True)

        def load_data():
            """Load the kymograph data."""
            nonlocal kymo_data

            kymo_path = current_kymo_path.get()
            if not kymo_path or not os.path.exists(kymo_path):
                return

            try:
                import numpy as np

                kymo_data = np.load(kymo_path)
                info_label.config(
                    text=f"Loaded kymograph: {kymo_data.shape}", foreground="blue"
                )

                update_preview()
            except Exception as e:
                self.safe_showerror("Error", f"Failed to load data:\n{e}")
                info_label.config(text=f"Error loading data: {e}", foreground="red")

        def update_preview():
            """Update the kymograph preview with current limits."""
            if kymo_data is None:
                return

            for widget in preview_canvas_frame.winfo_children():
                widget.destroy()
            
            # Close any old figures to prevent memory leaks
            plt.close('all')

            try:
                import numpy as np
                from matplotlib.colors import LinearSegmentedColormap
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

                colormap = colormap_var.get()
                if colormap == "Default":
                    from .parulamap import cm_data

                    cmap = LinearSegmentedColormap.from_list("parula", cm_data)
                else:
                    cmap = plt.get_cmap(colormap)


                fig = Figure(figsize=(8, 5), dpi=100)
                ax = fig.add_subplot(111)

                im = ax.imshow(
                    kymo_data,
                    aspect="auto",
                    cmap=cmap,
                    vmin=lower_limit_var.get(),
                    vmax=upper_limit_var.get(),
                )
                ax.set_xlabel("Frame")

                n_positions = kymo_data.shape[0]

                angle_ticks_positions = np.linspace(
                    0, n_positions - 1, 9
                )  
                angle_ticks_labels = [
                    "π",
                    "3π/4",
                    "π/2",
                    "π/4",
                    "0",
                    "-π/4",
                    "-π/2",
                    "-3π/4",
                    "-π",
                ]
                ax.set_yticks(angle_ticks_positions)
                ax.set_yticklabels(angle_ticks_labels)
                ax.set_ylabel("Angle")

                ax.set_title("Kymograph Preview")
                fig.colorbar(im, ax=ax, label="Normalized Intensity")

                fig.tight_layout()


                canvas = FigureCanvasTkAgg(fig, master=preview_canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            except Exception as e:
                error_label = ttk.Label(
                    preview_canvas_frame, text=f"Error displaying kymograph: {e}"
                )
                error_label.pack(pady=20)

        def save_kymograph():
            """Save the kymograph with current limits."""
            if kymo_data is None:
                self.safe_showwarning("No Data", "Please load a kymograph file first.")
                return

            try:
                import numpy as np
                from matplotlib.colors import LinearSegmentedColormap

                lower_limit = lower_limit_var.get()
                upper_limit = upper_limit_var.get()

                # Ask where to save
                output_path = filedialog.asksaveasfilename(
                    title="Save Adjusted Kymograph",
                    defaultextension=".png",
                    filetypes=[
                        ("PNG files", "*.png"),
                        ("PDF files", "*.pdf"),
                        ("SVG files", "*.svg"),
                        ("All files", "*.*"),
                    ],
                )

                if not output_path:
                    return

                # Get colormap
                colormap = colormap_var.get()
                if colormap == "Default":
                    from .parulamap import cm_data

                    cmap = LinearSegmentedColormap.from_list("parula", cm_data)
                else:
                    cmap = plt.get_cmap(colormap)

                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                im = ax.imshow(
                    kymo_data,
                    aspect="auto",
                    cmap=cmap,
                    vmin=lower_limit,
                    vmax=upper_limit,
                )
                ax.set_xlabel("Frame")

                n_positions = kymo_data.shape[0]

                angle_ticks_positions = np.linspace(
                    0, n_positions - 1, 9
                )  
                angle_ticks_labels = [
                    "π",
                    "3π/4",
                    "π/2",
                    "π/4",
                    "0",
                    "-π/4",
                    "-π/2",
                    "-3π/4",
                    "-π",
                ]
                ax.set_yticks(angle_ticks_positions)
                ax.set_yticklabels(angle_ticks_labels)
                ax.set_ylabel("Angle")

                ax.set_title("Kymograph")
                fig.colorbar(im, ax=ax, label="Normalized Intensity")

                fig.tight_layout()

                # Determine format from extension
                ext = os.path.splitext(output_path)[1].lower()
                if ext == ".pdf":
                    fig.savefig(output_path, format="pdf", bbox_inches="tight")
                elif ext == ".svg":
                    fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
                else:
                    fig.savefig(output_path, format="png", dpi=300, bbox_inches="tight")

                # Close the figure to free memory
                plt.close(fig)

                # Convert path to OS-specific format for display
                display_path = os.path.normpath(output_path)

                self.safe_showinfo(
                    "Success",
                    f"Saved adjusted kymograph to:\n{display_path}\n\n"
                    f"Limits: [{lower_limit:.2f}, {upper_limit:.2f}]",
                )

            except Exception as e:
                self.safe_showerror("Error", f"Failed to save kymograph:\n{e}")

        # Button frame
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=4, column=0, columnspan=4, pady=10)

        ttk.Button(
            button_frame,
            text="Update Preview",
            command=update_preview,
            style="primary.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame,
            text="Save Kymograph",
            command=save_kymograph,
            style="success.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame,
            text="Close",
            command=dialog.destroy,
            style="secondary.TButton",
        ).pack(side=tk.LEFT, padx=5)

    def open_correlation_dialog(self):
        """Open a dialog for correlation analysis between two kymographs."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Kymograph Correlation Analysis")

        # Get screen dimensions for dialog sizing
        screen_width = dialog.winfo_screenwidth()
        screen_height = dialog.winfo_screenheight()

        # Calculate dialog size
        dialog_width = min(800, int(screen_width * 0.7))
        dialog_height = min(600, int(screen_height * 0.7))

        # Center dialog on screen
        x_pos = (screen_width - dialog_width) // 2
        y_pos = (screen_height - dialog_height) // 2

        dialog.geometry(f"{dialog_width}x{dialog_height}+{x_pos}+{y_pos}")
        dialog.minsize(700, 500)

        # Variables
        kymo1_path = tk.StringVar(value="")
        kymo2_path = tk.StringVar(value="")
        kymo1_name = tk.StringVar(value="Channel 1")
        kymo2_name = tk.StringVar(value="Channel 2")
        exclude_zeros_var = tk.BooleanVar(value=True)

        # File loading frame
        file_frame = ttk.LabelFrame(dialog, text="Load Kymograph Files", padding=10)
        file_frame.pack(padx=10, pady=10, fill=tk.X)


        ttk.Label(file_frame, text="Kymograph 1 (..._smooth.npy):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        kymo1_entry = ttk.Entry(file_frame, textvariable=kymo1_path, width=50)
        kymo1_entry.grid(row=0, column=1, padx=5, pady=5)

        def browse_kymo1():
            filename = filedialog.askopenfilename(
                title="Select First Kymograph File",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            )
            if filename:
                kymo1_path.set(filename)
                basename = os.path.basename(filename).replace(".npy", "")
                if "channel" in basename.lower():
                    kymo1_name.set(basename)

        ttk.Button(file_frame, text="Browse", command=browse_kymo1).grid(
            row=0, column=2, padx=5, pady=5
        )


        ttk.Label(file_frame, text="Name:").grid(
            row=1, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Entry(file_frame, textvariable=kymo1_name, width=30).grid(
            row=1, column=1, padx=5, pady=5, sticky="w"
        )


        ttk.Label(file_frame, text="Kymograph 2 (..._smooth.npy):").grid(
            row=2, column=0, padx=5, pady=5, sticky="w"
        )
        kymo2_entry = ttk.Entry(file_frame, textvariable=kymo2_path, width=50)
        kymo2_entry.grid(row=2, column=1, padx=5, pady=5)

        def browse_kymo2():
            filename = filedialog.askopenfilename(
                title="Select Second Kymograph File",
                filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
            )
            if filename:
                kymo2_path.set(filename)
                # Auto-populate name from filename
                basename = os.path.basename(filename).replace(".npy", "")
                if "channel" in basename.lower():
                    kymo2_name.set(basename)

        ttk.Button(file_frame, text="Browse", command=browse_kymo2).grid(
            row=2, column=2, padx=5, pady=5
        )

        ttk.Label(file_frame, text="Name:").grid(
            row=3, column=0, padx=5, pady=5, sticky="w"
        )
        ttk.Entry(file_frame, textvariable=kymo2_name, width=30).grid(
            row=3, column=1, padx=5, pady=5, sticky="w"
        )

        options_frame = ttk.LabelFrame(dialog, text="Analysis Options", padding=10)
        options_frame.pack(padx=10, pady=10, fill=tk.X)

        ttk.Checkbutton(
            options_frame,
            text="Exclude zero values from correlation",
            variable=exclude_zeros_var,
        ).pack(anchor="w", padx=5, pady=5)


        note_label = ttk.Label(options_frame, text="", foreground="gray")
        note_label.pack(anchor="w", padx=5, pady=2)

        def update_note_label(*args):
            """Update the note text based on checkbox state."""
            if exclude_zeros_var.get():
                note_label.config(
                    text="Note: Zero values and NaN values will be excluded from analysis."
                )
            else:
                note_label.config(
                    text="Note: Only NaN values will be excluded from analysis."
                )

        update_note_label()
        exclude_zeros_var.trace("w", update_note_label)

        status_frame = ttk.LabelFrame(dialog, text="Status", padding=10)
        status_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        status_text = tk.Text(status_frame, height=10, width=70, wrap=tk.WORD)
        status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        status_scrollbar = ttk.Scrollbar(status_frame, command=status_text.yview)
        status_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        status_text.config(yscrollcommand=status_scrollbar.set)

        def log_status(message):
            """Add message to status text."""
            status_text.insert(tk.END, message + "\n")
            status_text.see(tk.END)

        log_status("Ready. Select two kymograph files to analyze.")

        def run_correlation():
            """Run the correlation analysis."""
            # Validate inputs
            if not kymo1_path.get() or not kymo2_path.get():
                self.safe_showwarning(
                    "Missing Files", "Please select both kymograph files."
                )
                return

            if not os.path.exists(kymo1_path.get()):
                self.safe_showerror(
                    "File Not Found", f"Kymograph 1 not found:\n{kymo1_path.get()}"
                )
                return

            if not os.path.exists(kymo2_path.get()):
                self.safe_showerror(
                    "File Not Found", f"Kymograph 2 not found:\n{kymo2_path.get()}"
                )
                return

            try:
                log_status("\n" + "=" * 70)
                log_status("Starting correlation analysis...")
                log_status(f"Kymograph 1: {os.path.basename(kymo1_path.get())}")
                log_status(f"Kymograph 2: {os.path.basename(kymo2_path.get())}")
                log_status(f"Exclude zeros: {exclude_zeros_var.get()}")

                # Create correlation analyzer
                corr_analyzer = KymographCorrelation()

                output_dir = os.path.join(
                    os.path.dirname(kymo1_path.get()), "Correlation_Analysis"
                )

                log_status(f"\nOutput directory: {output_dir}")

                # Run analysis
                results = corr_analyzer.run_full_analysis(
                    kymo1_path.get(),
                    kymo2_path.get(),
                    output_dir,
                    exclude_zeros=exclude_zeros_var.get(),
                    kymo1_name=kymo1_name.get(),
                    kymo2_name=kymo2_name.get(),
                )

                log_status("\n✓ Analysis complete!")
                log_status(f"✓ Processed {results['n_frames']} frames")
                log_status(f"✓ Excel file: {os.path.basename(results['excel_file'])}")
                log_status(
                    f"✓ Correlation plot: {os.path.basename(results['correlation_plot'])}"
                )
                log_status(
                    f"✓ P-value plot: {os.path.basename(results['pvalue_plot'])}"
                )

                # Show summary statistics
                df = results["dataframe"]
                log_status("\n--- Summary Statistics ---")
                log_status(
                    f"Pearson R: mean={df['Pearson_R'].mean():.3f}, "
                    f"std={df['Pearson_R'].std():.3f}"
                )
                log_status(
                    f"Spearman R: mean={df['Spearman_R'].mean():.3f}, "
                    f"std={df['Spearman_R'].std():.3f}"
                )

                # Convert paths to OS-specific format
                display_excel = os.path.normpath(results["excel_file"])
                display_dir = os.path.normpath(output_dir)

                self.safe_showinfo(
                    "Analysis Complete",
                    f"Correlation analysis completed successfully!\n\n"
                    f"Frames analyzed: {results['n_frames']}\n"
                    f"Results saved to:\n{display_dir}\n\n"
                    f"Files created:\n"
                    f"• correlation_analysis.xlsx\n"
                    f"• correlation_coefficients.png (and .pdf)\n"
                    f"• correlation_pvalues.png (and .pdf)",
                )

            except ValueError as e:
                log_status(f"\n✗ ERROR: {e}")
                self.safe_showerror("Validation Error", str(e))
            except Exception as e:
                log_status(f"\n✗ ERROR: {e}")
                import traceback

                traceback.print_exc()
                self.safe_showerror("Error", f"An error occurred:\n{e}")

        # Button frame
        button_frame = ttk.Frame(dialog)
        button_frame.pack(padx=10, pady=10, fill=tk.X)

        ttk.Button(
            button_frame,
            text="Run Analysis",
            command=run_correlation,
            style="success.TButton",
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            button_frame,
            text="Close",
            command=dialog.destroy,
            style="secondary.TButton",
        ).pack(side=tk.LEFT, padx=5)

    def run(self):
        """Start the GUI."""
        # Check if GUI was actually initialized
        if not hasattr(self, 'root') or self.root is None:
            return
        
        # macOS-specific: Ensure proper cleanup
        def on_closing():
            """Handle window close event - cross-platform."""
            try:
                # Stop any running processing
                if hasattr(self, 'processor') and self.processor:
                    self.processor.stop()
                
                if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
                    # Wait briefly for thread to finish
                    self.processing_thread.join(timeout=1.0)
                
                # Close all matplotlib figures
                plt.close('all')
                
                # Destroy canvas widgets
                if hasattr(self, 'kymo_canvas') and self.kymo_canvas:
                    try:
                        self.kymo_canvas.get_tk_widget().destroy()
                    except:
                        pass
                
                # Quit and destroy root window
                try:
                    self.root.quit()
                except:
                    pass
                
                try:
                    self.root.destroy()
                except:
                    pass
                
            except Exception as e:
                print(f"Error during cleanup: {e}")
            finally:
                # Force exit on macOS if needed
                if platform.system() == 'Darwin':
                    sys.exit(0)
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()


def _check_main_already_running():
    """Check if main is already running using environment variable."""
    return os.environ.get('MEMBRANE_KYMO_GUI_RUNNING') == '1'

def _set_main_running():
    """Mark main as running."""
    os.environ['MEMBRANE_KYMO_GUI_RUNNING'] = '1'


def main():
    """Main entry point for GUI."""
    
    # Prevent multiple calls to main() using environment variable
    if _check_main_already_running():
        return
    
    _set_main_running()
    
    if platform.system() == 'Darwin':
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
    
    app = KymographGUI()
    app.run()


if __name__ == "__main__":
    main()
