"""
Membrane Kymograph Generator

A tool for automatically generating kymographs along dynamic cell boundaries from multichannel live-cell microscopy images.
"""

import subprocess
import os


def get_version():
    """Get version from git tag, version file, or fall back to default."""
    # Try to import from _version.py file first (created during build)
    try:
        from . import _version

        return _version.__version__
    except ImportError:
        pass

    # Try to get version from git tag
    try:
        git_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=git_dir,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Remove 'v' prefix (e.g., v0.0.26 -> 0.0.26)
        return version.lstrip("v")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Fall back to default version if git is not available or no tags exist
    return "0.0.1"


__version__ = get_version()
__author__ = "Tatsat Banerjee"


from .processor import KymographProcessor
from .gui import main as gui_main

__all__ = ["KymographProcessor", "gui_main"]
