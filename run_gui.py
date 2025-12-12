#!/usr/bin/env python
"""
Launcher script for Membrane Kymograph GUI.
This script is the entry point for the PyInstaller executable.
"""

if __name__ == "__main__":
    # Set matplotlib backend early to prevent window creation issues
    import os
    import sys
    
    # Detect platform
    import platform
    
    # Set environment variable before ANY matplotlib import
    
    os.environ['MPLBACKEND'] = 'Agg'
    
    if platform.system() == 'Darwin':
        os.environ['TK_SILENCE_DEPRECATION'] = '1'
    
    # Pre-initialize matplotlib to build font cache BEFORE any GUI starts
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    plt.ioff()
    # Trigger font cache build now (silently)
    try:
        from matplotlib.font_manager import fontManager
        fontManager._get_font_list()
    except:
        pass
    
    # Now import and run main (which has all the matplotlib configuration)
    from membrane_kymograph.gui import main
    main()
