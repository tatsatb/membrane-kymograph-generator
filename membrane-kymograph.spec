# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Membrane Kymograph Generator
This creates a onedir executable with all dependencies bundled.
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os
import platform

# Get version dynamically from git tags
def get_app_version():
    """Get version from git tags."""
    import subprocess
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        # Remove 'v' prefix if present (e.g., v0.0.45 -> 0.0.45)
        version = version.lstrip("v")
        print(f"Building version: {version}")
        
        # Create _version.py file to bundle with the app
        version_file = os.path.join('membrane_kymograph', '_version.py')
        with open(version_file, 'w') as f:
            f.write(f'__version__ = "{version}"\n')
        print(f"Created {version_file} with version {version}")
        
        return version
    except Exception as e:
        print(f"Warning: Could not get version from git: {e}")
        return "0.0.1"

APP_VERSION = get_app_version()

block_cipher = None

# Determine icon file based on platform
if platform.system() == 'Windows':
    icon_file = 'icons/memkymo.ico'
elif platform.system() == 'Darwin':  # macOS
    icon_file = 'icons/memkymo.icns'
else:  # Linux
    icon_file = 'icons/memkymo.png'

# Check if icon exists, otherwise use None
if not os.path.exists(icon_file):
    print(f"Warning: Icon file {icon_file} not found. Building without custom icon.")
    icon_file = None

# Collect all data files from packages that need them
datas = []

# Add the generated _version.py file
version_file_path = os.path.join('membrane_kymograph', '_version.py')
if os.path.exists(version_file_path):
    datas.append((version_file_path, 'membrane_kymograph'))
    print(f"Added {version_file_path} to bundle")

# Add ttkbootstrap themes and assets
datas += collect_data_files('ttkbootstrap')

# Add example files from project root
example_dir = 'example_data_config'
if os.path.exists(example_dir):
    datas.append((example_dir, '.'))

# Add requirements.txt from project root
if os.path.exists('requirements.txt'):
    datas.append(('requirements.txt', '.'))

# Collect all submodules
hiddenimports = [
    # Main package modules
    'membrane_kymograph',
    'membrane_kymograph.gui',
    'membrane_kymograph.processor',
    'membrane_kymograph.kymohelpers',
    'membrane_kymograph.utils',
    'membrane_kymograph.config',
    'membrane_kymograph.parulamap',
    
    # tkinter and GUI
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.font',
    'ttkbootstrap',
    'ttkbootstrap.constants',
    
    # macOS native APIs for focus management
    'AppKit',
    'Foundation',
    'objc',
    
    # Matplotlib backends - include ALL backends for saving different formats
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg',
    'matplotlib.backends.backend_svg',
    'matplotlib.backends.backend_pdf',
    'matplotlib.backends.backend_ps',
    'matplotlib.backends.backend_pgf',
    
    # NumPy and SciPy
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.core._dtype_ctypes',
    'scipy',
    'scipy.ndimage',
    'scipy.interpolate',
    'scipy.spatial',
    'scipy.spatial.transform',
    'scipy.sparse',
    'scipy.sparse.csgraph',
    
    # Pandas
    'pandas',
    'pandas._libs',
    'pandas._libs.tslibs',
    
    # Image processing
    'cv2',
    'skimage',
    'skimage.measure',
    'skimage.draw',
    'PIL',
    'PIL._tkinter_finder',
    'tifffile',
    
    # Statistical models
    'statsmodels',
    'statsmodels.nonparametric',
    'statsmodels.nonparametric.smoothers_lowess',
    
    # Other dependencies
    'shapely',
    'shapely.geometry',
    'circle_fit',
    'joblib',
    'openpyxl',
    'openpyxl.cell',
    'openpyxl.styles',
    
    # Required for some packages
    'packaging',
    'dateutil',
    'pytz',
    
    # pkg_resources dependencies (fixes jaraco.text error)
    'jaraco',
    'jaraco.text',
    'jaraco.functools',
    'jaraco.context',
    'more_itertools',
]

# Add all statsmodels submodules
hiddenimports += collect_submodules('statsmodels')

# Add all scikit-image submodules
hiddenimports += collect_submodules('skimage')

# Add all matplotlib backends to ensure SVG, PDF, etc. work
hiddenimports += collect_submodules('matplotlib.backends')

# Binaries - PyInstaller should auto-detect most, but we can be explicit
binaries = []

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'jupyter',
        'notebook',
        'IPython',
        'pytest',
        'sphinx',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='membrane-kymograph',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window (GUI only)
    disable_windowed_traceback=False,
    argv_emulation=True,  
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,  # Platform-specific icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='membrane-kymograph',
)

# macOS-specific: Create a proper .app bundle
if platform.system() == 'Darwin':
    app = BUNDLE(
        coll,
        name='Membrane Kymograph.app',
        icon=icon_file,
        bundle_identifier='org.devreoteslab.membrane-kymograph',
        version=APP_VERSION,
        info_plist={
            'CFBundleDevelopmentRegion': 'en',
            'CFBundleDisplayName': 'Membrane Kymograph',
            'CFBundleExecutable': 'membrane-kymograph',
            'CFBundleIdentifier': 'in.tatsatbanerjee.membrane-kymograph',
            'CFBundleInfoDictionaryVersion': '6.0',
            'CFBundleName': 'Membrane Kymograph',
            'CFBundlePackageType': 'APPL',
            'CFBundleShortVersionString': APP_VERSION,
            'CFBundleVersion': APP_VERSION,
            'LSMinimumSystemVersion': '10.13',
            'NSHighResolutionCapable': True,
            'NSRequiresAquaSystemAppearance': False,
            'NSPrincipalClass': 'NSApplication',
            'LSApplicationCategoryType': 'public.app-category.education',
            'LSBackgroundOnly': False,
            'LSUIElement': False,
            'NSAppTransportSecurity': {'NSAllowsArbitraryLoads': True},
        },
)
