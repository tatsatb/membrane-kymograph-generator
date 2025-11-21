# PyInstaller hook for matplotlib backends
# Ensures all necessary matplotlib backends are included

from PyInstaller.utils.hooks import collect_submodules

# Collect all matplotlib backend modules
hiddenimports = collect_submodules('matplotlib.backends')

# Explicitly include the backends we need
hiddenimports += [
    'matplotlib.backends.backend_tkagg',
    'matplotlib.backends.backend_agg', 
    'matplotlib.backends.backend_svg',
    'matplotlib.backends.backend_pdf',
    'matplotlib.backends.backend_ps',
]
