"""
setup.py script for membrane-kymograph package.
"""

from setuptools import setup, find_packages
import subprocess
import os


def get_version():
    """Get version from git tag, version file, or fall back to default."""
    version_file = os.path.join(
        os.path.dirname(__file__), "membrane_kymograph", "_version.py"
    )

    # First, try to get version from git tag
    try:
        version = subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        version = version.lstrip("v")

        # Write version to file for isolated builds
        with open(version_file, "w") as f:
            f.write(f'__version__ = "{version}"\n')

        return version
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Second, try to read from version file (for isolated builds)
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            content = f.read()
            for line in content.split("\n"):
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")

    # Fall back to default version
    return "0.0.1"


# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='membrane-kymograph',
    version=get_version(),
    author='Tatsat Banerjee',
    download_url='https://github.com/tatsatb/membrane-kymograph-generator/releases/latest',
    description='A GUI-based cross-platform tool for generating membrane kymographs from live-cell time-lapse microscopy images',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tatsatb/membrane-kymograph-generator',
    packages=find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent',
    ],
    license='GPL-3.0-or-later',  
    python_requires='>=3.10,<3.14',
    install_requires=[
        'circle-fit==0.2.1',
        'contourpy==1.3.2',
        'cycler==0.12.1',
        'et_xmlfile==2.0.0',
        'fonttools==4.60.1',
        'future==1.0.0',
        'imageio==2.37.0',
        'joblib==1.5.2',
        'kiwisolver==1.4.9',
        'lazy_loader==0.4',
        'matplotlib==3.10.7',
        'networkx==3.4.2',
        'numpy==2.2.6',
        'opencv-python==4.12.0.88',
        'openpyxl==3.1.5',
        'packaging==25.0',
        'pandas==2.3.3',
        'patsy==1.0.2',
        'pillow==11.3.0',
        'pyparsing==3.2.5',
        'python-dateutil==2.9.0.post0',
        'pytz==2025.2',
        'scikit-image==0.25.2',
        'scikit-learn==1.7.2',
        'scipy==1.15.3',
        'seaborn==0.13.2',
        'shapely==2.1.2',
        'six==1.17.0',
        'statsmodels==0.14.5',
        'threadpoolctl==3.6.0',
        'tifffile==2025.5.10',
        'tiffile==2018.10.18',
        'ttkbootstrap==1.16.0',
        'tzdata==2025.2'
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'black>=22.0',
            'flake8>=5.0',
            'mypy>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'membrane-kymograph=membrane_kymograph.gui:main',
        ],
    },
    include_package_data=True,
    package_data={
        'membrane_kymograph': [
            'requirements.txt',
        ],
    },
    keywords='microscopy kymograph cell-biology fluorescence image-processing membrane-dynamics',
    project_urls={
        'Bug Reports': 'https://github.com/tatsatb/membrane-kymograph-generator/issues',
        'Source': 'https://github.com/tatsatb/membrane-kymograph-generator',
    },
)
