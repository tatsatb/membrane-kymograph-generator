# Membrane Kymograph Generator


<p align="center">
<img src="https://raw.githubusercontent.com/tatsatb/membrane-kymograph-generator/233f5c6074246d4c9a667ca47ee3ee1140c5786c/icons/ICON.svg" width="280" alt="Membrane Kymograph Generator Icon">
<br/><br/>
</p>


_Membrane Kymograph Generator_, is a cross-platform, free and open-source, GUI-based application for generating kymographs from live-cell microscopy images along dynamic cell boundaries. Starting from a time-lapse image sequence and a whole cell binary mask, it automatically extracts the boundaries, corrects for the changes in the shape and size of the boundaries (using a custom algorithm), properly aligns the boundaries between different frames, samples intensities across the boundaries, and finally generates publication-quality kymographs in various formats. 

---

## ✅ 📚 Please visit the [Wiki](https://github.com/tatsatb/membrane-kymograph-generator/wiki/Homepage) for detailed documentation. 

---


## 🚀 Quick Start

**[Download the latest release](https://github.com/tatsatb/membrane-kymograph-generator/releases)** and check out the **[Quick Start Guide](https://github.com/tatsatb/membrane-kymograph-generator/wiki/Homepage)** to begin generating kymographs in minutes!

**No Python installation, dependency management, or programming knowledge is required - just download, install (or extract, if you are using standalone binaries), and run!**

---

## 🎁 Out of the box features

- **User-Friendly GUI**: An intuitive, interactive interface built with ttkbootstrap which allows easy navigation and parameter adjustments. 
- **Robust Input Validation**: TIFF format checking and dimension compatibility verification. 
- **Flexible Output Formats**: Save kymographs as PNG, SVG, and/or PDF
- **Multi-Channel Support**: Process multiple fluorescence channels simultaneously. 
- **Efficient Processing**:
  - Automatic boundary detection and smoothing via custom rotational offset algorithm. 
  - Circle fitting for perpendicular line generation
  - LOWESS smoothing for intensity profiles
  - Parallel processing via `joblib` for improved performance
- **Comprehensive Data Export**:
  - NumPy arrays (.npy) for all intermediate data
  - Excel files (.xlsx) for centroid positions and kymograph data
  - High-quality vector and raster visualizations
- **Interactive Post-Processing**: 
  - Fine-tune color limits after processing
  - Built-in correlation analysis tool (and statistical tests) for kymograph data
- **Native Python API**: For advanced users to enable batch processing (of large-scale datasets) and custom downstream spatiotemporal analyses (examples are provided in the [Wiki](https://github.com/tatsatb/membrane-kymograph-generator/wiki/Advanced-Usage:-Python-API)).
- **Cross-Platform Compatibility**: Available for both x86-64 and ARM64 systems running either Windows, macOS, and any standard Linux distribution. No installation of Python or any dependencies required. 

- **Open Source**: Free and open-source under the GPL v3 License.

---

## 🔗 Links


- 🔽 To download the latest version of the software, please visit the [Releases](https://github.com/tatsatb/membrane-kymograph-generator/releases).

- 📄 To learn more about the features, capabilities, and inner workings of the software, please read the [Preprint](#) (coming soon 🔜) .

- 📚 For detailed documentation on how to use this software, please visit the [wiki](https://github.com/tatsatb/membrane-kymograph-generator/wiki/Homepage). 

- 💻 To access the source code, report issues, or contribute to the development of the software, please visit the [GitHub repository](https://github.com/tatsatb/membrane-kymograph-generator).

- 🐞 To report issues or request features, please use the [GitHub Issues](https://github.com/tatsatb/membrane-kymograph-generator/issues) page.


---

## 📖 Citation

If you use this software in your research, please cite:

> Membrane Kymograph Generator: A cross-platform GUI software for automated generation and analysis of kymographs along dynamic cell boundaries. [Preprint coming soon 🔜]

---

## 🙏 Acknowledgments

Developed at the Iglesias Lab and Devreotes Lab at Johns Hopkins University for analyzing membrane dynamics in dynamic cell physiological processes. We thank all members of both labs for their valuable feedback during the development and testing phases.

---

## ⚖️ License

Copyright © 2025 Tatsat Banerjee, Bedri Abubaker-Sharif, Peter N. Devreotes, and Pablo A. Iglesias.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 
