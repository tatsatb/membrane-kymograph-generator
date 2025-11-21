#!/bin/bash

# create_appimage.sh - Create Linux AppImage from PyInstaller output

# Requires: appimagetool (download from https://appimage.github.io/appimagetool/)

set -e

# Configuration
APP_NAME="Membrane Kymograph"
APP_ID="in.tatsatbanerjee.membrane-kymograph"
#Placeholder version; will be automatically updated during build process
VERSION="0.0.1"
PYINSTALLER_DIR="dist/membrane-kymograph"
APPDIR="dist/AppDir"
APPIMAGE_DIR="installers"

# Detect architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    APPIMAGE_NAME="membrane-kymograph-${VERSION}-linux-aarch64.AppImage"
    APPIMAGETOOL_ARCH="aarch64"
else
    APPIMAGE_NAME="membrane-kymograph-${VERSION}-linux-x86_64.AppImage"
    APPIMAGETOOL_ARCH="x86_64"
fi

echo "Creating Linux AppImage..."
echo "Architecture: ${ARCH}"
echo "Source: ${PYINSTALLER_DIR}"
echo "Output: ${APPIMAGE_DIR}/${APPIMAGE_NAME}"

# Create installers directory
mkdir -p "${APPIMAGE_DIR}"

# Check if PyInstaller output exists
if [ ! -d "${PYINSTALLER_DIR}" ]; then
    echo "Error: PyInstaller output directory not found at ${PYINSTALLER_DIR}"
    echo "Please run PyInstaller first:"
    echo "  pyinstaller membrane-kymograph.spec --clean --noconfirm"
    exit 1
fi

# Create AppDir structure
echo "Creating AppDir structure..."
rm -rf "${APPDIR}"
mkdir -p "${APPDIR}/usr/bin"
mkdir -p "${APPDIR}/usr/lib"
mkdir -p "${APPDIR}/usr/share/applications"
mkdir -p "${APPDIR}/usr/share/icons/hicolor/256x256/apps"
mkdir -p "${APPDIR}/usr/share/metainfo"

# Copy PyInstaller output
echo "Copying application files..."
cp -R "${PYINSTALLER_DIR}"/* "${APPDIR}/usr/bin/"

# Create a simple wrapper script
echo "Creating wrapper script..."
cat > "${APPDIR}/usr/bin/membrane-kymograph-wrapper" << 'EOF'
#!/bin/bash
# AppImage wrapper script for Membrane Kymograph

# Get the directory where this script is located
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set library path to include bundled libraries
export LD_LIBRARY_PATH="${DIR}:${DIR}/lib:${LD_LIBRARY_PATH}"

# Set Qt/GTK theme environment variables
export QT_QPA_PLATFORMTHEME=gtk3
export GDK_BACKEND=x11

# Run the application
exec "${DIR}/membrane-kymograph" "$@"
EOF

chmod +x "${APPDIR}/usr/bin/membrane-kymograph-wrapper"

# Create desktop file
echo "Creating desktop entry..."
cat > "${APPDIR}/usr/share/applications/${APP_ID}.desktop" << EOF
[Desktop Entry]
Type=Application
Name=${APP_NAME}
GenericName=Membrane Kymograph Generator
Comment=A GUI-based cross-platform tool for generating membrane kymographs from live-cell time-lapse microscopy images
Exec=membrane-kymograph-wrapper
Icon=${APP_ID}
Categories=Science;Education;DataVisualization;Biology;
Terminal=false
StartupNotify=true
StartupWMClass=Tk
Keywords=microscopy;kymograph;cell-biology;fluorescence;image-processing;
EOF

# Copy desktop file to AppDir root (as required for AppImage)
cp "${APPDIR}/usr/share/applications/${APP_ID}.desktop" "${APPDIR}/${APP_ID}.desktop"

# Copy custom icon or create a minimal one
ICON_FILE="icons/memkymo.png"
if [ -f "${ICON_FILE}" ]; then
    echo "Copying custom application icon..."
    cp "${ICON_FILE}" "${APPDIR}/usr/share/icons/hicolor/256x256/apps/${APP_ID}.png"
    echo "✓ Custom icon copied from ${ICON_FILE}"
else
    echo "Warning: Icon file ${ICON_FILE} not found. Creating minimal fallback icon..."
    # Create a minimal 256x256 transparent PNG
    printf '\x89\x50\x4e\x47\x0d\x0a\x1a\x0a\x00\x00\x00\x0d\x49\x48\x44\x52\x00\x00\x01\x00\x00\x00\x01\x00\x08\x06\x00\x00\x00\x5c\x72\xa8\x66\x00\x00\x00\x0a\x49\x44\x41\x54\x78\x9c\x63\x00\x01\x00\x00\x05\x00\x01\x0d\x0a\x2d\xb4\x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82' > "${APPDIR}/usr/share/icons/hicolor/256x256/apps/${APP_ID}.png"
    echo "✓ Minimal fallback icon created (replace ${ICON_FILE} with a custom 256x256 PNG icon)"
fi

# Copy icon to AppDir root (as required for AppImage)
cp "${APPDIR}/usr/share/icons/hicolor/256x256/apps/${APP_ID}.png" "${APPDIR}/${APP_ID}.png"
# Also create .DirIcon (alternative AppImage icon format)
cp "${APPDIR}/usr/share/icons/hicolor/256x256/apps/${APP_ID}.png" "${APPDIR}/.DirIcon"

# Create AppStream metadata
echo "Creating AppStream metadata..."
cat > "${APPDIR}/usr/share/metainfo/${APP_ID}.appdata.xml" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<component type="desktop-application">
    <id>${APP_ID}</id>
    <metadata_license>GPL-3.0-or-later</metadata_license>
    <project_license>GPL-3.0-or-later</project_license>
    <name>${APP_NAME}</name>
    <summary>Generate membrane kymographs from live-cell microscopy images</summary>
    <description>
        <p>
            Membrane Kymograph Generator is a specialized tool for creating kymographs
            from live-cell time-lapse fluorescence microscopy images. It provides an interactive
            GUI for automated boundary detection, multi-channel support, exports in multiple formats, and provides options for downstream analysis.
        </p>
        <p>Features:</p>
        <ul>
            <li>Interactive GUI for kymograph generation</li>
            <li>Multi-channel fluorescence support</li>
            <li>Multiple output formats (PNG, SVG, PDF)</li>
            <li>Kymograph adjustment and preview</li>
            <li>Automated boundary detection and smoothing</li>
            <li>Built-in correlation analysis tool</li>
            <li>Open-source and cross-platform (Windows, macOS, Linux)</li>
        </ul>
    </description>
    <categories>
        <category>Science</category>
        <category>Education</category>
        <category>Biology</category>
    </categories>
    <releases>
        <release version="${VERSION}" date="$(date +%Y-%m-%d)">
            <description>
                <p>Release ${VERSION}</p>
            </description>
        </release>
    </releases>
</component>
EOF

# Create AppRun script
echo "Creating AppRun..."
cat > "${APPDIR}/AppRun" << 'EOF'
#!/bin/bash
# AppRun script for AppImage

# Get the directory where AppImage is mounted
HERE="$(dirname "$(readlink -f "${0}")")"

# Export library paths
export LD_LIBRARY_PATH="${HERE}/usr/lib:${LD_LIBRARY_PATH}"
export PATH="${HERE}/usr/bin:${PATH}"

# Export XDG paths for proper integration
export XDG_DATA_DIRS="${HERE}/usr/share:${XDG_DATA_DIRS:-/usr/local/share:/usr/share}"

# Set Qt/GTK environment variables
export QT_QPA_PLATFORMTHEME=gtk3
export GDK_BACKEND=x11

# Change to user's home directory
cd "${HOME}"

# Run the application
exec "${HERE}/usr/bin/membrane-kymograph-wrapper" "$@"
EOF

chmod +x "${APPDIR}/AppRun"

# Download appimagetool if not present
APPIMAGETOOL="./appimagetool-${APPIMAGETOOL_ARCH}.AppImage"
if [ ! -f "${APPIMAGETOOL}" ]; then
    echo "Downloading appimagetool for ${APPIMAGETOOL_ARCH}..."
    wget -q "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-${APPIMAGETOOL_ARCH}.AppImage" -O "${APPIMAGETOOL}"
    chmod +x "${APPIMAGETOOL}"
fi

# Build AppImage
echo "Building AppImage..."
rm -f "${APPIMAGE_DIR}/${APPIMAGE_NAME}"

ARCH=${ARCH} "${APPIMAGETOOL}" "${APPDIR}" "${APPIMAGE_DIR}/${APPIMAGE_NAME}"

# Make AppImage executable
chmod +x "${APPIMAGE_DIR}/${APPIMAGE_NAME}"

# Clean up AppDir (optional - uncomment if desired)
# rm -rf "${APPDIR}"

echo ""
echo "✓ AppImage creation complete!"
echo "  Output: ${APPIMAGE_DIR}/${APPIMAGE_NAME}"
echo "  Size: $(du -h "${APPIMAGE_DIR}/${APPIMAGE_NAME}" | cut -f1)"
echo ""
echo "To test the AppImage:"
echo "  chmod +x ${APPIMAGE_DIR}/${APPIMAGE_NAME}"
echo "  ./${APPIMAGE_DIR}/${APPIMAGE_NAME}"
echo ""
echo "To install system-wide:"
echo "  sudo mv ${APPIMAGE_DIR}/${APPIMAGE_NAME} /usr/local/bin/membrane-kymograph"
echo ""
echo "Note: AppImage files are self-contained and can be run from anywhere."
