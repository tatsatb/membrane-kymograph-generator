#!/bin/bash
# Installation script for Membrane Kymograph Generator
# This script installs the application system-wide

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Installation directories
INSTALL_DIR="/opt/membrane-kymograph"
BIN_LINK="/usr/local/bin/membrane-kymograph"
DESKTOP_FILE="/usr/share/applications/membrane-kymograph.desktop"
ICON_DIR="/usr/share/icons/hicolor"

echo -e "${GREEN}Membrane Kymograph Generator - Installation Script${NC}"
echo "=================================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    echo "Usage: sudo ./install.sh"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if the application directory exists
if [ ! -d "$SCRIPT_DIR/membrane-kymograph" ]; then
    echo -e "${RED}Error: Application directory 'membrane-kymograph' not found${NC}"
    echo "Make sure you extracted the tarball correctly."
    exit 1
fi

# Check if executable exists
if [ ! -f "$SCRIPT_DIR/membrane-kymograph/membrane-kymograph" ]; then
    echo -e "${RED}Error: Executable 'membrane-kymograph' not found${NC}"
    exit 1
fi

echo "Installing Membrane Kymograph Generator to $INSTALL_DIR..."
echo ""

# Remove old installation if it exists
if [ -d "$INSTALL_DIR" ]; then
    echo -e "${YELLOW}Removing existing installation...${NC}"
    rm -rf "$INSTALL_DIR"
fi

# Create installation directory and copy files
echo "Copying application files..."
mkdir -p "$INSTALL_DIR"
cp -r "$SCRIPT_DIR/membrane-kymograph"/* "$INSTALL_DIR/"

# Ensure executable has correct permissions
chmod +x "$INSTALL_DIR/membrane-kymograph"

# Create symlink in /usr/local/bin
echo "Creating symlink in /usr/local/bin..."
rm -f "$BIN_LINK"
ln -s "$INSTALL_DIR/membrane-kymograph" "$BIN_LINK"

# Create .desktop file
echo "Creating desktop entry..."
mkdir -p /usr/share/applications

cat > "$DESKTOP_FILE" << 'EOF'
[Desktop Entry]
Version=0.0.1
Type=Application
Name=Membrane Kymograph Generator
Comment=A GUI-based cross-platform tool for generating membrane kymographs from live-cell time-lapse microscopy images
Exec=/usr/local/bin/membrane-kymograph
Icon=membrane-kymograph
Terminal=false
Categories=Science;Education;Biology;ImageProcessing;
Keywords=microscopy;kymograph;cell-biology;fluorescence;image-processing;
StartupWMClass=Tk
EOF

# Copy icon if it exists
if [ -f "$SCRIPT_DIR/icons/memkymo.png" ]; then
    echo "Installing application icon..."
    # Install icon in multiple sizes
    for size in 16 22 24 32 48 64 128 256; do
        ICON_SIZE_DIR="$ICON_DIR/${size}x${size}/apps"
        mkdir -p "$ICON_SIZE_DIR"
        # Use ImageMagick convert if available, otherwise just copy the PNG
        if command -v convert &> /dev/null; then
            convert "$SCRIPT_DIR/icons/memkymo.png" -resize ${size}x${size} "$ICON_SIZE_DIR/membrane-kymograph.png" 2>/dev/null || \
            cp "$SCRIPT_DIR/icons/memkymo.png" "$ICON_SIZE_DIR/membrane-kymograph.png"
        else
            cp "$SCRIPT_DIR/icons/memkymo.png" "$ICON_SIZE_DIR/membrane-kymograph.png"
        fi
    done
    
    # Update icon cache
    if command -v gtk-update-icon-cache &> /dev/null; then
        gtk-update-icon-cache -f -t "$ICON_DIR" 2>/dev/null || true
    fi
elif [ -f "$INSTALL_DIR/_internal/icons/memkymo.png" ]; then
    echo "Installing application icon from bundle..."
    for size in 16 22 24 32 48 64 128 256; do
        ICON_SIZE_DIR="$ICON_DIR/${size}x${size}/apps"
        mkdir -p "$ICON_SIZE_DIR"
        if command -v convert &> /dev/null; then
            convert "$INSTALL_DIR/_internal/icons/memkymo.png" -resize ${size}x${size} "$ICON_SIZE_DIR/membrane-kymograph.png" 2>/dev/null || \
            cp "$INSTALL_DIR/_internal/icons/memkymo.png" "$ICON_SIZE_DIR/membrane-kymograph.png"
        else
            cp "$INSTALL_DIR/_internal/icons/memkymo.png" "$ICON_SIZE_DIR/membrane-kymograph.png"
        fi
    done
    
    if command -v gtk-update-icon-cache &> /dev/null; then
        gtk-update-icon-cache -f -t "$ICON_DIR" 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}Warning: Icon file not found, skipping icon installation${NC}"
fi

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    echo "Updating desktop database..."
    update-desktop-database /usr/share/applications 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}Installation complete!${NC}"
echo ""
echo "You can now run the application by:"
echo "  1. Typing 'membrane-kymograph' in terminal"
echo "  2. Searching for 'Membrane Kymograph Generator' in your application menu"
echo ""
echo "Installation location: $INSTALL_DIR"
echo "Executable symlink: $BIN_LINK"
echo "Desktop entry: $DESKTOP_FILE"
echo ""
echo "To uninstall, run: sudo $SCRIPT_DIR/uninstall.sh"
echo ""
