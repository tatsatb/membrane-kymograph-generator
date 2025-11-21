#!/bin/bash
# Uninstallation script for Membrane Kymograph Generator
# This script removes the system-wide installation

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

echo -e "${YELLOW}Membrane Kymograph Generator - Uninstallation Script${NC}"
echo "===================================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
    echo "Usage: sudo ./uninstall.sh"
    exit 1
fi

# Check if installation exists
if [ ! -d "$INSTALL_DIR" ] && [ ! -L "$BIN_LINK" ] && [ ! -f "$DESKTOP_FILE" ]; then
    echo -e "${YELLOW}No installation found. Nothing to uninstall.${NC}"
    exit 0
fi

echo "This will remove Membrane Kymograph Generator from your system."
echo ""
echo "The following will be removed:"
if [ -d "$INSTALL_DIR" ]; then
    echo "  - Application directory: $INSTALL_DIR"
fi
if [ -L "$BIN_LINK" ] || [ -f "$BIN_LINK" ]; then
    echo "  - Executable symlink: $BIN_LINK"
fi
if [ -f "$DESKTOP_FILE" ]; then
    echo "  - Desktop entry: $DESKTOP_FILE"
fi
echo "  - Application icons from $ICON_DIR"
echo ""

# Ask for confirmation
read -p "Do you want to continue? (y/N): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstallation cancelled."
    exit 0
fi

echo ""
echo "Uninstalling..."

# Remove installation directory
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing application directory..."
    rm -rf "$INSTALL_DIR"
    echo -e "${GREEN}✓${NC} Removed $INSTALL_DIR"
fi

# Remove symlink
if [ -L "$BIN_LINK" ] || [ -f "$BIN_LINK" ]; then
    echo "Removing executable symlink..."
    rm -f "$BIN_LINK"
    echo -e "${GREEN}✓${NC} Removed $BIN_LINK"
fi

# Remove desktop file
if [ -f "$DESKTOP_FILE" ]; then
    echo "Removing desktop entry..."
    rm -f "$DESKTOP_FILE"
    echo -e "${GREEN}✓${NC} Removed $DESKTOP_FILE"
fi

# Remove icons
echo "Removing application icons..."
for size in 16 22 24 32 48 64 128 256; do
    ICON_FILE="$ICON_DIR/${size}x${size}/apps/membrane-kymograph.png"
    if [ -f "$ICON_FILE" ]; then
        rm -f "$ICON_FILE"
    fi
done
echo -e "${GREEN}✓${NC} Removed application icons"

# Update icon cache
if command -v gtk-update-icon-cache &> /dev/null; then
    echo "Updating icon cache..."
    gtk-update-icon-cache -f -t "$ICON_DIR" 2>/dev/null || true
fi

# Update desktop database
if command -v update-desktop-database &> /dev/null; then
    echo "Updating desktop database..."
    update-desktop-database /usr/share/applications 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}Uninstallation complete!${NC}"
echo ""
echo "Membrane Kymograph Generator has been removed from your system."
echo ""
