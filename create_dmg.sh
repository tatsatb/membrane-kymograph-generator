#!/bin/bash

# create_dmg.sh - Create macOS DMG installer from PyInstaller output
# Requires: create-dmg (install via: brew install create-dmg)

set -e

# Configuration
APP_NAME="Membrane Kymograph"
#Placeholder version; will be automatically updated during build process
VERSION="0.0.1"
PYINSTALLER_DIR="dist/membrane-kymograph"
APP_BUNDLE_DIR="dist/${APP_NAME}.app"
DMG_DIR="installers"
VOLUME_NAME="${APP_NAME} ${VERSION}"

# Determine architecture
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    ARCH_SUFFIX="arm64"
else
    ARCH_SUFFIX="x64"
fi

DMG_NAME="membrane-kymograph-${VERSION}-macos-${ARCH_SUFFIX}.dmg"

echo "Creating macOS DMG installer..."
echo "Architecture: ${ARCH} (${ARCH_SUFFIX})"
echo "Source: ${PYINSTALLER_DIR}"
echo "Output: ${DMG_DIR}/${DMG_NAME}"

# Create installers directory if it doesn't exist
mkdir -p "${DMG_DIR}"

# Check if PyInstaller output exists
if [ ! -d "${PYINSTALLER_DIR}" ]; then
    echo "Error: PyInstaller output directory not found at ${PYINSTALLER_DIR}"
    echo "Please run PyInstaller first:"
    echo "  pyinstaller membrane-kymograph.spec --clean --noconfirm"
    exit 1
fi

# Create .app bundle structure
echo "Creating .app bundle..."
rm -rf "${APP_BUNDLE_DIR}"
mkdir -p "${APP_BUNDLE_DIR}/Contents/MacOS"
mkdir -p "${APP_BUNDLE_DIR}/Contents/Resources"

# Copy PyInstaller output to .app bundle
echo "Copying executable and dependencies..."
cp -R "${PYINSTALLER_DIR}"/* "${APP_BUNDLE_DIR}/Contents/MacOS/"

# Ensure all copied files are fully written
sync

# Copy icon to Resources directory
ICON_FILE="icons/memkymo.icns"
if [ -f "${ICON_FILE}" ]; then
    echo "Copying application icon..."
    cp "${ICON_FILE}" "${APP_BUNDLE_DIR}/Contents/Resources/icon.icns"
else
    echo "Warning: Icon file ${ICON_FILE} not found. App will use default icon."
fi

# Create Info.plist
echo "Creating Info.plist..."
cat > "${APP_BUNDLE_DIR}/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>${APP_NAME}</string>
    <key>CFBundleExecutable</key>
    <string>membrane-kymograph</string>
    <key>CFBundleIdentifier</key>
    <string>in.tatsatbanerjee.membrane-kymograph</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>${APP_NAME}</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>${VERSION}</string>
    <key>CFBundleVersion</key>
    <string>${VERSION}</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSRequiresAquaSystemAppearance</key>
    <false/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.education</string>
</dict>
</plist>
EOF

# Make executable
chmod +x "${APP_BUNDLE_DIR}/Contents/MacOS/membrane-kymograph"

# Ensure all writes are flushed to disk before creating DMG
sync

# Give filesystem a moment to settle (helps prevent "Resource busy" errors)
sleep 4

# Check if create-dmg is installed
if ! command -v create-dmg &> /dev/null; then
    echo ""
    echo "Warning: create-dmg is not installed."
    echo "Install it with: brew install create-dmg"
    echo ""
    echo "Creating basic DMG without create-dmg..."
    
    # Create a simple DMG using hdiutil
    TEMP_DMG="${DMG_DIR}/temp-${DMG_NAME}"
    rm -f "${TEMP_DMG}" "${DMG_DIR}/${DMG_NAME}"
    
    # Create temporary directory for DMG contents
    TEMP_DIR=$(mktemp -d)
    cp -R "${APP_BUNDLE_DIR}" "${TEMP_DIR}/"
    ln -s /Applications "${TEMP_DIR}/Applications"
    
    # Create README
    cat > "${TEMP_DIR}/README.txt" << EOF
${APP_NAME} ${VERSION}

Installation:
1. Drag "${APP_NAME}.app" to the Applications folder
2. Open from Applications or Launchpad

First Launch:
If macOS prevents opening the app (because it's not from the App Store):
1. Right-click the app and select "Open"
2. Click "Open" in the dialog that appears
3. The app will now open and can be launched normally in the future

Alternatively, you can allow the app in System Preferences:
System Preferences → Security & Privacy → General → "Open Anyway"

For more information, visit:
${MyAppURL:-https://github.com/tatsatb/membrane-kymograph-generator}
EOF
    
    # Create DMG
    hdiutil create -volname "${VOLUME_NAME}" -srcfolder "${TEMP_DIR}" -ov -format UDZO "${DMG_DIR}/${DMG_NAME}"
    
    # Clean up
    rm -rf "${TEMP_DIR}"
    
    echo "Basic DMG created: ${DMG_DIR}/${DMG_NAME}"
else
    # Use create-dmg for a more professional DMG
    echo "Creating DMG with create-dmg..."
    
    # Clean up any existing DMG files thoroughly
    rm -f "${DMG_DIR}/${DMG_NAME}"
    # Remove any temp DMGs from previous runs (create-dmg uses rw.*.dmg pattern)
    find "${DMG_DIR}" -name "rw.*.dmg" -delete 2>/dev/null || true
    
    # Unmount any volumes that might interfere
    hdiutil detach "/Volumes/${VOLUME_NAME}" 2>/dev/null || true
    # Also try to unmount any dmg.* volumes (create-dmg uses these)
    for vol in /Volumes/dmg.*; do
        if [ -d "$vol" ]; then
            hdiutil detach "$vol" 2>/dev/null || true
        fi
    done
    
    # Try create-dmg with retry logic
    # FIXME: Workaround for create-dmg reliability issues on GitHub Actions macOS runners
    # See: https://github.com/actions/runner-images/issues/7522
    # (create-dmg sometimes has issues with locked files in CI environments)
    
    # Create a clean temporary directory with just the .app bundle
    TEMP_SOURCE=$(mktemp -d)
    cp -R "${APP_BUNDLE_DIR}" "${TEMP_SOURCE}/"
    
    MAX_TRIES=10
    ATTEMPT=0
    CREATE_DMG_SUCCESS=false
    
    set +e  # Temporarily disable exit on error
    
    echo "Attempting to create DMG (will retry up to ${MAX_TRIES} times if needed)..."
    
    while [ $ATTEMPT -lt $MAX_TRIES ]; do
        ATTEMPT=$((ATTEMPT + 1))
        
        if [ $ATTEMPT -gt 1 ]; then
            echo "Retry attempt ${ATTEMPT}/${MAX_TRIES}..."
            # Clean up any partial DMG from previous attempt
            rm -f "${DMG_DIR}/${DMG_NAME}"
            find "${DMG_DIR}" -name "rw.*.dmg" -delete 2>/dev/null || true
            # Brief pause between retries
            sleep 2
        fi
        
        create-dmg \
            --volname "${VOLUME_NAME}" \
            --window-pos 200 120 \
            --window-size 600 400 \
            --icon-size 100 \
            --icon "${APP_NAME}.app" 150 190 \
            --hide-extension "${APP_NAME}.app" \
            --app-drop-link 450 190 \
            --no-internet-enable \
            "${DMG_DIR}/${DMG_NAME}" \
            "${TEMP_SOURCE}"
        
        CREATE_DMG_EXIT=$?
        
        # Check if DMG was actually created successfully
        if [ $CREATE_DMG_EXIT -eq 0 ] && [ -f "${DMG_DIR}/${DMG_NAME}" ]; then
            CREATE_DMG_SUCCESS=true
            echo "✓ Professional DMG created successfully on attempt ${ATTEMPT}"
            break
        fi
    done
    
    set -e  # Re-enable exit on error
    
    rm -rf "${TEMP_SOURCE}"
    
    # If create-dmg failed after all retries, use hdiutil fallback
    if [ "$CREATE_DMG_SUCCESS" = false ]; then
        echo ""
        echo "⚠ Warning: create-dmg failed after ${MAX_TRIES} attempts"
        echo "Using hdiutil fallback for basic DMG..."
        
        # Fallback to basic hdiutil method
        # Clean up any partial DMG from create-dmg
        rm -f "${DMG_DIR}/${DMG_NAME}"
        find "${DMG_DIR}" -name "rw.*.dmg" -delete 2>/dev/null || true
        
        TEMP_DIR=$(mktemp -d)
        cp -R "${APP_BUNDLE_DIR}" "${TEMP_DIR}/"
        ln -s /Applications "${TEMP_DIR}/Applications"
        
        # Create README for fallback DMG
        cat > "${TEMP_DIR}/README.txt" << 'FALLBACK_EOF'
Installation:
1. Drag "Membrane Kymograph.app" to the Applications folder
2. Open from Applications or Launchpad

First Launch:
If macOS prevents opening the app:
1. Right-click the app and select "Open"
2. Click "Open" in the dialog
3. The app will now launch normally
FALLBACK_EOF
        
        # Create DMG with hdiutil
        hdiutil create -volname "${VOLUME_NAME}" -srcfolder "${TEMP_DIR}" -ov -format UDZO "${DMG_DIR}/${DMG_NAME}"
        
        # Clean up
        rm -rf "${TEMP_DIR}"
        
        echo "Basic DMG created: ${DMG_DIR}/${DMG_NAME}"
    fi
fi

# Clean up app bundle (optional - comment out if you want to keep it)
# rm -rf "${APP_BUNDLE_DIR}"

echo ""
echo "✓ DMG creation complete!"
echo "  Output: ${DMG_DIR}/${DMG_NAME}"
echo "  Size: $(du -h "${DMG_DIR}/${DMG_NAME}" | cut -f1)"
echo ""
echo "To test the DMG:"
echo "  open ${DMG_DIR}/${DMG_NAME}"
