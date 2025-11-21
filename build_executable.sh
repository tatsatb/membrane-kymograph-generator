#!/bin/bash
# Build script for creating cross-platform executable with PyInstaller

echo "=== Membrane Kymograph - PyInstaller Build Script ==="
echo ""

# Check if virtual environment is active
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Virtual environment is not active."
    echo "It's recommended to build in a clean virtual environment."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please activate virtual environment and try again."
        exit 1
    fi
fi

# Install PyInstaller if not already installed
echo "Step 1: Checking PyInstaller installation"
echo "------------------------------------------"
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller not found. Installing..."
    pip install pyinstaller
else
    echo "✓ PyInstaller is already installed"
fi
echo ""

# Clean previous builds
echo "Step 2: Cleaning previous builds"
echo "---------------------------------"
if [ -d "build" ]; then
    echo "Removing build/ directory..."
    rm -rf build/
fi
if [ -d "dist" ]; then
    echo "Removing dist/ directory..."
    rm -rf dist/
fi
echo "✓ Cleanup complete"
echo ""

# Run PyInstaller
echo "Step 3: Building executable with PyInstaller"
echo "---------------------------------------------"
echo "This WILL take several minutes..."
echo ""

pyinstaller membrane-kymograph.spec --clean --noconfirm

# Check if build succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "Output location: dist/membrane-kymograph/"
    echo ""
    echo "To run the application:"
    echo "  cd dist/membrane-kymograph"
    echo "  ./membrane-kymograph"
    echo ""
    
    # Get size of distribution
    if command -v du &> /dev/null; then
        SIZE=$(du -sh dist/membrane-kymograph | cut -f1)
        echo "Distribution size: $SIZE"
    fi
    
    echo ""
    echo "You can now distribute the entire 'dist/membrane-kymograph' folder."
    echo "Users would be able to run the 'membrane-kymograph' executable inside it."
else
    echo ""
    echo "❌ Build failed!"
    echo "Check the error messages for details."
    exit 1
fi

echo ""
echo "=== Build script finished ==="
