#!/bin/bash

# DataCode Installation Script
# This script installs DataCode as a global command

set -e

echo "🧠 DataCode Installation Script"
echo "==============================="
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Error: Rust/Cargo is not installed"
    echo "💡 Please install Rust first: https://rustup.rs/"
    exit 1
fi

echo "✅ Rust/Cargo found"

# Check if we're in the DataCode directory
if [ ! -f "Cargo.toml" ] || ! grep -q "name = \"data-code\"" Cargo.toml; then
    echo "❌ Error: Please run this script from the DataCode project directory"
    exit 1
fi

echo "✅ DataCode project directory confirmed"

# Detect platform and determine GPU feature
GPU_FEATURE=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - use Metal
    GPU_FEATURE="metal"
    echo "🍎 macOS detected - will compile with Metal GPU support"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux - use CUDA (if available)
    GPU_FEATURE="cuda"
    echo "🐧 Linux detected - will compile with CUDA GPU support"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows - use CUDA
    GPU_FEATURE="cuda"
    echo "🪟 Windows detected - will compile with CUDA GPU support"
else
    echo "⚠️  Unknown platform - will compile without GPU support"
    GPU_FEATURE=""
fi

# Build the project in release mode with GPU support
echo ""
if [ -n "$GPU_FEATURE" ]; then
    echo "🔨 Building DataCode in release mode with $GPU_FEATURE GPU support..."
    cargo build --release --features "$GPU_FEATURE"
else
    echo "🔨 Building DataCode in release mode..."
    cargo build --release
fi

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to build DataCode"
    exit 1
fi

echo "✅ Build completed successfully"

# Install using cargo install with GPU support
echo ""
echo "📦 Installing DataCode globally..."
if [ -n "$GPU_FEATURE" ]; then
    cargo install --path . --features "$GPU_FEATURE" --force
else
    cargo install --path . --force
fi

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install DataCode"
    exit 1
fi

echo "✅ DataCode installed successfully!"

# Function to install icon to executable on macOS
install_icon_to_executable() {
    local executable_path="$1"
    local icon_source="$2"
    
    if [ ! -f "$executable_path" ]; then
        echo "⚠️  Executable not found: $executable_path"
        return 1
    fi
    
    if [ ! -f "$icon_source" ]; then
        echo "⚠️  Icon source not found: $icon_source"
        return 1
    fi
    
    # Create temporary iconset directory
    local temp_iconset=$(mktemp -d -t datacode-icon.XXXXXX)
    local temp_icns="$temp_iconset/datacode-plot.icns"
    
    # Create iconset
    mkdir -p "$temp_iconset/datacode-plot.iconset"
    local iconset_dir="$temp_iconset/datacode-plot.iconset"
    
    # Create various icon sizes
    sips -z 16 16     "$icon_source" --out "$iconset_dir/icon_16x16.png" > /dev/null 2>&1
    sips -z 32 32     "$icon_source" --out "$iconset_dir/icon_16x16@2x.png" > /dev/null 2>&1
    sips -z 32 32     "$icon_source" --out "$iconset_dir/icon_32x32.png" > /dev/null 2>&1
    sips -z 64 64     "$icon_source" --out "$iconset_dir/icon_32x32@2x.png" > /dev/null 2>&1
    sips -z 128 128   "$icon_source" --out "$iconset_dir/icon_128x128.png" > /dev/null 2>&1
    sips -z 256 256   "$icon_source" --out "$iconset_dir/icon_128x128@2x.png" > /dev/null 2>&1
    sips -z 256 256   "$icon_source" --out "$iconset_dir/icon_256x256.png" > /dev/null 2>&1
    sips -z 512 512   "$icon_source" --out "$iconset_dir/icon_256x256@2x.png" > /dev/null 2>&1
    sips -z 512 512   "$icon_source" --out "$iconset_dir/icon_512x512.png" > /dev/null 2>&1
    sips -z 1024 1024 "$icon_source" --out "$iconset_dir/icon_512x512@2x.png" > /dev/null 2>&1
    
    # Convert iconset to ICNS
    iconutil -c icns "$iconset_dir" -o "$temp_icns" > /dev/null 2>&1
    
    if [ ! -f "$temp_icns" ]; then
        echo "⚠️  Failed to create ICNS file"
        rm -rf "$temp_iconset"
        return 1
    fi
    
    # Method 1: Use Swift (built-in on macOS, no dependencies)
    # Create a temporary Swift script to set the icon
    local swift_script=$(mktemp -t datacode-icon.XXXXXX.swift)
    cat > "$swift_script" << 'SWIFT_EOF'
import AppKit
import Foundation

let args = CommandLine.arguments
guard args.count == 3 else {
    exit(1)
}

let filePath = args[1]
let iconPath = args[2]

guard let icon = NSImage(contentsOfFile: iconPath) else {
    exit(1)
}

let workspace = NSWorkspace.shared
let success = workspace.setIcon(icon, forFile: filePath, options: [])
exit(success ? 0 : 1)
SWIFT_EOF
    
    swift "$swift_script" "$executable_path" "$temp_icns" > /dev/null 2>&1
    local swift_result=$?
    rm -f "$swift_script"
    
    if [ $swift_result -eq 0 ]; then
        echo "✅ Icon installed to executable using Swift"
        rm -rf "$temp_iconset"
        return 0
    fi
    
    # Method 2: Use Python with PyObjC (if available)
    python3 << EOF > /dev/null 2>&1
import sys
import os

try:
    from AppKit import NSWorkspace, NSImage
    
    def set_icon(file_path, icon_path):
        try:
            icon = NSImage.alloc().initWithContentsOfFile_(icon_path)
            if icon is None:
                return False
            workspace = NSWorkspace.sharedWorkspace()
            success = workspace.setIcon_forFile_options_(icon, file_path, 0)
            return success
        except Exception:
            return False
    
    success = set_icon('$executable_path', '$temp_icns')
    sys.exit(0 if success else 1)
except ImportError:
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo "✅ Icon installed to executable using Python/PyObjC"
        rm -rf "$temp_iconset"
        return 0
    fi
    
    # Method 3: Use AppleScript (improved version)
    osascript << EOF > /dev/null 2>&1
tell application "Finder"
    try
        set targetFile to POSIX file "$executable_path" as alias
        set iconFile to POSIX file "$temp_icns" as alias
        set fileIcon to icon of iconFile
        set icon of targetFile to fileIcon
        return true
    on error
        return false
    end try
end tell
EOF
    
    if [ $? -eq 0 ]; then
        echo "✅ Icon installed to executable using AppleScript"
        rm -rf "$temp_iconset"
        return 0
    fi
    
    # Method 4: Try fileicon as last resort (if user has it installed)
    if command -v fileicon &> /dev/null; then
        fileicon set "$executable_path" "$temp_icns" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "✅ Icon installed to executable using fileicon"
            rm -rf "$temp_iconset"
            return 0
        fi
    fi
    
    # Cleanup
    rm -rf "$temp_iconset"
    echo "⚠️  Failed to install icon to executable (tried Swift, Python/PyObjC, AppleScript, and fileicon)"
    echo "💡 Tip: Install PyObjC for better compatibility: pip3 install pyobjc-framework-Cocoa"
    return 1
}

# On macOS, install icon to executable and create app bundle if requested
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Install icon to executable
    echo ""
    echo "🎨 Installing icon to executable..."
    EXECUTABLE_PATH="$HOME/.cargo/bin/datacode"
    ICON_SOURCE="$(cd "$(dirname "$0")" && pwd)/src/lib/plot/icon/datacode-plot.png"
    
    if [ -f "$ICON_SOURCE" ]; then
        install_icon_to_executable "$EXECUTABLE_PATH" "$ICON_SOURCE"
    else
        echo "⚠️  Icon source not found: $ICON_SOURCE"
    fi
    
    # Check if we should create app bundle (from make install or explicit request)
    if [ "$CREATE_APP_BUNDLE" = "1" ] || [ "$1" = "--with-app-bundle" ]; then
        echo ""
        echo "🍎 Creating macOS app bundle..."
        APP_BUNDLE_SCRIPT="$(cd "$(dirname "$0")" && pwd)/packaging/macos/build-app-bundle.sh"
        if [ -f "$APP_BUNDLE_SCRIPT" ]; then
            chmod +x "$APP_BUNDLE_SCRIPT"
            "$APP_BUNDLE_SCRIPT"
            if [ $? -eq 0 ]; then
                echo "✅ App bundle created successfully!"
                echo "📱 You can find it at: packaging/macos/DataCode.app"
                echo "💡 To install to Applications: cp -r packaging/macos/DataCode.app /Applications/"
            else
                echo "⚠️  App bundle creation failed, but installation completed"
            fi
        else
            echo "⚠️  App bundle script not found: $APP_BUNDLE_SCRIPT"
        fi
    fi
fi

# Check if cargo bin directory is in PATH
CARGO_BIN_DIR="$HOME/.cargo/bin"
if [[ ":$PATH:" != *":$CARGO_BIN_DIR:"* ]]; then
    echo ""
    echo "⚠️  Warning: Cargo bin directory is not in your PATH"
    echo "📝 Add this line to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
    echo "   export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    echo ""
    echo "🔄 Or run this command now:"
    echo "   export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    echo ""
    echo "🔄 Then reload your shell or run:"
    echo "   source ~/.bashrc  # or ~/.zshrc"
else
    echo "✅ Cargo bin directory is already in PATH"
fi

# Test the installation
echo ""
echo "🧪 Testing installation..."
if command -v datacode &> /dev/null; then
    echo "✅ DataCode command is available!"
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "📚 Usage:"
    echo "  datacode                 # Start interactive REPL (default)"
    echo "  datacode filename.dc     # Execute DataCode file"
    echo "  datacode filename.dc --build_model  # Export tables to SQLite"
    echo "  datacode --websocket     # Start WebSocket server"
    echo "  datacode --websocket --host 0.0.0.0 --port 8899  # Custom host/port"
    echo "  datacode --websocket --use-ve  # Virtual environment mode"
    echo "  datacode --help          # Show help"
    echo ""
    echo "🚀 Try running: datacode --help"
else
    echo "⚠️  DataCode command not found in PATH"
    echo "💡 You may need to restart your terminal or update your PATH"
    echo ""
    echo "🔄 Try running:"
    echo "   export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    echo "   datacode --help"
fi

echo ""
echo "✨ Happy coding with DataCode! ✨"
