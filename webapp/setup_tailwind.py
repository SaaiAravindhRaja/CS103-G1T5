#!/usr/bin/env python3
"""
Setup script for initializing Tailwind CSS in the SVD Image Compression webapp.
"""

import subprocess
import sys
from pathlib import Path
import json

def create_directories():
    """Create necessary directories."""
    webapp_dir = Path(__file__).parent
    styles_dir = webapp_dir / "styles"
    styles_dir.mkdir(exist_ok=True)
    print("✅ Created styles directory")

def check_requirements():
    """Check if required tools are available."""
    print("🔍 Checking requirements...")
    
    # Check Python
    print(f"✅ Python: {sys.version}")
    
    # Check Node.js (optional)
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js: {result.stdout.strip()}")
            return True
        else:
            print("⚠️  Node.js not found (optional for CDN mode)")
            return False
    except FileNotFoundError:
        print("⚠️  Node.js not found (optional for CDN mode)")
        return False

def setup_cdn_mode():
    """Set up Tailwind CSS in CDN mode (no build required)."""
    print("🌐 Setting up Tailwind CSS in CDN mode...")
    
    webapp_dir = Path(__file__).parent
    
    # Create a simple indicator file
    cdn_indicator = webapp_dir / "styles" / ".cdn-mode"
    cdn_indicator.write_text("Using Tailwind CSS CDN mode\n")
    
    print("✅ CDN mode configured")
    print("📋 The webapp will use Tailwind CSS from CDN")
    print("   No build step required - styles load automatically")

def setup_build_mode():
    """Set up Tailwind CSS in build mode."""
    print("🔧 Setting up Tailwind CSS in build mode...")
    
    webapp_dir = Path(__file__).parent
    
    # Check if package.json exists
    package_json = webapp_dir / "package.json"
    if not package_json.exists():
        print("❌ package.json not found")
        return False
    
    # Install dependencies
    try:
        print("📦 Installing npm dependencies...")
        subprocess.run(['npm', 'install'], cwd=webapp_dir, check=True)
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Build CSS
    try:
        print("🎨 Building initial CSS...")
        subprocess.run(['python', 'build_css.py', 'build'], cwd=webapp_dir, check=True)
        print("✅ CSS built successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build CSS: {e}")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🎨 SVD Image Compression - Tailwind CSS Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Check requirements
    has_node = check_requirements()
    
    print("\n📋 Setup Options:")
    print("1. CDN Mode (Recommended) - No build required, loads from CDN")
    print("2. Build Mode - Compile CSS locally (requires Node.js)")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        if has_node:
            mode = input("\nChoose mode (cdn/build) [cdn]: ").lower() or "cdn"
        else:
            print("\n🌐 Using CDN mode (Node.js not available)")
            mode = "cdn"
    
    if mode == "cdn":
        setup_cdn_mode()
        print("\n🚀 Setup complete! You can now run the webapp:")
        print("   cd webapp && streamlit run app.py")
        
    elif mode == "build":
        if not has_node:
            print("❌ Build mode requires Node.js")
            print("   Install Node.js from https://nodejs.org/")
            sys.exit(1)
        
        if setup_build_mode():
            print("\n🚀 Setup complete! You can now run the webapp:")
            print("   cd webapp && streamlit run app.py")
            print("\n🔧 For development with auto-rebuild:")
            print("   python build_css.py watch")
        else:
            print("❌ Setup failed")
            sys.exit(1)
    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()