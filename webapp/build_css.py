#!/usr/bin/env python3
"""
Build script for compiling Tailwind CSS for the SVD Image Compression webapp.
"""

import subprocess
import sys
from pathlib import Path
import os

def check_node_installed():
    """Check if Node.js is installed."""
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Node.js is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Node.js is not installed or not in PATH")
            return False
    except FileNotFoundError:
        print("❌ Node.js is not installed")
        return False

def install_dependencies():
    """Install npm dependencies."""
    print("📦 Installing npm dependencies...")
    try:
        result = subprocess.run(['npm', 'install'], cwd=Path(__file__).parent, check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ npm is not installed")
        return False

def build_css(production=False):
    """Build Tailwind CSS."""
    print("🎨 Building Tailwind CSS...")
    
    webapp_dir = Path(__file__).parent
    input_css = webapp_dir / "styles" / "input.css"
    output_css = webapp_dir / "styles" / "output.css"
    
    # Ensure styles directory exists
    (webapp_dir / "styles").mkdir(exist_ok=True)
    
    # Build command
    cmd = [
        'npx', 'tailwindcss',
        '-i', str(input_css),
        '-o', str(output_css)
    ]
    
    if production:
        cmd.append('--minify')
        print("🚀 Building for production (minified)")
    else:
        print("🔧 Building for development")
    
    try:
        result = subprocess.run(cmd, cwd=webapp_dir, check=True, capture_output=True, text=True)
        print("✅ Tailwind CSS built successfully")
        print(f"📄 Output: {output_css}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to build CSS: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def watch_css():
    """Watch for changes and rebuild CSS automatically."""
    print("👀 Watching for changes... (Press Ctrl+C to stop)")
    
    webapp_dir = Path(__file__).parent
    input_css = webapp_dir / "styles" / "input.css"
    output_css = webapp_dir / "styles" / "output.css"
    
    cmd = [
        'npx', 'tailwindcss',
        '-i', str(input_css),
        '-o', str(output_css),
        '--watch'
    ]
    
    try:
        subprocess.run(cmd, cwd=webapp_dir, check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Stopped watching")
    except subprocess.CalledProcessError as e:
        print(f"❌ Watch failed: {e}")

def main():
    """Main build function."""
    print("🎨 SVD Image Compression - Tailwind CSS Build Tool")
    print("=" * 50)
    
    # Check if Node.js is installed
    if not check_node_installed():
        print("\n📋 To install Node.js:")
        print("  - Visit: https://nodejs.org/")
        print("  - Or use a package manager like brew, apt, etc.")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
    else:
        command = "build"
    
    webapp_dir = Path(__file__).parent
    os.chdir(webapp_dir)
    
    if command == "install":
        install_dependencies()
    elif command == "build":
        if not install_dependencies():
            sys.exit(1)
        build_css(production=False)
    elif command == "build-prod":
        if not install_dependencies():
            sys.exit(1)
        build_css(production=True)
    elif command == "watch":
        if not install_dependencies():
            sys.exit(1)
        watch_css()
    else:
        print(f"❌ Unknown command: {command}")
        print("\n📋 Available commands:")
        print("  install    - Install npm dependencies")
        print("  build      - Build CSS for development")
        print("  build-prod - Build CSS for production (minified)")
        print("  watch      - Watch for changes and rebuild automatically")
        sys.exit(1)

if __name__ == "__main__":
    main()