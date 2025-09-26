#!/usr/bin/env python3
"""
Simple script to run the Streamlit web application.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit application."""
    
    # Get the webapp directory
    webapp_dir = Path(__file__).parent
    app_file = webapp_dir / "app.py"
    
    if not app_file.exists():
        print(f"Error: {app_file} not found!")
        sys.exit(1)
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_file),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("Starting SVD Image Compression Web Application...")
    print(f"Running: {' '.join(cmd)}")
    print("Open your browser to: http://localhost:8501")
    
    try:
        subprocess.run(cmd, cwd=webapp_dir)
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as e:
        print(f"Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()