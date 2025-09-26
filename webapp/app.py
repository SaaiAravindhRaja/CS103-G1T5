"""
SVD Image Compression Web Application

A streamlined, professional interface for exploring image compression
using Singular Value Decomposition (SVD).
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import core modules
try:
    from utils.styling import load_core_styles
    from pages import single_compression
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()


def main():
    """Main application entry point with streamlined interface."""
    
    # Configure page settings
    st.set_page_config(
        page_title="SVD Image Compression",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load core styling
    load_core_styles()
    
    # Main header
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        ">
            <h1 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 600;">
                🖼️ SVD Image Compression
            </h1>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
                Interactive tool for image compression analysis using Singular Value Decomposition
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display main compression interface
    single_compression.show()
    
    # Simple footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem; color: #6b7280; font-size: 0.9rem;">
            <p><strong>SVD Image Compression Tool</strong> • Built for educational purposes</p>
        </div>
        """,
        unsafe_allow_html=True
    )





if __name__ == "__main__":
    main()