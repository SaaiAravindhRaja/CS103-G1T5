"""
Test script for the enhanced upload component.
"""

import streamlit as st
import sys
from pathlib import Path

# Add utils to path
utils_path = Path(__file__).parent / "utils"
sys.path.insert(0, str(utils_path))

from upload_component import create_enhanced_upload_component

def main():
    st.title("Enhanced Upload Component Test")
    
    # Test the upload component
    upload_result = create_enhanced_upload_component(
        key="test_upload",
        multiple=True,
        show_preview=True,
        show_progress=True
    )
    
    if upload_result:
        st.success("Upload component working correctly!")
        st.json({
            "filename": upload_result['file'].name,
            "array_shape": upload_result['array'].shape,
            "metadata": upload_result['metadata']
        })
    else:
        st.info("Upload an image to test the component")

if __name__ == "__main__":
    main()