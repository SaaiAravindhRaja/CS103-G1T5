"""
Simple, streamlined image upload component.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io


def create_simple_upload(key="upload", help_text=""):
    """Create a simple, clean upload component."""
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help=help_text or "Drag and drop an image file or click to browse. Supported formats: PNG, JPG, JPEG, BMP, TIFF",
        key=key
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array and normalize
            image_array = np.array(image) / 255.0
            
            return {
                'file': uploaded_file,
                'pil': image,
                'array': image_array,
                'success': True
            }
            
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    
    return None


def show_image_info(image_data):
    """Display basic image information."""
    
    if image_data is None:
        return
    
    array = image_data['array']
    file = image_data['file']
    
    height, width = array.shape[:2]
    file_size = len(file.getvalue()) / 1024
    
    info_html = f"""
    <div style="
        background: white; 
        padding: 1.5rem; 
        border-radius: 12px; 
        border: 1px solid var(--gray-200);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    ">
        <h4 style="margin: 0 0 1rem 0; color: var(--gray-700);">📋 Image Information</h4>
        <div style="display: grid; gap: 0.5rem;">
            <div><strong>Filename:</strong> {file.name}</div>
            <div><strong>Dimensions:</strong> {width} × {height} pixels</div>
            <div><strong>File Size:</strong> {file_size:.1f} KB</div>
            <div><strong>Format:</strong> RGB (3 channels)</div>
            <div><strong>Data Type:</strong> {array.dtype}</div>
        </div>
    </div>
    """
    
    st.markdown(info_html, unsafe_allow_html=True)