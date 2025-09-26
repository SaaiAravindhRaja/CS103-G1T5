"""
Streamlined styling for the SVD Image Compression webapp.
"""

import streamlit as st


def load_core_styles():
    """Load essential styles for the streamlined interface."""
    
    css = """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Core Variables */
    :root {
        --primary: #3b82f6;
        --primary-dark: #1d4ed8;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-500: #6b7280;
        --gray-700: #374151;
        --gray-900: #111827;
    }
    
    /* Base Styles */
    .main .block-container {
        max-width: 1200px;
        padding: 2rem 1rem;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: var(--gray-900);
    }
    
    /* Streamlit Component Overrides */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    .stFileUploader > div {
        border: 2px dashed var(--gray-200);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        background: var(--gray-50);
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary);
        background: #eff6ff;
    }
    
    .stSlider > div > div > div > div {
        background: var(--primary);
    }
    
    /* Custom Components */
    .metric-card {
        background: white;
        border: 1px solid var(--gray-200);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    .upload-zone {
        border: 2px dashed var(--gray-300);
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        background: var(--gray-50);
        margin: 1rem 0;
        transition: all 0.2s ease;
    }
    
    .upload-zone:hover {
        border-color: var(--primary);
        background: #eff6ff;
    }
    
    .status-success {
        color: var(--success);
        font-weight: 500;
    }
    
    .status-warning {
        color: var(--warning);
        font-weight: 500;
    }
    
    .status-error {
        color: var(--error);
        font-weight: 500;
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 3px solid var(--gray-200);
        border-top: 3px solid var(--primary);
        border-radius: 50%;
        width: 24px;
        height: 24px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 8px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem 0.5rem;
        }
        
        .metric-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .upload-zone {
            padding: 2rem 1rem;
        }
        
        /* Stack columns on mobile */
        .stColumns > div {
            margin-bottom: 1rem;
        }
        
        /* Adjust button sizes */
        .stButton > button {
            padding: 0.5rem 1rem;
            font-size: 0.9rem;
        }
        
        /* Improve file uploader on mobile */
        .stFileUploader > div {
            padding: 1.5rem 1rem;
        }
        
        /* Better slider on mobile */
        .stSlider {
            margin: 1rem 0;
        }
    }
    
    @media (max-width: 480px) {
        .main .block-container {
            padding: 0.5rem 0.25rem;
        }
        
        .metric-card {
            padding: 0.75rem;
            font-size: 0.9rem;
        }
        
        /* Single column layout on very small screens */
        .stColumns {
            flex-direction: column;
        }
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


def create_metric_card(title, value, description="", status=""):
    """Create a simple metric card."""
    status_class = f"status-{status}" if status else ""
    
    return f"""
    <div class="metric-card">
        <h4 style="margin: 0 0 0.5rem 0; color: var(--gray-700);">{title}</h4>
        <div style="font-size: 1.5rem; font-weight: 600; margin: 0.5rem 0; {f'class="{status_class}"' if status_class else ''}">{value}</div>
        {f'<p style="margin: 0; color: var(--gray-500); font-size: 0.9rem;">{description}</p>' if description else ''}
    </div>
    """


def show_loading(message="Processing..."):
    """Show a simple loading indicator."""
    return f"""
    <div style="text-align: center; padding: 2rem;">
        <div class="loading-spinner"></div>
        <span style="color: var(--gray-600);">{message}</span>
    </div>
    """