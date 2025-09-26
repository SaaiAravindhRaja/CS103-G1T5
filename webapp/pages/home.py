"""
Enhanced home page with project overview, presentation features, and interactive elements.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.styling import (
    create_metric_card, create_enhanced_loading_animation, 
    create_status_indicator, enable_presentation_mode,
    create_zoomable_image
)
from utils.animations import (
    add_entrance_animation, add_hover_effect, create_notification,
    create_animated_counter, animation_manager
)
from utils.accessibility import (
    create_accessible_button, create_accessible_image, 
    announce_to_screen_reader, accessibility_manager
)


def show():
    """Display the enhanced home page with presentation features."""
    
    # Use the new layout system
    from utils.styling import create_main_content_area, close_main_content_area
    
    # Create main content area
    create_main_content_area()
    
    # Presentation mode toggle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎨 Enable Presentation Mode", use_container_width=True, help="Optimize interface for presentations and demonstrations"):
            enable_presentation_mode()
            st.success("✅ Presentation mode enabled! Interface optimized for presentations.")
    
    # Enhanced hero section with animations
    st.markdown(
        """
        <div class="hero-section animate-fade-in hover-glow" 
             style="text-align: center; padding: 3rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0; color: white; transition: all 0.3s ease;">
            <h1 class="animate-slide-down" style="color: white; font-size: 3.5rem; margin-bottom: 1rem; animation-delay: 0.2s;">
                🖼️ SVD Image Compression
            </h1>
            <p class="animate-slide-up" style="font-size: 1.4rem; margin-bottom: 1rem; opacity: 0.9; animation-delay: 0.4s;">
                Interactive Academic Tool for Image Compression Analysis
            </p>
            <p class="animate-slide-up" style="font-size: 1.1rem; opacity: 0.8; animation-delay: 0.6s;">
                Explore Singular Value Decomposition with real-time visualizations and comprehensive metrics
            </p>
        </div>
        
        <style>
        .hero-section:hover {
            transform: translateY(-2px);
            box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        }
        
        @keyframes slideDown {
            from { transform: translateY(-30px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .animate-slide-down {
            animation: slideDown 0.8s ease-out forwards;
            opacity: 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Quick navigation cards with animations
    st.markdown('<h3 class="animate-fade-in" style="animation-delay: 0.8s;">🚀 Quick Start</h3>', unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3, nav_col4 = st.columns(4)
    
    # Add staggered animations to navigation cards
    nav_cards = [
        {"col": nav_col1, "text": "📷 Single Image\nAnalysis", "help": "Analyze individual images", "page": "pages/single_compression.py", "delay": "1.0s"},
        {"col": nav_col2, "text": "📊 Batch\nProcessing", "help": "Process multiple images", "page": "pages/batch_processing.py", "delay": "1.2s"},
        {"col": nav_col3, "text": "⚖️ Comparison\nAnalysis", "help": "Compare compression levels", "page": "pages/comparison.py", "delay": "1.4s"},
        {"col": nav_col4, "text": "📚 Tutorial\n& Help", "help": "Learn how to use the tool", "page": "pages/tutorial.py", "delay": "1.6s"}
    ]
    
    # Add CSS for navigation card animations
    st.markdown("""
    <style>
    .nav-card-container {
        animation: scaleIn 0.6s ease-out forwards;
        opacity: 0;
        transform: scale(0.9);
    }
    
    @keyframes scaleIn {
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    .stButton > button {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)
    
    for card in nav_cards:
        with card["col"]:
            st.markdown(f'<div class="nav-card-container" style="animation-delay: {card["delay"]};">', unsafe_allow_html=True)
            if st.button(card["text"], use_container_width=True, help=card["help"], key=f"nav_{card['page']}"):
                create_notification("Navigating to " + card["text"].replace("\n", " "), "info", 2.0)
                st.switch_page(card["page"])
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Overview section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 📚 What is SVD Image Compression?")
        st.markdown(
            """
            Singular Value Decomposition (SVD) is a powerful mathematical technique that can be used 
            for image compression. It works by decomposing an image matrix into three matrices and 
            keeping only the most significant components.
            
            **How it works:**
            1. **Decomposition**: Break down the image matrix A into U, Σ, and V^T
            2. **Truncation**: Keep only the top k singular values and corresponding vectors
            3. **Reconstruction**: Rebuild the image using the reduced components
            
            **Key Benefits:**
            - Adjustable compression levels
            - Preserves important image features
            - Educational value for understanding linear algebra
            - Quality vs. compression trade-off analysis
            """
        )
        
        st.markdown("## 🎯 Features")
        
        features = [
            ("🔍", "Single Image Analysis", "Upload and compress individual images with real-time quality metrics"),
            ("📊", "Batch Processing", "Process multiple images simultaneously with comprehensive reporting"),
            ("⚖️", "Comparison Tools", "Side-by-side analysis of different compression levels"),
            ("📈", "Quality Metrics", "PSNR, SSIM, MSE, and compression ratio calculations"),
            ("💾", "Export Options", "Download compressed images and analysis reports"),
            ("🎨", "Interactive Visualization", "Real-time plots and charts for compression analysis")
        ]
        
        for icon, title, description in features:
            st.markdown(
                f"""
                <div class="metric-card" style="margin: 1rem 0;">
                    <h4>{icon} {title}</h4>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        # Interactive SVD visualization
        st.markdown("### 🔬 Interactive SVD Demo")
        
        # Create demo button
        if st.button("🎯 Generate Interactive Demo", use_container_width=True):
            create_status_indicator("processing", "Generating SVD demonstration...")
            
            # Create a simple demonstration pattern
            x = np.linspace(0, 4*np.pi, 50)
            y = np.linspace(0, 4*np.pi, 50)
            X, Y = np.meshgrid(x, y)
            original = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X) * np.sin(2*Y)
            
            # SVD compression demonstration
            U, s, Vt = np.linalg.svd(original, full_matrices=False)
            
            # Interactive k-value selector
            k_demo = st.slider("Adjust compression level (k):", 1, 30, 10, key="home_demo_k")
            
            # Create interactive plot
            fig = go.Figure()
            
            # Add singular values
            fig.add_trace(go.Scatter(
                x=list(range(1, len(s[:30]) + 1)),
                y=s[:30],
                mode='lines+markers',
                name='Singular Values',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=6)
            ))
            
            # Highlight selected k
            fig.add_trace(go.Scatter(
                x=list(range(1, k_demo + 1)),
                y=s[:k_demo],
                mode='markers',
                name=f'Kept (k={k_demo})',
                marker=dict(color='#ef4444', size=8, symbol='circle')
            ))
            
            # Add vertical line at k
            fig.add_vline(
                x=k_demo,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text=f"k = {k_demo}"
            )
            
            fig.update_layout(
                title="Interactive Singular Values",
                xaxis_title="Component Index",
                yaxis_title="Singular Value",
                yaxis_type="log",
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show reconstruction
            reconstructed = U[:, :k_demo] @ np.diag(s[:k_demo]) @ Vt[:k_demo, :]
            
            col_orig, col_comp = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.image(original, use_column_width=True, clamp=True)
            
            with col_comp:
                st.markdown(f"**Compressed (k={k_demo})**")
                st.image(reconstructed, use_column_width=True, clamp=True)
            
            # Show compression stats
            energy_kept = np.sum(s[:k_demo]**2) / np.sum(s**2) * 100
            compression_ratio = (50 * 50) / (k_demo * (50 + 50 + 1))
            
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Energy Retained", f"{energy_kept:.1f}%")
            with col_stat2:
                st.metric("Compression Ratio", f"{compression_ratio:.1f}:1")
            
            create_status_indicator("complete", "Demo generated successfully!")
        
        # Quick stats
        st.markdown("### 📊 Tool Capabilities")
        
        stats_data = [
            ("🎛️", "K-Values", "1-256", "Adjustable compression levels"),
            ("📏", "Metrics", "4 Types", "PSNR, SSIM, MSE, Ratio"),
            ("📁", "Formats", "PNG/JPG", "Common image formats"),
            ("⚡", "Processing", "Real-time", "Instant compression preview")
        ]
        
        for icon, title, value, desc in stats_data:
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                    border: 1px solid #e2e8f0;
                    border-radius: 10px;
                    padding: 1rem;
                    margin: 0.5rem 0;
                    text-align: center;
                ">
                    <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-weight: 600; color: #1f2937;">{title}</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: #3b82f6; margin: 0.25rem 0;">{value}</div>
                    <div style="font-size: 0.9rem; color: #6b7280;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Getting started section
    st.markdown("---")
    st.markdown("## 🚀 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="info-card">
                <h4>1️⃣ Upload Image</h4>
                <p>Go to <strong>Single Image Compression</strong> and upload your image file (PNG, JPG, JPEG supported).</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="info-card">
                <h4>2️⃣ Adjust Settings</h4>
                <p>Use the k-value slider to control compression level. Lower k = higher compression, higher k = better quality.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="info-card">
                <h4>3️⃣ Analyze Results</h4>
                <p>View quality metrics, compare images side-by-side, and download your compressed results.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Mathematical background
    st.markdown("---")
    st.markdown("## 🧮 Mathematical Background")
    
    st.markdown(
        """
        The SVD of an image matrix **A** (m×n) can be written as:
        
        **A = UΣV^T**
        
        Where:
        - **U** (m×m): Left singular vectors (image spatial patterns)
        - **Σ** (m×n): Diagonal matrix of singular values (importance weights)
        - **V^T** (n×n): Right singular vectors (image frequency patterns)
        
        For compression with parameter **k**, we keep only the first **k** components:
        
        **A_k = U_k Σ_k V_k^T**
        
        **Storage Requirements:**
        - Original: m × n values
        - Compressed: k(m + n + 1) values
        - Compression ratio: mn / k(m + n + 1)
        """
    )
    
    # Tips and best practices
    st.markdown("---")
    st.markdown("## 💡 Tips & Best Practices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="success-card">
                <h4>✅ Best Practices</h4>
                <ul>
                    <li>Start with k=10-20 for initial experiments</li>
                    <li>Use square images for best results</li>
                    <li>Try different image types (portraits, landscapes, textures)</li>
                    <li>Compare PSNR and SSIM metrics together</li>
                    <li>Use batch processing for systematic analysis</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="warning-card">
                <h4>⚠️ Considerations</h4>
                <ul>
                    <li>Very low k values may produce artifacts</li>
                    <li>High k values offer minimal compression</li>
                    <li>Processing time increases with image size</li>
                    <li>Some image types compress better than others</li>
                    <li>Quality metrics are guidelines, not absolutes</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Enhanced call to action with status indicators
    st.markdown("---")
    st.markdown("## 🎯 Ready to Get Started?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                margin: 1rem 0;
            ">
                <h3 style="color: white; margin-bottom: 1rem;">🚀 Start Analyzing</h3>
                <p style="margin-bottom: 1.5rem;">Jump right into image compression with our intuitive tools</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("📷 Try Single Image Analysis", use_container_width=True, type="primary"):
            create_status_indicator("processing", "Navigating to Single Image Analysis...")
    
    with col2:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                color: white;
                padding: 2rem;
                border-radius: 15px;
                text-align: center;
                margin: 1rem 0;
            ">
                <h3 style="color: white; margin-bottom: 1rem;">📚 Learn First</h3>
                <p style="margin-bottom: 1.5rem;">New to SVD compression? Start with our interactive tutorial</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if st.button("📖 Open Tutorial", use_container_width=True):
            create_status_indicator("processing", "Loading interactive tutorial...")
    
    # Footer with additional resources
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem; background: #f8fafc; border-radius: 10px; margin-top: 2rem;">
            <p style="color: #6b7280; margin-bottom: 0.5rem;">
                <strong>Academic Tool for Educational Purposes</strong>
            </p>
            <p style="color: #9ca3af; font-size: 0.9rem;">
                Built for learning and understanding image compression principles • 
                Perfect for coursework, research, and demonstrations
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Close main content area
    close_main_content_area()