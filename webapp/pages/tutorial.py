"""
Interactive tutorial and help page for the SVD Image Compression tool.
"""

import streamlit as st
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import io

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.styling import (
    create_tutorial_step, create_status_indicator, 
    create_zoomable_image, enable_presentation_mode
)
from compression.svd_compressor import SVDCompressor
from evaluation.metrics_calculator import MetricsCalculator
import plotly.graph_objects as go


def show():
    """Display the interactive tutorial and help page."""
    
    # Use the new layout system
    from utils.styling import create_main_content_area, close_main_content_area
    
    # Create main content area
    create_main_content_area()
    
    st.markdown("# 📚 Interactive Tutorial & Help")
    st.markdown("Learn how to use the SVD Image Compression tool with interactive examples and comprehensive guides.")
    
    # Tutorial navigation
    tutorial_sections = [
        "🚀 Getting Started",
        "🔍 Understanding SVD Compression", 
        "📷 Single Image Analysis",
        "📊 Batch Processing",
        "⚖️ Comparison Analysis",
        "🎯 Tips & Best Practices",
        "🎨 Presentation Mode",
        "❓ FAQ & Troubleshooting"
    ]
    
    selected_section = st.selectbox(
        "Choose a tutorial section:",
        tutorial_sections,
        help="Select the topic you'd like to learn about"
    )
    
    # Display selected section
    if selected_section == "🚀 Getting Started":
        show_getting_started()
    elif selected_section == "🔍 Understanding SVD Compression":
        show_svd_theory()
    elif selected_section == "📷 Single Image Analysis":
        show_single_image_tutorial()
    elif selected_section == "📊 Batch Processing":
        show_batch_processing_tutorial()
    elif selected_section == "⚖️ Comparison Analysis":
        show_comparison_tutorial()
    elif selected_section == "🎯 Tips & Best Practices":
        show_tips_and_practices()
    elif selected_section == "🎨 Presentation Mode":
        show_presentation_mode_tutorial()
    else:  # FAQ & Troubleshooting
        show_faq()
    
    # Close main content area
    close_main_content_area()


def show_getting_started():
    """Show getting started tutorial."""
    
    st.markdown("## 🚀 Getting Started with SVD Image Compression")
    
    create_tutorial_step(
        1,
        "Welcome to SVD Image Compression",
        """
        This tool helps you explore image compression using Singular Value Decomposition (SVD).
        SVD is a mathematical technique that can compress images while preserving important visual information.
        """,
        highlight_text="SVD compression is lossless in theory but lossy in practice when we keep only the top k singular values."
    )    

    create_tutorial_step(
        2,
        "Navigation Overview",
        """
        The tool has four main sections:
        • <strong>Single Image Compression</strong>: Analyze individual images with real-time controls
        • <strong>Batch Processing</strong>: Process multiple images simultaneously
        • <strong>Comparison Analysis</strong>: Compare different compression levels side-by-side
        • <strong>Tutorial & Help</strong>: This section with guides and examples
        """,
        highlight_text="Start with Single Image Compression to get familiar with the concepts."
    )
    
    create_tutorial_step(
        3,
        "Key Concepts",
        """
        <strong>K-Value:</strong> Number of singular values to keep (lower k = higher compression)<br>
        <strong>PSNR:</strong> Peak Signal-to-Noise Ratio (higher is better, >30dB is good)<br>
        <strong>SSIM:</strong> Structural Similarity Index (0-1 scale, >0.8 is good)<br>
        <strong>Compression Ratio:</strong> How much smaller the compressed image is (higher is more compressed)
        """,
        highlight_text="The k-value is the most important parameter - it controls the quality vs compression trade-off."
    )
    
    # Interactive demo
    st.markdown("### 🎮 Try a Quick Demo")
    
    if st.button("🎯 Generate Demo Image", help="Create a simple test image to try compression"):
        demo_image = create_demo_image()
        st.session_state.demo_image = demo_image
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Demo Image (Original)**")
            st.image(demo_image, caption="Simple test pattern", use_column_width=True)
        
        with col2:
            k_demo = st.slider("Try different k-values:", 1, 50, 10, key="demo_k")
            
            if st.button("Compress Demo", key="compress_demo"):
                compressed_demo = compress_demo_image(demo_image, k_demo)
                st.markdown(f"**Compressed (k={k_demo})**")
                st.image(compressed_demo, caption=f"Compressed with k={k_demo}", use_column_width=True)


def show_svd_theory():
    """Show SVD theory and mathematical background."""
    
    st.markdown("## 🔍 Understanding SVD Compression")
    
    create_tutorial_step(
        1,
        "What is Singular Value Decomposition?",
        """
        SVD decomposes any matrix A into three matrices: A = U × Σ × V^T<br>
        • <strong>U</strong>: Left singular vectors (image patterns)<br>
        • <strong>Σ</strong>: Singular values (importance weights)<br>
        • <strong>V^T</strong>: Right singular vectors (spatial patterns)
        """,
        highlight_text="The singular values in Σ are ordered from largest to smallest - the largest ones contain the most important information."
    )
    
    create_tutorial_step(
        2,
        "How Compression Works",
        """
        Instead of storing all singular values, we keep only the top k largest ones.
        This reduces storage from m×n (original) to k×(m+n+1) (compressed).
        The compression ratio depends on how small k is compared to min(m,n).
        """,
        highlight_text="Typical compression ratios range from 2:1 (high quality) to 20:1 (high compression)."
    )
    
    # Interactive SVD visualization
    st.markdown("### 📊 Interactive SVD Visualization")
    
    if st.button("🎯 Show SVD Breakdown"):
        show_svd_breakdown()


def show_single_image_tutorial():
    """Show single image analysis tutorial."""
    
    st.markdown("## 📷 Single Image Analysis Tutorial")
    
    create_tutorial_step(
        1,
        "Upload Your Image",
        """
        Start by uploading an image (PNG, JPG, or JPEG format).
        The tool supports drag-and-drop and can handle multiple images.
        Images are automatically resized to 256×256 pixels for consistent processing.
        """,
        highlight_text="For best results, use images with clear subjects and good contrast."
    )
    
    create_tutorial_step(
        2,
        "Adjust Compression Settings",
        """
        Use the k-value slider to control compression level:
        • <strong>Low k (1-10):</strong> High compression, lower quality
        • <strong>Medium k (10-30):</strong> Balanced compression and quality  
        • <strong>High k (30-100):</strong> Lower compression, higher quality
        """,
        highlight_text="Enable 'Real-time Preview' to see changes instantly as you move the slider."
    )
    
    create_tutorial_step(
        3,
        "Interpret Quality Metrics",
        """
        The dashboard shows four key metrics:
        • <strong>PSNR:</strong> Overall image quality (aim for >25dB)
        • <strong>SSIM:</strong> Structural similarity (aim for >0.7)
        • <strong>MSE:</strong> Mean squared error (lower is better)
        • <strong>Compression Ratio:</strong> Space savings achieved
        """,
        highlight_text="Use the Quality Score (0-100) as a quick overall assessment."
    )


def show_batch_processing_tutorial():
    """Show batch processing tutorial."""
    
    st.markdown("## 📊 Batch Processing Tutorial")
    
    create_tutorial_step(
        1,
        "Upload Multiple Images",
        """
        Batch processing allows you to analyze multiple images simultaneously.
        Upload 2-50 images using the enhanced drag-and-drop interface.
        All images will be processed with the same settings for consistent comparison.
        """,
        highlight_text="Batch processing is perfect for comparing how different images respond to compression."
    )
    
    create_tutorial_step(
        2,
        "Configure Processing Settings",
        """
        Set up your batch experiment:
        • <strong>K-values:</strong> Choose a range or specific values to test
        • <strong>Processing Mode:</strong> RGB, Grayscale, or both
        • <strong>Output Options:</strong> Save compressed images and generate reports
        """,
        highlight_text="Start with a small range of k-values (e.g., 5, 15, 30) to understand the patterns."
    )
    
    create_tutorial_step(
        3,
        "Analyze Results",
        """
        The results include:
        • <strong>Interactive plots:</strong> Quality trends across images and k-values
        • <strong>Comparison grids:</strong> Visual comparison of compression levels
        • <strong>Statistical analysis:</strong> Performance consistency and optimization recommendations
        """,
        highlight_text="Use the heatmap view to quickly identify which images compress well at different k-values."
    )


def show_comparison_tutorial():
    """Show comparison analysis tutorial."""
    
    st.markdown("## ⚖️ Comparison Analysis Tutorial")
    
    create_tutorial_step(
        1,
        "K-Value Comparison",
        """
        Comparison analysis helps you find the optimal k-value for your images.
        Upload one or more images and select multiple k-values to compare side-by-side.
        This is ideal for fine-tuning compression settings.
        """,
        highlight_text="Use comparison analysis to find the 'sweet spot' between quality and compression for your specific images."
    )
    
    create_tutorial_step(
        2,
        "Analysis Types",
        """
        Choose from three analysis modes:
        • <strong>Single Image Focus:</strong> Detailed analysis of one image across k-values
        • <strong>Multi-Image Comparison:</strong> Compare how different images perform
        • <strong>Cross-Image Analysis:</strong> Find k-values that work well across all images
        """,
        highlight_text="Cross-Image Analysis is best when you need one k-value setting for multiple image types."
    )
    
    create_tutorial_step(
        3,
        "Optimization Recommendations",
        """
        The tool provides three types of recommendations:
        • <strong>Quality-Focused:</strong> Best k-values for maximum image quality
        • <strong>Compression-Focused:</strong> Best k-values for maximum space savings
        • <strong>Balanced:</strong> Optimal trade-off between quality and compression
        """,
        highlight_text="The balanced recommendations are usually the best starting point for most applications."
    )


def show_tips_and_practices():
    """Show tips and best practices."""
    
    st.markdown("## 🎯 Tips & Best Practices")
    
    st.markdown("### 📈 Optimization Strategies")
    
    create_tutorial_step(
        1,
        "Choosing the Right K-Value",
        """
        • <strong>For photos:</strong> Start with k=20-50 depending on detail level
        • <strong>For graphics/diagrams:</strong> Often k=10-30 is sufficient
        • <strong>For textures:</strong> May need k=30-100 to preserve detail
        • <strong>For line art:</strong> Very low k=5-15 often works well
        """,
        highlight_text="Always validate with visual inspection - metrics don't tell the whole story."
    )
    
    create_tutorial_step(
        2,
        "Quality Assessment Guidelines",
        """
        • <strong>PSNR >35dB:</strong> Excellent quality, often visually lossless
        • <strong>PSNR 25-35dB:</strong> Good quality, suitable for most applications
        • <strong>PSNR 20-25dB:</strong> Fair quality, noticeable but acceptable artifacts
        • <strong>PSNR <20dB:</strong> Poor quality, significant artifacts
        """,
        highlight_text="SSIM >0.8 combined with PSNR >25dB usually indicates good perceptual quality."
    )
    
    create_tutorial_step(
        3,
        "Performance Optimization",
        """
        • Use grayscale mode for faster processing when color isn't critical
        • Limit batch processing to 10-20 images for reasonable processing time
        • Enable real-time preview only for single images to avoid lag
        • Use range analysis with 5-8 k-values for comprehensive evaluation
        """,
        highlight_text="Processing time increases quadratically with image size and linearly with k-value."
    )


def show_presentation_mode_tutorial():
    """Show presentation mode tutorial."""
    
    st.markdown("## 🎨 Presentation Mode")
    
    create_tutorial_step(
        1,
        "Activating Presentation Mode",
        """
        Presentation mode optimizes the interface for academic presentations and demonstrations.
        It increases font sizes, improves contrast, and enhances visual elements for better visibility.
        """,
        highlight_text="Perfect for classroom demonstrations, conference presentations, and academic reviews."
    )
    
    # Presentation mode toggle
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎨 Enable Presentation Mode", use_container_width=True):
            enable_presentation_mode()
            st.success("✅ Presentation mode enabled! Fonts and UI elements are now optimized for presentations.")
    
    with col2:
        if st.button("📱 Standard Mode", use_container_width=True):
            st.rerun()
    
    create_tutorial_step(
        2,
        "Presentation Features",
        """
        Presentation mode includes:
        • <strong>Larger fonts:</strong> All text is scaled up for better visibility
        • <strong>Enhanced metrics:</strong> Key numbers are prominently displayed
        • <strong>High contrast:</strong> Improved readability on projectors
        • <strong>Simplified layout:</strong> Focus on essential information
        """,
        highlight_text="Use presentation mode when demonstrating to groups or recording tutorials."
    )


def show_faq():
    """Show FAQ and troubleshooting."""
    
    st.markdown("## ❓ Frequently Asked Questions")
    
    faqs = [
        {
            "question": "Why do my images look blurry after compression?",
            "answer": "This is normal with SVD compression. Lower k-values remove high-frequency details, causing blur. Try increasing the k-value or check if your image has fine details that need higher k-values to preserve."
        },
        {
            "question": "What's the difference between PSNR and SSIM?",
            "answer": "PSNR measures pixel-level differences (technical quality), while SSIM measures structural similarity (perceptual quality). SSIM often correlates better with human perception of image quality."
        },
        {
            "question": "Why is processing slow for large batches?",
            "answer": "SVD computation is intensive, especially for larger k-values. Try reducing the number of images, using smaller k-value ranges, or switching to grayscale mode for faster processing."
        },
        {
            "question": "Can I use this for real image compression?",
            "answer": "SVD compression is primarily educational. For practical compression, use formats like JPEG, WebP, or AVIF which are much more efficient. SVD helps understand compression principles."
        },
        {
            "question": "What image formats are supported?",
            "answer": "The tool supports PNG, JPG, and JPEG formats. Images are automatically converted to RGB and resized to 256×256 pixels for consistent processing."
        }
    ]
    
    for i, faq in enumerate(faqs, 1):
        with st.expander(f"❓ {faq['question']}"):
            st.markdown(faq['answer'])
    
    st.markdown("### 🔧 Troubleshooting")
    
    st.markdown("""
    **Common Issues and Solutions:**
    
    • **Upload fails:** Check file size (<10MB) and format (PNG/JPG/JPEG)
    • **Processing hangs:** Reduce k-value range or number of images
    • **Poor quality results:** Try higher k-values or check image suitability
    • **Memory errors:** Use grayscale mode or smaller image batches
    • **Slow performance:** Close other browser tabs and reduce processing load
    """)


def create_demo_image():
    """Create a simple demo image for tutorial."""
    
    # Create a simple geometric pattern
    size = 64
    image = np.zeros((size, size, 3))
    
    # Add some patterns
    for i in range(size):
        for j in range(size):
            # Checkerboard pattern
            if (i // 8 + j // 8) % 2 == 0:
                image[i, j] = [0.8, 0.2, 0.2]  # Red
            else:
                image[i, j] = [0.2, 0.2, 0.8]  # Blue
    
    # Add a circle in the center
    center = size // 2
    radius = size // 4
    for i in range(size):
        for j in range(size):
            if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                image[i, j] = [0.2, 0.8, 0.2]  # Green
    
    return image


def compress_demo_image(image, k):
    """Compress the demo image with given k-value."""
    
    try:
        compressor = SVDCompressor()
        compressed_image, _ = compressor.compress_image(image, k)
        return compressed_image
    except:
        return image  # Return original if compression fails


def show_svd_breakdown():
    """Show interactive SVD breakdown visualization."""
    
    # Create a simple test matrix
    test_image = create_demo_image()
    
    # Convert to grayscale for SVD
    gray_image = np.dot(test_image[...,:3], [0.2989, 0.5870, 0.1140])
    
    # Compute SVD
    U, s, Vt = np.linalg.svd(gray_image, full_matrices=False)
    
    st.markdown("### 📊 SVD Components Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Singular Values (Σ)**")
        
        # Plot singular values
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(s) + 1)),
            y=s,
            mode='lines+markers',
            name='Singular Values'
        ))
        fig.update_layout(
            xaxis_title="Index",
            yaxis_title="Value",
            yaxis_type="log",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Energy Distribution**")
        
        # Calculate cumulative energy
        energy = s ** 2
        cumulative_energy = np.cumsum(energy) / np.sum(energy)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_energy) + 1)),
            y=cumulative_energy * 100,
            mode='lines+markers',
            name='Cumulative Energy %'
        ))
        fig.update_layout(
            xaxis_title="Number of Components",
            yaxis_title="Energy Captured (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Show reconstruction with different k values
    st.markdown("**Reconstruction with Different K-Values**")
    
    k_values = [1, 5, 10, 20]
    cols = st.columns(len(k_values))
    
    for i, k in enumerate(k_values):
        with cols[i]:
            # Reconstruct with k components
            reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
            reconstructed = np.clip(reconstructed, 0, 1)
            
            # Convert back to RGB for display
            reconstructed_rgb = np.stack([reconstructed] * 3, axis=-1)
            
            st.image(reconstructed_rgb, caption=f"k={k}", use_column_width=True)
            
            # Show energy captured
            energy_captured = np.sum(s[:k] ** 2) / np.sum(s ** 2) * 100
            st.markdown(f"Energy: {energy_captured:.1f}%")