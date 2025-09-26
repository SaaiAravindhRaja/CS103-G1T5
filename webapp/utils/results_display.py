"""
Results display component with before/after comparison, metrics dashboard, 
zoom/pan functionality, and download options.
Enhanced with comprehensive tooltip system and contextual help.
"""

import streamlit as st
import numpy as np
import io
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import base64
import time
from datetime import datetime
from .tooltip_system import tooltip_system, create_tooltip


def create_results_display_component(original_image, compressed_image, compression_data, filename="image"):
    """
    Create a comprehensive results display with all required features.
    
    Args:
        original_image: Original image array
        compressed_image: Compressed image array  
        compression_data: Dictionary with compression metrics and settings
        filename: Original filename for downloads
    """
    
    st.markdown("## 🎯 Compression Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🖼️ Image Comparison", 
        "🔍 Detailed Analysis",
        "💾 Download Results"
    ])
    
    with tab1:
        create_metrics_dashboard(compression_data)
    
    with tab2:
        create_image_comparison_component(original_image, compressed_image, compression_data)
    
    with tab3:
        create_detailed_analysis_component(original_image, compressed_image, compression_data)
    
    with tab4:
        create_download_component(original_image, compressed_image, compression_data, filename)


def create_metrics_dashboard(compression_data):
    """Create an interactive metrics dashboard with visual indicators and comprehensive tooltips."""
    
    # Dashboard header with help
    dashboard_tooltip = """
    **Quality Metrics Dashboard**
    
    Interactive gauges showing compression quality and efficiency.
    
    **Metrics Explained:**
    • PSNR: Peak Signal-to-Noise Ratio (higher = better quality)
    • SSIM: Structural Similarity Index (closer to 1 = better)
    • Compression Ratio: Space savings (higher = more compression)
    • Overall Quality: Composite score (0-100 scale)
    
    **Color Coding:**
    • Green: Excellent values
    • Yellow: Good values  
    • Red: Poor values
    
    **Tips:**
    • Hover over gauges for detailed information
    • Use PSNR and SSIM together for complete assessment
    • Balance compression ratio with quality metrics
    """
    
    st.markdown(
        create_tooltip(
            '<h3 style="margin: 0;">📈 Quality Metrics Dashboard</h3>',
            dashboard_tooltip,
            position="right"
        ),
        unsafe_allow_html=True
    )
    
    # Calculate quality score if not present
    if 'quality_score' not in compression_data:
        compression_data['quality_score'] = calculate_quality_score(
            compression_data.get('psnr', 0), 
            compression_data.get('ssim', 0)
        )
    
    # Create gauge charts using Plotly
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PSNR Quality', 'SSIM Similarity', 'Compression Ratio', 'Overall Quality'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # PSNR Gauge (0-50 dB range)
    psnr_value = compression_data.get('psnr', 0)
    psnr_color = get_metric_color(psnr_value, [20, 30, 35])
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=psnr_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "PSNR (dB)"},
        delta={'reference': 30, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 50]},
            'bar': {'color': psnr_color},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 30], 'color': "yellow"},
                {'range': [30, 50], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 35
            }
        }
    ), row=1, col=1)
    
    # SSIM Gauge (0-1 range)
    ssim_value = compression_data.get('ssim', 0)
    ssim_color = get_metric_color(ssim_value, [0.6, 0.8, 0.9])
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=ssim_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "SSIM"},
        delta={'reference': 0.8, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': ssim_color},
            'steps': [
                {'range': [0, 0.6], 'color': "lightgray"},
                {'range': [0.6, 0.8], 'color': "yellow"},
                {'range': [0.8, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.9
            }
        }
    ), row=1, col=2)
    
    # Compression Ratio Gauge
    compression_ratio = compression_data.get('compression_ratio', 1)
    compression_color = get_metric_color(compression_ratio, [2, 5, 10])
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=compression_ratio,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Compression Ratio"},
        gauge={
            'axis': {'range': [None, 20]},
            'bar': {'color': compression_color},
            'steps': [
                {'range': [0, 2], 'color': "lightgray"},
                {'range': [2, 5], 'color': "yellow"},
                {'range': [5, 20], 'color': "lightgreen"}
            ]
        }
    ), row=2, col=1)
    
    # Overall Quality Score
    quality_score = compression_data['quality_score']
    quality_color = get_metric_color(quality_score, [60, 80, 90])
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=quality_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Quality Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': quality_color},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ]
        }
    ), row=2, col=2)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Image Compression Quality Dashboard",
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Responsive summary metrics in cards
    st.markdown("""
    <style>
    .responsive-metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Tablet Styles */
    @media (max-width: 1024px) {
        .responsive-metrics-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.75rem;
        }
    }
    
    /* Mobile Styles */
    @media (max-width: 768px) {
        .responsive-metrics-grid {
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
        }
    }
    
    /* Small Mobile Styles */
    @media (max-width: 480px) {
        .responsive-metrics-grid {
            grid-template-columns: 1fr;
            gap: 0.5rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create responsive metric columns
    # Desktop: 4 columns, Tablet: 2 columns, Mobile: 1-2 columns
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.metric(
            "PSNR Quality", 
            f"{psnr_value:.2f} dB",
            delta=f"{get_quality_assessment_text(psnr_value, 'psnr')}"
        )
    
    with col2:
        st.metric(
            "SSIM Similarity", 
            f"{ssim_value:.3f}",
            delta=f"{get_quality_assessment_text(ssim_value, 'ssim')}"
        )
    
    with col3:
        st.metric(
            "Compression", 
            f"{compression_ratio:.1f}:1",
            delta=f"{((compression_ratio - 1) / compression_ratio * 100):.1f}% saved"
        )
    
    with col4:
        st.metric(
            "Overall Score", 
            f"{quality_score:.1f}/100",
            delta=f"{get_quality_assessment_text(quality_score, 'overall')}"
        )


def create_image_comparison_component(original_image, compressed_image, compression_data):
    """Create side-by-side image comparison with zoom and pan functionality."""
    
    st.markdown("### 🖼️ Interactive Image Comparison")
    
    # Comparison mode selector
    comparison_mode = st.radio(
        "Comparison View:",
        ["Side-by-Side", "Before/After Slider", "Grid View", "Overlay Mode"],
        horizontal=True,
        help="Choose how to display the image comparison"
    )
    
    if comparison_mode == "Side-by-Side":
        create_side_by_side_comparison(original_image, compressed_image, compression_data)
    
    elif comparison_mode == "Before/After Slider":
        create_slider_comparison(original_image, compressed_image, compression_data)
    
    elif comparison_mode == "Grid View":
        create_grid_comparison(original_image, compressed_image, compression_data)
    
    elif comparison_mode == "Overlay Mode":
        create_overlay_comparison(original_image, compressed_image, compression_data)
    
    # Zoom and pan controls
    create_zoom_pan_controls()


def create_side_by_side_comparison(original_image, compressed_image, compression_data):
    """Create responsive side-by-side image comparison."""
    
    # Check if we're on mobile using a simple heuristic
    # In a real implementation, you might use JavaScript to detect screen size
    
    # Create responsive layout
    st.markdown("""
    <style>
    .responsive-image-comparison {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1.5rem;
        margin: 1rem 0;
    }
    
    .image-comparison-item {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .image-comparison-title {
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1f2937;
        font-size: 1.125rem;
    }
    
    /* Tablet Styles */
    @media (max-width: 1024px) {
        .responsive-image-comparison {
            gap: 1rem;
        }
        
        .image-comparison-item {
            padding: 0.75rem;
        }
        
        .image-comparison-title {
            font-size: 1rem;
            margin-bottom: 0.75rem;
        }
    }
    
    /* Mobile Styles */
    @media (max-width: 768px) {
        .responsive-image-comparison {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .image-comparison-item {
            padding: 0.75rem;
            margin: 0 0.5rem;
        }
        
        .image-comparison-title {
            font-size: 0.9375rem;
            text-align: center;
        }
    }
    
    /* Small Mobile Styles */
    @media (max-width: 480px) {
        .responsive-image-comparison {
            gap: 0.75rem;
        }
        
        .image-comparison-item {
            padding: 0.5rem;
            margin: 0 0.25rem;
        }
        
        .image-comparison-title {
            font-size: 0.875rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Use responsive columns that stack on mobile
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="image-comparison-item">', unsafe_allow_html=True)
        st.markdown("**📷 Original Image**")
        create_zoomable_image_container(original_image, "original", "Original")
        
        # Original image stats - make expandable on mobile
        with st.expander("📊 Original Image Stats", expanded=False):
            # Create responsive stats layout
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write(f"**Dimensions:** {original_image.shape[0]} × {original_image.shape[1]}")
                st.write(f"**Channels:** {original_image.shape[2]}")
                st.write(f"**Data Type:** {original_image.dtype}")
            
            with stats_col2:
                st.write(f"**Value Range:** [{original_image.min():.3f}, {original_image.max():.3f}]")
                st.write(f"**Mean:** {original_image.mean():.3f}")
                st.write(f"**Std Dev:** {original_image.std():.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="image-comparison-item">', unsafe_allow_html=True)
        k_value = compression_data.get('k_value', 'N/A')
        st.markdown(f"**🗜️ Compressed Image (k={k_value})**")
        create_zoomable_image_container(compressed_image, "compressed", f"Compressed (k={k_value})")
        
        # Compressed image stats - make expandable on mobile
        with st.expander("📊 Compressed Image Stats", expanded=False):
            # Create responsive stats layout
            stats_col1, stats_col2 = st.columns(2)
            
            with stats_col1:
                st.write(f"**Dimensions:** {compressed_image.shape[0]} × {compressed_image.shape[1]}")
                st.write(f"**K-Value:** {k_value}")
                st.write(f"**Compression:** {compression_data.get('compression_ratio', 'N/A'):.1f}:1")
            
            with stats_col2:
                st.write(f"**PSNR:** {compression_data.get('psnr', 'N/A'):.2f} dB")
                st.write(f"**SSIM:** {compression_data.get('ssim', 'N/A'):.3f}")
                st.write(f"**Mean:** {compressed_image.mean():.3f}")
        
        st.markdown('</div>', unsafe_allow_html=True)


def create_slider_comparison(original_image, compressed_image, compression_data):
    """Create before/after slider comparison."""
    
    st.markdown("**🔄 Before/After Slider Comparison**")
    
    # Create slider for comparison
    slider_value = st.slider(
        "Drag to compare (0 = Original, 100 = Compressed)",
        min_value=0,
        max_value=100,
        value=50,
        help="Slide to see the transition between original and compressed images"
    )
    
    # Blend images based on slider value
    alpha = slider_value / 100.0
    blended_image = (1 - alpha) * original_image + alpha * compressed_image
    
    # Display blended image
    st.image(blended_image, caption=f"Blend: {100-slider_value}% Original + {slider_value}% Compressed", use_column_width=True)
    
    # Show current state info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current View", f"{100-slider_value}% Original")
    with col2:
        st.metric("Blend Ratio", f"{slider_value}% Compressed")
    with col3:
        if slider_value == 0:
            current_quality = "Original Quality"
        elif slider_value == 100:
            current_quality = f"{compression_data.get('quality_score', 0):.1f}/100"
        else:
            # Interpolate quality score
            original_quality = 100
            compressed_quality = compression_data.get('quality_score', 0)
            current_quality = f"{(1-alpha) * original_quality + alpha * compressed_quality:.1f}/100"
        st.metric("Estimated Quality", current_quality)


def create_grid_comparison(original_image, compressed_image, compression_data):
    """Create grid view comparison with difference images."""
    
    st.markdown("**📋 Grid View Comparison**")
    
    # Calculate difference images
    diff_absolute = np.abs(original_image - compressed_image)
    diff_enhanced = np.clip(diff_absolute * 5, 0, 1)  # Enhanced difference
    
    # Create 2x2 grid
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📷 Original**")
        st.image(original_image, use_column_width=True)
        
        st.markdown("**🔍 Absolute Difference**")
        st.image(diff_absolute, use_column_width=True)
    
    with col2:
        k_value = compression_data.get('k_value', 'N/A')
        st.markdown(f"**🗜️ Compressed (k={k_value})**")
        st.image(compressed_image, use_column_width=True)
        
        st.markdown("**🔍 Enhanced Difference (5x)**")
        st.image(diff_enhanced, use_column_width=True)
    
    # Difference statistics
    st.markdown("**📊 Difference Analysis**")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Max Difference", f"{diff_absolute.max():.4f}")
    with col2:
        st.metric("Mean Difference", f"{diff_absolute.mean():.4f}")
    with col3:
        st.metric("Std Difference", f"{diff_absolute.std():.4f}")
    with col4:
        pixels_changed = np.sum(diff_absolute > 0.01) / diff_absolute.size * 100
        st.metric("Pixels Changed", f"{pixels_changed:.1f}%")


def create_overlay_comparison(original_image, compressed_image, compression_data):
    """Create overlay comparison mode."""
    
    st.markdown("**🎭 Overlay Comparison Mode**")
    
    # Overlay controls
    col1, col2 = st.columns(2)
    
    with col1:
        opacity = st.slider(
            "Compressed Image Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Adjust opacity to see overlay effect"
        )
    
    with col2:
        overlay_mode = st.selectbox(
            "Overlay Mode",
            ["Normal", "Difference Highlight", "Error Map"],
            help="Choose overlay visualization mode"
        )
    
    if overlay_mode == "Normal":
        # Simple alpha blending
        overlay_image = (1 - opacity) * original_image + opacity * compressed_image
        caption = f"Overlay: {opacity:.1f} opacity"
    
    elif overlay_mode == "Difference Highlight":
        # Highlight differences in red
        diff = np.abs(original_image - compressed_image)
        diff_threshold = st.slider("Difference Threshold", 0.01, 0.1, 0.05, 0.01)
        
        overlay_image = original_image.copy()
        mask = diff > diff_threshold
        overlay_image[mask] = [1, 0, 0]  # Red for differences
        caption = f"Differences > {diff_threshold:.3f} highlighted in red"
    
    else:  # Error Map
        # Show error intensity as heatmap
        diff = np.abs(original_image - compressed_image)
        error_intensity = np.mean(diff, axis=2)  # Average across channels
        
        # Create heatmap overlay
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Normalize error intensity
        error_norm = (error_intensity - error_intensity.min()) / (error_intensity.max() - error_intensity.min() + 1e-8)
        
        # Apply colormap
        colormap = cm.get_cmap('hot')
        error_colored = colormap(error_norm)[:, :, :3]  # Remove alpha channel
        
        overlay_image = (1 - opacity) * original_image + opacity * error_colored
        caption = f"Error intensity heatmap (opacity: {opacity:.1f})"
    
    st.image(overlay_image, caption=caption, use_column_width=True)


def create_zoomable_image_container(image, image_id, caption):
    """Create a container with zoom and pan functionality."""
    
    # Create unique container ID
    container_id = f"zoom_container_{image_id}_{int(time.time() * 1000)}"
    
    # Add zoom controls
    col1, col2, col3, col4 = st.columns([1, 1, 1, 5])
    
    with col1:
        if st.button("🔍+", key=f"zoom_in_{container_id}", help="Zoom In"):
            st.session_state[f"zoom_{container_id}"] = st.session_state.get(f"zoom_{container_id}", 1.0) * 1.2
    
    with col2:
        if st.button("🔍-", key=f"zoom_out_{container_id}", help="Zoom Out"):
            st.session_state[f"zoom_{container_id}"] = max(st.session_state.get(f"zoom_{container_id}", 1.0) / 1.2, 0.5)
    
    with col3:
        if st.button("↺", key=f"reset_{container_id}", help="Reset Zoom"):
            st.session_state[f"zoom_{container_id}"] = 1.0
    
    with col4:
        current_zoom = st.session_state.get(f"zoom_{container_id}", 1.0)
        st.write(f"Zoom: {current_zoom:.1f}x")
    
    # Display image with zoom effect (simulated with caption)
    zoom_level = st.session_state.get(f"zoom_{container_id}", 1.0)
    zoom_caption = f"{caption} (Zoom: {zoom_level:.1f}x)" if zoom_level != 1.0 else caption
    
    st.image(image, caption=zoom_caption, use_column_width=True)


def create_zoom_pan_controls():
    """Create global zoom and pan controls."""
    
    with st.expander("🔧 Advanced View Controls"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Zoom Controls**")
            st.write("• Use zoom buttons above each image")
            st.write("• Zoom range: 0.5x to 3.0x")
            st.write("• Reset button returns to 1.0x")
        
        with col2:
            st.markdown("**🎛️ Display Options**")
            
            show_grid = st.checkbox("Show pixel grid at high zoom", value=False)
            show_coordinates = st.checkbox("Show pixel coordinates", value=False)
            sync_zoom = st.checkbox("Synchronize zoom between images", value=True)
            
            if show_grid:
                st.info("Pixel grid will be visible at zoom levels > 2.0x")
            
            if show_coordinates:
                st.info("Click on images to see pixel coordinates and values")
            
            if sync_zoom:
                st.info("Zoom actions will apply to all images simultaneously")


def create_detailed_analysis_component(original_image, compressed_image, compression_data):
    """Create detailed analysis with histograms and advanced metrics."""
    
    st.markdown("### 🔬 Detailed Analysis")
    
    # Analysis tabs
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs([
        "📈 Histograms", 
        "📊 Statistical Analysis", 
        "🎯 Quality Assessment"
    ])
    
    with analysis_tab1:
        create_histogram_analysis(original_image, compressed_image)
    
    with analysis_tab2:
        create_statistical_analysis(original_image, compressed_image, compression_data)
    
    with analysis_tab3:
        create_quality_assessment(compression_data)


def create_histogram_analysis(original_image, compressed_image):
    """Create histogram comparison analysis."""
    
    st.markdown("**📈 Histogram Comparison**")
    
    # Create histogram comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Original RGB Histogram', 'Compressed RGB Histogram', 
                       'Original Grayscale', 'Compressed Grayscale'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # RGB histograms
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        # Original RGB
        fig.add_trace(
            go.Histogram(
                x=original_image[:, :, i].flatten(),
                nbinsx=50,
                name=f'Original {color.title()}',
                marker_color=color,
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Compressed RGB
        fig.add_trace(
            go.Histogram(
                x=compressed_image[:, :, i].flatten(),
                nbinsx=50,
                name=f'Compressed {color.title()}',
                marker_color=color,
                opacity=0.7
            ),
            row=1, col=2
        )
    
    # Grayscale histograms
    orig_gray = np.dot(original_image[...,:3], [0.2989, 0.5870, 0.1140])
    comp_gray = np.dot(compressed_image[...,:3], [0.2989, 0.5870, 0.1140])
    
    fig.add_trace(
        go.Histogram(
            x=orig_gray.flatten(),
            nbinsx=50,
            name='Original Gray',
            marker_color='gray',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=comp_gray.flatten(),
            nbinsx=50,
            name='Compressed Gray',
            marker_color='darkgray',
            opacity=0.7
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Histogram statistics
    st.markdown("**📊 Histogram Statistics**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Image**")
        st.write(f"Mean: {original_image.mean():.4f}")
        st.write(f"Std: {original_image.std():.4f}")
        st.write(f"Min: {original_image.min():.4f}")
        st.write(f"Max: {original_image.max():.4f}")
        st.write(f"Median: {np.median(original_image):.4f}")
    
    with col2:
        st.markdown("**Compressed Image**")
        st.write(f"Mean: {compressed_image.mean():.4f}")
        st.write(f"Std: {compressed_image.std():.4f}")
        st.write(f"Min: {compressed_image.min():.4f}")
        st.write(f"Max: {compressed_image.max():.4f}")
        st.write(f"Median: {np.median(compressed_image):.4f}")


def create_statistical_analysis(original_image, compressed_image, compression_data):
    """Create statistical analysis of the compression."""
    
    st.markdown("**📊 Statistical Analysis**")
    
    # Calculate additional metrics
    diff = original_image - compressed_image
    abs_diff = np.abs(diff)
    
    # Correlation analysis
    orig_flat = original_image.flatten()
    comp_flat = compressed_image.flatten()
    correlation = np.corrcoef(orig_flat, comp_flat)[0, 1]
    
    # Energy preservation
    orig_energy = np.sum(original_image ** 2)
    comp_energy = np.sum(compressed_image ** 2)
    energy_preservation = comp_energy / orig_energy * 100
    
    # Display metrics in organized layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔗 Correlation Analysis**")
        st.metric("Pixel Correlation", f"{correlation:.4f}")
        st.metric("R² Score", f"{correlation**2:.4f}")
        
        correlation_quality = "Excellent" if correlation > 0.95 else "Good" if correlation > 0.9 else "Fair"
        st.write(f"**Quality:** {correlation_quality}")
    
    with col2:
        st.markdown("**⚡ Energy Analysis**")
        st.metric("Energy Preserved", f"{energy_preservation:.1f}%")
        st.metric("Energy Lost", f"{100 - energy_preservation:.1f}%")
        
        energy_quality = "Excellent" if energy_preservation > 95 else "Good" if energy_preservation > 90 else "Fair"
        st.write(f"**Quality:** {energy_quality}")
    
    with col3:
        st.markdown("**📏 Error Analysis**")
        st.metric("Max Error", f"{abs_diff.max():.4f}")
        st.metric("Mean Error", f"{abs_diff.mean():.4f}")
        st.metric("Error Std", f"{abs_diff.std():.4f}")
    
    # Error distribution plot
    st.markdown("**📈 Error Distribution**")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=abs_diff.flatten(),
        nbinsx=50,
        name='Absolute Error Distribution',
        marker_color='red',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Distribution of Absolute Pixel Errors",
        xaxis_title="Absolute Error",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def create_quality_assessment(compression_data):
    """Create comprehensive quality assessment."""
    
    st.markdown("**🎯 Quality Assessment**")
    
    # Overall quality score
    quality_score = compression_data.get('quality_score', 0)
    psnr = compression_data.get('psnr', 0)
    ssim = compression_data.get('ssim', 0)
    compression_ratio = compression_data.get('compression_ratio', 1)
    
    # Quality assessment text
    overall_assessment = get_overall_quality_assessment(quality_score)
    
    # Display assessment
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Overall Assessment:** {overall_assessment}")
        
        # Detailed breakdown
        st.markdown("**Quality Breakdown:**")
        
        psnr_assessment = get_quality_assessment_text(psnr, 'psnr')
        ssim_assessment = get_quality_assessment_text(ssim, 'ssim')
        compression_assessment = get_compression_assessment(compression_ratio)
        
        st.write(f"• **PSNR Quality:** {psnr_assessment}")
        st.write(f"• **SSIM Quality:** {ssim_assessment}")
        st.write(f"• **Compression Efficiency:** {compression_assessment}")
        
        # Recommendations
        st.markdown("**💡 Recommendations:**")
        recommendations = generate_quality_recommendations(compression_data)
        for rec in recommendations:
            st.write(f"• {rec}")
    
    with col2:
        # Quality score visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=quality_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Quality"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': get_metric_color(quality_score, [60, 80, 90])},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def create_download_component(original_image, compressed_image, compression_data, filename):
    """Create comprehensive download component with format options."""
    
    st.markdown("### 💾 Download Results")
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📥 Download Options**")
        
        # Format selection
        download_format = st.selectbox(
            "Image Format",
            ["PNG", "JPEG", "TIFF", "BMP"],
            help="Choose the format for downloaded images"
        )
        
        # Quality settings for JPEG
        if download_format == "JPEG":
            jpeg_quality = st.slider(
                "JPEG Quality",
                min_value=1,
                max_value=100,
                value=95,
                help="JPEG compression quality (higher = better quality, larger file)"
            )
        else:
            jpeg_quality = 95
        
        # Additional options
        include_metadata = st.checkbox("Include compression metadata", value=True)
        create_comparison_image = st.checkbox("Create side-by-side comparison", value=False)
        
    with col2:
        st.markdown("**📊 File Size Estimates**")
        
        # Estimate file sizes
        original_size_est = estimate_file_size(original_image, download_format, jpeg_quality)
        compressed_size_est = estimate_file_size(compressed_image, download_format, jpeg_quality)
        
        st.metric("Original Image", f"~{original_size_est:.1f} KB")
        st.metric("Compressed Image", f"~{compressed_size_est:.1f} KB")
        st.metric("Size Reduction", f"~{((original_size_est - compressed_size_est) / original_size_est * 100):.1f}%")
    
    # Download buttons
    st.markdown("**📁 Download Files**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Download compressed image
        compressed_data = prepare_image_download(
            compressed_image, 
            f"compressed_{filename}", 
            download_format, 
            jpeg_quality,
            compression_data if include_metadata else None
        )
        
        st.download_button(
            label="📥 Compressed Image",
            data=compressed_data,
            file_name=f"compressed_{filename}.{download_format.lower()}",
            mime=f"image/{download_format.lower()}",
            use_container_width=True
        )
    
    with col2:
        # Download original image
        original_data = prepare_image_download(
            original_image, 
            f"original_{filename}", 
            download_format, 
            jpeg_quality
        )
        
        st.download_button(
            label="📥 Original Image",
            data=original_data,
            file_name=f"original_{filename}.{download_format.lower()}",
            mime=f"image/{download_format.lower()}",
            use_container_width=True
        )
    
    with col3:
        # Download difference image
        diff_image = np.abs(original_image - compressed_image)
        diff_enhanced = np.clip(diff_image * 5, 0, 1)  # Enhanced for visibility
        
        diff_data = prepare_image_download(
            diff_enhanced, 
            f"difference_{filename}", 
            download_format, 
            jpeg_quality
        )
        
        st.download_button(
            label="📥 Difference Image",
            data=diff_data,
            file_name=f"difference_{filename}.{download_format.lower()}",
            mime=f"image/{download_format.lower()}",
            use_container_width=True
        )
    
    with col4:
        # Download comprehensive report
        report_data = generate_comprehensive_report(compression_data, filename)
        
        st.download_button(
            label="📄 Full Report",
            data=report_data,
            file_name=f"compression_report_{filename}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Optional: Create comparison image
    if create_comparison_image:
        st.markdown("**🖼️ Side-by-Side Comparison Download**")
        
        comparison_image = create_comparison_image_for_download(original_image, compressed_image, compression_data)
        comparison_data = prepare_image_download(
            comparison_image, 
            f"comparison_{filename}", 
            download_format, 
            jpeg_quality
        )
        
        st.download_button(
            label="📥 Download Comparison Image",
            data=comparison_data,
            file_name=f"comparison_{filename}.{download_format.lower()}",
            mime=f"image/{download_format.lower()}",
            use_container_width=True
        )


# Helper functions

def calculate_quality_score(psnr, ssim):
    """Calculate a composite quality score from PSNR and SSIM."""
    # Normalize PSNR to 0-100 scale (assuming max useful PSNR is 50)
    psnr_normalized = min(psnr / 50 * 100, 100)
    # SSIM is already 0-1, convert to 0-100
    ssim_normalized = ssim * 100
    # Weighted average (SSIM is more perceptually relevant)
    return (psnr_normalized * 0.4 + ssim_normalized * 0.6)


def get_metric_color(value, thresholds):
    """Get color based on metric value and thresholds."""
    if value >= thresholds[2]:
        return "green"
    elif value >= thresholds[1]:
        return "yellow"
    elif value >= thresholds[0]:
        return "orange"
    else:
        return "red"


def get_quality_assessment_text(value, metric_type):
    """Get quality assessment text for different metrics."""
    if metric_type == 'psnr':
        if value >= 35:
            return "Excellent"
        elif value >= 25:
            return "Good"
        elif value >= 20:
            return "Fair"
        else:
            return "Poor"
    
    elif metric_type == 'ssim':
        if value >= 0.9:
            return "Excellent"
        elif value >= 0.7:
            return "Good"
        elif value >= 0.5:
            return "Fair"
        else:
            return "Poor"
    
    elif metric_type == 'overall':
        if value >= 90:
            return "Excellent"
        elif value >= 80:
            return "Very Good"
        elif value >= 70:
            return "Good"
        elif value >= 60:
            return "Fair"
        else:
            return "Needs Improvement"
    
    return "Unknown"


def get_overall_quality_assessment(quality_score):
    """Get overall quality assessment with emoji."""
    if quality_score >= 90:
        return "🟢 Excellent Quality - Outstanding compression with minimal quality loss"
    elif quality_score >= 80:
        return "🟡 Very Good Quality - High quality with good compression efficiency"
    elif quality_score >= 70:
        return "🟠 Good Quality - Acceptable quality with reasonable compression"
    elif quality_score >= 60:
        return "🔴 Fair Quality - Noticeable quality loss but still usable"
    else:
        return "⚫ Poor Quality - Significant quality degradation, consider higher k-value"


def get_compression_assessment(compression_ratio):
    """Get compression efficiency assessment."""
    if compression_ratio >= 10:
        return "Excellent compression efficiency"
    elif compression_ratio >= 5:
        return "Good compression efficiency"
    elif compression_ratio >= 2:
        return "Moderate compression efficiency"
    else:
        return "Low compression efficiency"


def generate_quality_recommendations(compression_data):
    """Generate quality improvement recommendations."""
    recommendations = []
    
    psnr = compression_data.get('psnr', 0)
    ssim = compression_data.get('ssim', 0)
    k_value = compression_data.get('k_value', 0)
    compression_ratio = compression_data.get('compression_ratio', 1)
    quality_score = compression_data.get('quality_score', 0)
    
    if psnr < 25:
        recommendations.append("Consider increasing k-value to improve PSNR (target: >25 dB)")
    
    if ssim < 0.7:
        recommendations.append("Increase k-value to improve structural similarity (target: >0.7)")
    
    if compression_ratio < 2:
        recommendations.append("Decrease k-value for better compression efficiency")
    
    if quality_score > 85:
        recommendations.append("Excellent results! Current settings provide optimal balance")
    elif quality_score < 60:
        recommendations.append("Consider adjusting k-value for better quality-compression balance")
    
    if k_value < 10:
        recommendations.append("Very low k-value may cause significant quality loss")
    elif k_value > 100:
        recommendations.append("High k-value may not provide significant compression benefits")
    
    if not recommendations:
        recommendations.append("Current settings provide good balance between quality and compression")
    
    return recommendations


def estimate_file_size(image, format_type, jpeg_quality=95):
    """Estimate file size in KB for different formats."""
    height, width, channels = image.shape
    pixels = height * width
    
    if format_type == "PNG":
        # PNG: roughly 3-4 bytes per pixel for RGB
        return pixels * 3.5 / 1024
    elif format_type == "JPEG":
        # JPEG: varies with quality, roughly 0.5-2 bytes per pixel
        quality_factor = jpeg_quality / 100
        return pixels * (0.5 + 1.5 * quality_factor) / 1024
    elif format_type == "TIFF":
        # TIFF: roughly 3 bytes per pixel uncompressed
        return pixels * 3 / 1024
    elif format_type == "BMP":
        # BMP: 3 bytes per pixel + header
        return (pixels * 3 + 54) / 1024
    else:
        return pixels * 3 / 1024  # Default estimate


def prepare_image_download(image, filename, format_type, jpeg_quality=95, metadata=None):
    """Prepare image data for download with specified format."""
    # Convert to PIL Image
    image_pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    
    # Create buffer
    img_buffer = io.BytesIO()
    
    # Save with appropriate format and settings
    if format_type == "JPEG":
        image_pil.save(img_buffer, format='JPEG', quality=jpeg_quality, optimize=True)
    elif format_type == "PNG":
        image_pil.save(img_buffer, format='PNG', optimize=True)
    elif format_type == "TIFF":
        image_pil.save(img_buffer, format='TIFF')
    elif format_type == "BMP":
        image_pil.save(img_buffer, format='BMP')
    else:
        image_pil.save(img_buffer, format='PNG')
    
    img_buffer.seek(0)
    return img_buffer.getvalue()


def create_comparison_image_for_download(original_image, compressed_image, compression_data):
    """Create a side-by-side comparison image for download."""
    height, width, channels = original_image.shape
    
    # Create comparison image (side by side)
    comparison = np.zeros((height, width * 2, channels))
    comparison[:, :width, :] = original_image
    comparison[:, width:, :] = compressed_image
    
    return comparison


def generate_comprehensive_report(compression_data, filename):
    """Generate a comprehensive text report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
SVD IMAGE COMPRESSION - COMPREHENSIVE ANALYSIS REPORT
====================================================

ANALYSIS TIMESTAMP: {timestamp}
FILENAME: {filename}

COMPRESSION SETTINGS
===================
K-Value: {compression_data.get('k_value', 'N/A')}
Processing Mode: {compression_data.get('mode', 'RGB')}
Compression Ratio: {compression_data.get('compression_ratio', 'N/A'):.2f}:1

QUALITY METRICS
===============
PSNR (Peak Signal-to-Noise Ratio): {compression_data.get('psnr', 0):.2f} dB
SSIM (Structural Similarity Index): {compression_data.get('ssim', 0):.4f}
MSE (Mean Squared Error): {compression_data.get('mse', 0):.6f}
Overall Quality Score: {compression_data.get('quality_score', 0):.1f}/100

QUALITY ASSESSMENT
==================
{get_overall_quality_assessment(compression_data.get('quality_score', 0))}

DETAILED ANALYSIS
=================
PSNR Assessment: {get_quality_assessment_text(compression_data.get('psnr', 0), 'psnr')}
SSIM Assessment: {get_quality_assessment_text(compression_data.get('ssim', 0), 'ssim')}
Compression Efficiency: {get_compression_assessment(compression_data.get('compression_ratio', 1))}

RECOMMENDATIONS
===============
"""
    
    recommendations = generate_quality_recommendations(compression_data)
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"
    
    report += f"""

TECHNICAL DETAILS
=================
SVD (Singular Value Decomposition) compresses images by:
1. Decomposing the image matrix into U, Σ, and V^T matrices
2. Keeping only the top k singular values and corresponding vectors
3. Reconstructing an approximation using these k components

Storage Reduction: From O(m×n) to O(k×(m+n)) where k << min(m,n)
Quality depends on the energy captured by the top k singular values.

GENERATED BY
============
SVD Image Compression Tool - Advanced Results Display
Report generated on {timestamp}
"""
    
    return report