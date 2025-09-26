"""
Streamlined compression controls component with real-time preview and intuitive interface.
Enhanced with comprehensive tooltip system and contextual help.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .tooltip_system import tooltip_system, create_tooltip


def create_compression_controls_panel(
    original_image: np.ndarray,
    max_k: Optional[int] = None,
    default_k: int = 20,
    enable_real_time: bool = True,
    show_advanced: bool = True,
    show_preview: bool = True
) -> Dict[str, Any]:
    """
    Create a streamlined compression controls panel with real-time preview.
    
    Args:
        original_image: The original image array
        max_k: Maximum k value (auto-calculated if None)
        default_k: Default k value
        enable_real_time: Enable real-time compression preview
        show_advanced: Show advanced options panel
        
    Returns:
        Dictionary containing all compression parameters
    """
    
    if max_k is None:
        max_k = min(256, original_image.shape[0])
    
    # Initialize session state for compression controls
    if 'compression_params' not in st.session_state:
        st.session_state.compression_params = {
            'k_value': default_k,
            'mode': 'RGB (Color)',
            'real_time_enabled': enable_real_time,
            'quality_preset': 'Medium',
            'show_difference': True,
            'show_singular_values': True,
            'difference_enhancement': 5
        }
    
    # Create main controls container with enhanced help system
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("## ⚙️ Compression Controls")
    with col2:
        # Enhanced keyboard shortcuts with tooltips
        shortcuts_tooltip = """
        **Keyboard Shortcuts:**
        • ↑/↓: Adjust k-value by ±1
        • Shift+↑/↓: Adjust k-value by ±10
        • R: Toggle real-time preview
        • G: Switch to grayscale mode
        • C: Switch to color mode
        • 1-5: Apply quality presets
        • H: Toggle help mode
        • ?: Show this help
        """
        st.markdown(
            create_tooltip(
                '<button style="background: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 6px; cursor: pointer;">⌨️ Shortcuts</button>',
                shortcuts_tooltip,
                position="bottom"
            ),
            unsafe_allow_html=True
        )
    with col3:
        # Contextual help button
        tooltip_system.create_help_button("compression_controls", button_text="📖 Help", button_style="full")
    
    # Add accessibility improvements
    st.markdown(
        """
        <style>
        /* Improve focus visibility for accessibility */
        .stSlider > div > div > div > div {
            outline: 2px solid transparent;
            transition: outline 0.2s ease;
        }
        
        .stSlider > div > div > div > div:focus-within {
            outline: 2px solid #3b82f6;
            outline-offset: 2px;
        }
        
        .stButton > button:focus {
            outline: 2px solid #3b82f6;
            outline-offset: 2px;
        }
        
        /* High contrast mode support */
        @media (prefers-contrast: high) {
            .stButton > button {
                border: 2px solid currentColor;
            }
        }
        
        /* Reduced motion support */
        @media (prefers-reduced-motion: reduce) {
            * {
                transition: none !important;
                animation: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Real-time toggle and quality indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Real-time toggle with enhanced tooltip
        realtime_tooltip = """
        **Real-time Preview**
        
        Automatically compresses image as you adjust parameters.
        
        **When Enabled:**
        • Immediate feedback as you change settings
        • Uses smaller image size for faster processing
        • Great for experimenting with different values
        
        **When Disabled:**
        • Manual compression with "Compress Image" button
        • Better performance for large images
        • Reduces CPU usage during parameter adjustment
        
        **Performance Tips:**
        • Disable for images larger than 1024x1024
        • Enable for quick experimentation
        • Preview uses downsampled image for speed
        
        **Keyboard shortcut:** Press R to toggle
        """
        
        st.markdown(
            create_tooltip(
                f'<label style="display: flex; align-items: center; gap: 8px; cursor: pointer;"><input type="checkbox" {"checked" if st.session_state.compression_params["real_time_enabled"] else ""}> 🔄 Real-time Preview</label>',
                realtime_tooltip,
                position="bottom"
            ),
            unsafe_allow_html=True
        )
        
        real_time = st.toggle(
            "🔄 Real-time Preview",
            value=st.session_state.compression_params['real_time_enabled'],
            help="Auto-compress as you adjust parameters. Press R to toggle quickly."
        )
        st.session_state.compression_params['real_time_enabled'] = real_time
    
    with col2:
        # Quality indicator with detailed tooltip
        quality_score = _calculate_quality_indicator(st.session_state.compression_params['k_value'], max_k)
        quality_color = _get_quality_color(quality_score)
        
        quality_tooltip = f"""
        **Quality Indicator: {quality_score}**
        
        Based on k-value relative to image dimensions.
        
        **Current k-value:** {st.session_state.compression_params['k_value']} / {max_k}
        **Quality ratio:** {(st.session_state.compression_params['k_value'] / max_k * 100):.1f}%
        
        **Quality Levels:**
        • Excellent (>40%): Minimal visible artifacts
        • Good (20-40%): Slight artifacts in detailed areas
        • Fair (10-20%): Noticeable but acceptable quality loss
        • Poor (<10%): Significant quality degradation
        
        **Tips:**
        • Higher k-values generally mean better quality
        • Optimal k depends on image content and use case
        • Use quality metrics (PSNR, SSIM) for precise assessment
        """
        
        quality_badge = f"""
        <div style="
            background: {quality_color}; 
            color: white; 
            padding: 8px 16px; 
            border-radius: 20px; 
            text-align: center;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: help;
        ">
            Quality: {quality_score}
        </div>
        """
        
        st.markdown(
            create_tooltip(quality_badge, quality_tooltip, position="bottom"),
            unsafe_allow_html=True
        )
    
    with col3:
        # Compression ratio indicator with detailed tooltip
        compression_ratio = _estimate_compression_ratio(st.session_state.compression_params['k_value'], original_image.shape)
        
        ratio_tooltip = f"""
        **Compression Ratio: {compression_ratio:.1f}:1**
        
        Estimated space savings with current settings.
        
        **Current k-value:** {st.session_state.compression_params['k_value']}
        **Image dimensions:** {original_image.shape[0]} × {original_image.shape[1]}
        
        **What this means:**
        • Original size: {original_image.shape[0] * original_image.shape[1] * (3 if len(original_image.shape) == 3 else 1):,} values
        • Compressed size: ~{int(original_image.shape[0] * original_image.shape[1] * (3 if len(original_image.shape) == 3 else 1) / compression_ratio):,} values
        • Space saved: {((compression_ratio - 1) / compression_ratio * 100):.1f}%
        
        **Compression Levels:**
        • 2:1 - 5:1: Moderate compression
        • 5:1 - 10:1: High compression
        • 10:1+: Very high compression
        
        **Note:** Actual file size depends on format and encoding
        """
        
        ratio_badge = f"""
        <div style="
            background: #3b82f6; 
            color: white; 
            padding: 8px 16px; 
            border-radius: 20px; 
            text-align: center;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: help;
        ">
            Ratio: {compression_ratio:.1f}:1
        </div>
        """
        
        st.markdown(
            create_tooltip(ratio_badge, ratio_tooltip, position="bottom"),
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Main controls layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Enhanced k-value slider with comprehensive tooltips
        k_tooltip = f"""
        **K-Value (Compression Level)**
        
        Controls how many singular values to keep from SVD decomposition.
        
        **Current: {st.session_state.compression_params['k_value']} / {max_k}**
        
        **Guidelines:**
        • 2-10: High compression (thumbnails, previews)
        • 20-50: Balanced (web images, documents)  
        • 100+: High quality (archival, detailed images)
        
        **Tips:**
        • Lower k = Higher compression, lower quality
        • Higher k = Lower compression, higher quality
        • Use energy-based selection for optimal results
        
        **Keyboard:** ↑/↓ to adjust by 1, Shift+↑/↓ by 10
        """
        
        st.markdown(
            create_tooltip(
                '<h3 style="margin: 0;">🎛️ Compression Level</h3>',
                k_tooltip,
                position="right"
            ),
            unsafe_allow_html=True
        )
        
        # Create k-value slider with enhanced help
        k_value = st.slider(
            label="",  # Empty label since we have custom header
            min_value=1,
            max_value=max_k,
            value=st.session_state.compression_params['k_value'],
            step=1,
            key="k_slider_main",
            help=f"Singular values to keep (1-{max_k}). Use ↑/↓ keys for fine adjustment."
        )
        
        # Visual k-value indicator with energy retention
        _create_k_value_indicator(k_value, max_k, original_image)
        
        # Processing mode selection with detailed tooltips
        mode_tooltip = """
        **Processing Mode**
        
        Determines how color channels are processed during compression.
        
        **RGB (Color):**
        • Processes red, green, blue channels separately
        • Best for: Color photos, artwork, detailed images
        • Higher quality but larger compressed size
        
        **Grayscale:**
        • Converts to grayscale first, then compresses
        • Best for: Text documents, diagrams, B&W photos
        • Higher compression ratios, smaller file sizes
        • Faster processing and less memory usage
        
        **Tips:**
        • Use RGB for color-critical applications
        • Use Grayscale for documents and diagrams
        
        **Keyboard:** G for grayscale, C for color
        """
        
        st.markdown(
            create_tooltip(
                '<h3 style="margin: 0;">🎨 Processing Mode</h3>',
                mode_tooltip,
                position="right"
            ),
            unsafe_allow_html=True
        )
        
        mode = st.selectbox(
            label="",  # Empty label since we have custom header
            options=["RGB (Color)", "Grayscale"],
            index=0 if st.session_state.compression_params['mode'] == 'RGB (Color)' else 1,
            key="mode_select_main",
            help="Choose how color channels are processed. Use G/C keys to switch quickly."
        )
        
        # Update session state
        st.session_state.compression_params['k_value'] = k_value
        st.session_state.compression_params['mode'] = mode
    
    with col2:
        # Quality presets panel with enhanced tooltips
        presets_tooltip = """
        **Quality Presets**
        
        Pre-configured compression settings for common use cases.
        
        **Preset Guidelines:**
        • Ultra Low (k=2): Thumbnails, previews, maximum space saving
        • Low (k=5): Basic quality, high compression for web
        • Medium (k=20): Balanced quality/compression for most uses
        • High (k=50): Professional quality, moderate compression
        • Ultra High (k=100+): Archival quality, minimal compression
        
        **Tips:**
        • Start with Medium for most images
        • Use Low for thumbnails and previews
        • Use High for professional applications
        
        **Keyboard:** Press 1-5 to apply presets quickly
        """
        
        st.markdown(
            create_tooltip(
                '<h3 style="margin: 0;">🎯 Quick Presets</h3>',
                presets_tooltip,
                position="left"
            ),
            unsafe_allow_html=True
        )
        
        presets = [
            ("🔴 Ultra Low", 2, "Maximum compression, lowest quality (k=2)"),
            ("🟠 Low", 5, "High compression, basic quality (k=5)"),
            ("🟡 Medium", 20, "Balanced compression and quality (k=20)"),
            ("🟢 High", 50, "Low compression, high quality (k=50)"),
            ("🔵 Ultra High", min(100, max_k), f"Minimal compression, maximum quality (k={min(100, max_k)})")
        ]
        
        for i, (label, k_val, description) in enumerate(presets, 1):
            preset_tooltip = f"""
            **{label.split(' ', 1)[1]} Quality Preset**
            
            {description}
            
            **Recommended for:**
            {self._get_preset_recommendations(label)}
            
            **Keyboard shortcut:** Press {i}
            """
            
            button_html = f"""
            <button onclick="applyPreset({k_val})" style="
                width: 100%;
                background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                color: white;
                border: none;
                padding: 10px;
                border-radius: 8px;
                cursor: pointer;
                margin: 2px 0;
                font-weight: 500;
                transition: all 0.2s ease;
            " onmouseover="this.style.transform='translateY(-1px)'" 
               onmouseout="this.style.transform='translateY(0)'">
                {label}
            </button>
            """
            
            st.markdown(
                create_tooltip(button_html, preset_tooltip, position="left"),
                unsafe_allow_html=True
            )
            
            # Hidden button for actual functionality
            if st.button(
                label, 
                use_container_width=True, 
                key=f"preset_{k_val}_main",
                help=f"{description} - Press {i} for quick access"
            ):
                st.session_state.compression_params['k_value'] = k_val
                st.rerun()
        
        # Smart recommendations with detailed tooltips
        st.markdown("---")
        
        smart_tooltip = """
        **Smart Recommendations**
        
        AI-powered suggestions for optimal compression settings.
        
        **Auto-Optimize:**
        • Analyzes image content and complexity
        • Considers edge density and texture patterns
        • Suggests k-value for best quality/compression balance
        
        **Energy-Based Selection:**
        • 90% Energy: Good visual quality for most applications
        • 95% Energy: Excellent quality with moderate compression
        • Based on singular value energy distribution
        
        **Tips:**
        • Auto-optimize works best for natural images
        • Energy-based selection is more consistent
        • Try different options to find your preference
        """
        
        st.markdown(
            create_tooltip(
                '<strong>🧠 Smart Recommendations</strong>',
                smart_tooltip,
                position="left"
            ),
            unsafe_allow_html=True
        )
        
        # Auto-optimize with enhanced tooltip
        auto_optimize_tooltip = """
        **Auto-Optimize**
        
        Analyzes your image to suggest the optimal k-value.
        
        **Analysis includes:**
        • Edge density detection
        • Texture complexity assessment
        • Content type classification
        • Optimal quality/compression balance
        
        **Best for:**
        • Natural photographs
        • Complex images with varied content
        • When unsure about optimal settings
        
        **Note:** May take a few seconds for large images
        """
        
        st.markdown(
            create_tooltip(
                '<button style="width: 100%; background: #10b981; color: white; border: none; padding: 10px; border-radius: 8px; cursor: pointer; margin: 2px 0;">🎯 Auto-Optimize</button>',
                auto_optimize_tooltip,
                position="left"
            ),
            unsafe_allow_html=True
        )
        
        if st.button(
            "🎯 Auto-Optimize", 
            use_container_width=True,
            help="Analyze image content and suggest optimal k value based on complexity"
        ):
            with st.spinner("Analyzing image..."):
                optimal_k = _analyze_image_and_recommend_k(original_image)
                st.session_state.compression_params['k_value'] = min(optimal_k, max_k)
                st.success(f"✨ Recommended k={optimal_k} based on image analysis")
                st.rerun()
        
        # Energy-based recommendations with tooltips
        energy_90_tooltip = """
        **90% Energy Retention**
        
        Automatically selects k-value that retains 90% of image energy.
        
        **What this means:**
        • Preserves 90% of the image's mathematical "energy"
        • Good balance between quality and compression
        • Suitable for most web and display applications
        
        **Advantages:**
        • Consistent quality across different images
        • Based on mathematical principles
        • Reliable for batch processing
        
        **Typical results:**
        • PSNR: 25-35 dB
        • Good visual quality
        • Moderate compression ratios
        """
        
        st.markdown(
            create_tooltip(
                '<button style="width: 100%; background: #f59e0b; color: white; border: none; padding: 10px; border-radius: 8px; cursor: pointer; margin: 2px 0;">⚡ 90% Energy</button>',
                energy_90_tooltip,
                position="left"
            ),
            unsafe_allow_html=True
        )
        
        if st.button(
            "⚡ 90% Energy", 
            use_container_width=True,
            help="Select k-value that retains 90% of image energy - good quality/compression balance"
        ):
            optimal_k = _calculate_energy_based_k(original_image, energy_threshold=0.9)
            st.session_state.compression_params['k_value'] = min(optimal_k, max_k)
            st.success(f"⚡ Set k={optimal_k} for 90% energy retention")
            st.rerun()
        
        energy_95_tooltip = """
        **95% Energy Retention**
        
        Automatically selects k-value that retains 95% of image energy.
        
        **What this means:**
        • Preserves 95% of the image's mathematical "energy"
        • Excellent quality with moderate compression
        • Suitable for professional and archival use
        
        **Advantages:**
        • Higher quality than 90% energy
        • Still provides meaningful compression
        • Good for quality-critical applications
        
        **Typical results:**
        • PSNR: 30-40 dB
        • Excellent visual quality
        • Lower compression ratios
        """
        
        st.markdown(
            create_tooltip(
                '<button style="width: 100%; background: #3b82f6; color: white; border: none; padding: 10px; border-radius: 8px; cursor: pointer; margin: 2px 0;">⚡ 95% Energy</button>',
                energy_95_tooltip,
                position="left"
            ),
            unsafe_allow_html=True
        )
        
        if st.button(
            "⚡ 95% Energy", 
            use_container_width=True,
            help="Select k-value that retains 95% of image energy - excellent quality"
        ):
            optimal_k = _calculate_energy_based_k(original_image, energy_threshold=0.95)
            st.session_state.compression_params['k_value'] = min(optimal_k, max_k)
            st.success(f"⚡ Set k={optimal_k} for 95% energy retention")
            st.rerun()
    
    # Advanced options panel
    if show_advanced:
        with st.expander("🔧 Advanced Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Visualization Options**")
                show_difference = st.checkbox(
                    "Show difference visualization", 
                    value=st.session_state.compression_params['show_difference'],
                    help="Display the difference between original and compressed images"
                )
                
                show_singular_values = st.checkbox(
                    "Show singular values analysis", 
                    value=st.session_state.compression_params['show_singular_values'],
                    help="Display interactive singular values plot and energy analysis"
                )
                
                st.session_state.compression_params['show_difference'] = show_difference
                st.session_state.compression_params['show_singular_values'] = show_singular_values
            
            with col2:
                st.markdown("**Enhancement Options**")
                difference_enhancement = st.slider(
                    "Difference enhancement factor",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.compression_params['difference_enhancement'],
                    help="Multiply difference values to make them more visible (1 = no enhancement)"
                )
                
                st.session_state.compression_params['difference_enhancement'] = difference_enhancement
                
                # Image analysis results (if available)
                if hasattr(st.session_state, 'image_analysis'):
                    st.markdown("**📊 Image Analysis**")
                    analysis = st.session_state.image_analysis
                    st.markdown(f"""
                    - **Content type**: {analysis['reason']}
                    - **Recommended k**: {analysis['recommended_k']}
                    - **90% energy k**: {analysis['k_90']}
                    - **95% energy k**: {analysis['k_95']}
                    - **Edge density**: {analysis['edge_density']:.3f}
                    - **Texture complexity**: {analysis['texture_complexity']:.3f}
                    """)
                
                # Performance options
                st.markdown("**Performance Options**")
                if st.button("🔄 Reset to Defaults", help="Reset all parameters to default values"):
                    st.session_state.compression_params = {
                        'k_value': default_k,
                        'mode': 'RGB (Color)',
                        'real_time_enabled': enable_real_time,
                        'quality_preset': 'Medium',
                        'show_difference': True,
                        'show_singular_values': True,
                        'difference_enhancement': 5
                    }
                    # Clear analysis results
                    if hasattr(st.session_state, 'image_analysis'):
                        delattr(st.session_state, 'image_analysis')
                    st.rerun()
    
    # Real-time preview section (if enabled and real-time is on)
    if show_preview and real_time and k_value != st.session_state.compression_params.get('last_preview_k', -1):
        st.markdown("---")
        st.markdown("### 👁️ Real-time Preview")
        
        with st.spinner("Generating preview..."):
            try:
                # Quick preview with smaller image for performance
                preview_image = _generate_quick_preview(original_image, k_value, mode)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original**")
                    # Show smaller version for preview
                    preview_size = min(200, original_image.shape[0])
                    if original_image.shape[0] > preview_size:
                        step = original_image.shape[0] // preview_size
                        orig_preview = original_image[::step, ::step]
                    else:
                        orig_preview = original_image
                    st.image(orig_preview, use_column_width=True)
                
                with col2:
                    st.markdown(f"**Compressed (k={k_value})**")
                    st.image(preview_image, use_column_width=True)
                
                # Store last preview k to avoid unnecessary recomputation
                st.session_state.compression_params['last_preview_k'] = k_value
                
            except Exception as e:
                st.warning(f"Preview generation failed: {str(e)}")
    
    # Compression action button (for non-real-time mode)
    if not real_time:
        st.markdown("---")
        if st.button(
            "🚀 Compress Image", 
            type="primary", 
            use_container_width=True,
            help="Apply compression with current settings"
        ):
            st.session_state.compression_params['trigger_compression'] = True
    
    return st.session_state.compression_params


def create_compression_tooltip_guide():
    """Create a comprehensive tooltip guide for compression parameters."""
    
    with st.expander("📖 Compression Guide & Tips", expanded=False):
        
        # Create tabs for different topics
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Parameters", "🎯 Quality Tips", "⚡ Performance", "🔬 Technical"])
        
        with tab1:
            st.markdown("""
            ### Understanding Compression Parameters
            
            **K-Value (Compression Level):**
            - Controls how many singular values to keep from the SVD decomposition
            - Lower k = Higher compression, smaller file size, lower quality
            - Higher k = Lower compression, larger file size, higher quality
            - **Recommended ranges:**
              - 2-10: High compression (good for thumbnails, previews)
              - 20-50: Balanced (good for web images, documents)
              - 100+: High quality (good for archival, detailed images)
            
            **Processing Mode:**
            - **RGB (Color)**: Processes red, green, and blue channels separately
              - Best for: Color photos, artwork, detailed images
              - Higher quality but larger compressed size
            - **Grayscale**: Converts image to grayscale first, then compresses
              - Best for: Text documents, diagrams, black & white photos
              - Higher compression ratios, smaller file sizes
            
            **Energy Retention:**
            - Measures how much of the original image's "energy" is preserved
            - Based on the sum of squared singular values
            - **Guidelines:**
              - 90% energy: Good visual quality for most applications
              - 95% energy: Excellent quality with moderate compression
              - 99% energy: Near-lossless quality
            """)
        
        with tab2:
            st.markdown("""
            ### Quality Assessment & Tips
            
            **Quality Indicators:**
            - **Excellent** (k > 40% of max): Minimal visible artifacts, suitable for professional use
            - **Good** (k = 20-40% of max): Slight artifacts in detailed areas, good for web use
            - **Fair** (k = 10-20% of max): Noticeable but acceptable quality loss
            - **Poor** (k < 10% of max): Significant quality degradation, use only for previews
            
            **Choosing the Right K-Value:**
            1. **Start with presets**: Use quality presets as starting points
            2. **Use energy-based selection**: 90% energy is often optimal
            3. **Consider image content**:
               - Smooth gradients: Lower k values work well
               - High detail/texture: Higher k values needed
               - Text/diagrams: Very low k values sufficient
            
            **Visual Quality Metrics:**
            - **PSNR > 30 dB**: Excellent quality
            - **PSNR 25-30 dB**: Good quality
            - **PSNR 20-25 dB**: Fair quality
            - **SSIM > 0.8**: High structural similarity
            - **SSIM 0.6-0.8**: Good structural similarity
            """)
        
        with tab3:
            st.markdown("""
            ### Performance Optimization
            
            **Real-time Preview:**
            - Uses smaller image size for faster processing
            - Automatically enabled for images < 512x512
            - Disable for very large images to improve responsiveness
            
            **Processing Speed Tips:**
            1. **Use Grayscale mode** for faster processing
            2. **Lower k values** process faster
            3. **Disable real-time preview** for large images
            4. **Use presets** instead of manual adjustment for speed
            
            **Memory Usage:**
            - SVD requires temporary storage of U, S, V matrices
            - Memory usage ≈ 3 × image_size for RGB mode
            - Memory usage ≈ 1 × image_size for Grayscale mode
            - Large images (>2048x2048) may require significant memory
            
            **Batch Processing Tips:**
            - Process similar images with same k value
            - Use energy-based k selection for consistent quality
            - Consider grayscale mode for document processing
            """)
        
        with tab4:
            st.markdown("""
            ### Technical Details
            
            **SVD Compression Theory:**
            - Singular Value Decomposition: A = U × Σ × V^T
            - Keeping top k singular values approximates original image
            - Storage requirement: O(k×(m+n)) vs O(m×n) for original
            - Quality depends on energy captured by top k singular values
            
            **Compression Ratio Calculation:**
            ```
            Original size: m × n × channels
            Compressed size: k × (m + n + 1) × channels
            Ratio = Original / Compressed
            ```
            
            **Quality Metrics:**
            - **PSNR**: Peak Signal-to-Noise Ratio (logarithmic scale)
            - **SSIM**: Structural Similarity Index (0-1 scale)
            - **MSE**: Mean Squared Error (lower is better)
            
            **Energy Retention Formula:**
            ```
            Energy = Σ(singular_values²)
            Retention = Σ(first_k_values²) / Total_Energy
            ```
            
            **Optimal K Selection:**
            - Elbow method: Find point where quality improvement diminishes
            - Energy threshold: Select k that retains desired energy %
            - Visual inspection: Balance compression vs quality trade-off
            """)
        
        # Add interactive examples
        st.markdown("---")
        st.markdown("### 🎮 Interactive Examples")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📸 Photo Compression Example", help="Optimal settings for photographs"):
                st.info("**Photo Settings:** k=30-50, RGB mode, 90% energy retention")
        
        with col2:
            if st.button("📄 Document Compression Example", help="Optimal settings for text documents"):
                st.info("**Document Settings:** k=5-15, Grayscale mode, 85% energy retention")


def _create_k_value_indicator(k_value: int, max_k: int, original_image: np.ndarray):
    """Create a visual indicator for the current k-value with energy information."""
    
    # Calculate energy retention for current k
    try:
        if len(original_image.shape) == 3:
            gray_image = np.dot(original_image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = original_image
        
        # Quick SVD for energy calculation (use smaller image for performance)
        if gray_image.shape[0] > 128:
            # Downsample for faster calculation
            step = gray_image.shape[0] // 128
            gray_sample = gray_image[::step, ::step]
        else:
            gray_sample = gray_image
        
        U, s, Vt = np.linalg.svd(gray_sample, full_matrices=False)
        total_energy = np.sum(s**2)
        k_energy = np.sum(s[:k_value]**2) if k_value <= len(s) else total_energy
        energy_retention = (k_energy / total_energy) * 100
        
    except Exception:
        energy_retention = 0
    
    # Create visual progress bar for k-value
    k_percentage = (k_value / max_k) * 100
    
    # Color coding based on k value
    if k_value < 10:
        color = "#ef4444"  # Red
        quality_text = "High Compression"
    elif k_value < 30:
        color = "#f59e0b"  # Yellow
        quality_text = "Balanced"
    else:
        color = "#10b981"  # Green
        quality_text = "High Quality"
    
    st.markdown(
        f"""
        <div style="margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-size: 0.9rem; font-weight: 600;">k = {k_value}</span>
                <span style="font-size: 0.9rem; color: {color};">{quality_text}</span>
            </div>
            <div style="
                width: 100%;
                height: 8px;
                background-color: #e5e7eb;
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    width: {k_percentage}%;
                    height: 100%;
                    background: linear-gradient(90deg, {color} 0%, {color}aa 100%);
                    transition: width 0.3s ease;
                "></div>
            </div>
            <div style="
                display: flex; 
                justify-content: space-between; 
                margin-top: 5px;
                font-size: 0.8rem;
                color: #6b7280;
            ">
                <span>Energy retained: {energy_retention:.1f}%</span>
                <span>Max k: {max_k}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def _calculate_quality_indicator(k_value: int, max_k: int) -> str:
    """Calculate quality indicator based on k value."""
    
    k_ratio = k_value / max_k
    
    if k_ratio > 0.4:
        return "Excellent"
    elif k_ratio > 0.2:
        return "Good"
    elif k_ratio > 0.1:
        return "Fair"
    else:
        return "Poor"


def _get_quality_color(quality: str) -> str:
    """Get color for quality indicator."""
    
    colors = {
        "Excellent": "#10b981",
        "Good": "#3b82f6", 
        "Fair": "#f59e0b",
        "Poor": "#ef4444"
    }
    return colors.get(quality, "#6b7280")


def _estimate_compression_ratio(k_value: int, image_shape: Tuple[int, ...]) -> float:
    """Estimate compression ratio based on k value and image dimensions."""
    
    if len(image_shape) == 3:
        m, n, channels = image_shape
    else:
        m, n = image_shape
        channels = 1
    
    # Original size (in terms of matrix elements)
    original_size = m * n * channels
    
    # Compressed size: U(:, :k) + S(:k) + V(:k, :) for each channel
    compressed_size = channels * (m * k_value + k_value + k_value * n)
    
    return original_size / compressed_size if compressed_size > 0 else 1.0


def _calculate_energy_based_k(image: np.ndarray, energy_threshold: float = 0.9) -> int:
    """Calculate optimal k value based on energy retention threshold."""
    
    try:
        # Convert to grayscale for SVD
        if len(image.shape) == 3:
            gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = image
        
        # Use smaller image for faster calculation
        if gray_image.shape[0] > 128:
            step = gray_image.shape[0] // 128
            gray_sample = gray_image[::step, ::step]
        else:
            gray_sample = gray_image
        
        # Compute SVD
        U, s, Vt = np.linalg.svd(gray_sample, full_matrices=False)
        
        # Calculate cumulative energy
        total_energy = np.sum(s**2)
        cumulative_energy = np.cumsum(s**2) / total_energy
        
        # Find k that achieves desired energy retention
        k_optimal = np.argmax(cumulative_energy >= energy_threshold) + 1
        
        # Scale back to original image size if we downsampled
        if gray_image.shape[0] > 128:
            scale_factor = gray_image.shape[0] / gray_sample.shape[0]
            k_optimal = int(k_optimal * scale_factor)
        
        return max(1, min(k_optimal, min(256, image.shape[0])))
        
    except Exception:
        # Fallback to reasonable default
        return 20


def create_compression_metrics_display(compression_data: Dict[str, Any]):
    """Create a streamlined metrics display for compression results."""
    
    if not compression_data:
        return
    
    st.markdown("### 📊 Compression Metrics")
    
    # Create metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        psnr = compression_data.get('psnr', 0)
        psnr_color = "#10b981" if psnr > 30 else "#f59e0b" if psnr > 20 else "#ef4444"
        st.markdown(
            f"""
            <div style="
                background: white;
                border: 2px solid {psnr_color};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 2rem; font-weight: bold; color: {psnr_color};">
                    {psnr:.1f}
                </div>
                <div style="font-size: 0.9rem; color: #6b7280; margin-top: 4px;">
                    PSNR (dB)
                </div>
                <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 2px;">
                    Signal Quality
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        ssim = compression_data.get('ssim', 0)
        ssim_color = "#10b981" if ssim > 0.8 else "#f59e0b" if ssim > 0.6 else "#ef4444"
        st.markdown(
            f"""
            <div style="
                background: white;
                border: 2px solid {ssim_color};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 2rem; font-weight: bold; color: {ssim_color};">
                    {ssim:.3f}
                </div>
                <div style="font-size: 0.9rem; color: #6b7280; margin-top: 4px;">
                    SSIM
                </div>
                <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 2px;">
                    Structural Similarity
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        ratio = compression_data.get('compression_ratio', 1)
        ratio_color = "#10b981" if ratio > 5 else "#f59e0b" if ratio > 2 else "#ef4444"
        st.markdown(
            f"""
            <div style="
                background: white;
                border: 2px solid {ratio_color};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 2rem; font-weight: bold; color: {ratio_color};">
                    {ratio:.1f}:1
                </div>
                <div style="font-size: 0.9rem; color: #6b7280; margin-top: 4px;">
                    Compression
                </div>
                <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 2px;">
                    Space Savings
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        k_value = compression_data.get('k_value', 0)
        quality_score = _calculate_composite_quality_score(psnr, ssim)
        quality_color = "#10b981" if quality_score > 80 else "#f59e0b" if quality_score > 60 else "#ef4444"
        st.markdown(
            f"""
            <div style="
                background: white;
                border: 2px solid {quality_color};
                border-radius: 12px;
                padding: 16px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 2rem; font-weight: bold; color: {quality_color};">
                    {quality_score:.0f}
                </div>
                <div style="font-size: 0.9rem; color: #6b7280; margin-top: 4px;">
                    Quality Score
                </div>
                <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 2px;">
                    k = {k_value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def _get_preset_recommendations(preset_label: str) -> str:
    """Get recommendations for each preset."""
    recommendations = {
        "🔴 Ultra Low": "• Thumbnails and previews\n• Maximum space savings\n• Quick loading images",
        "🟠 Low": "• Web images with size constraints\n• Email attachments\n• Basic quality requirements",
        "🟡 Medium": "• General web use\n• Social media images\n• Balanced quality/size",
        "🟢 High": "• Professional presentations\n• Print materials\n• High-quality web content",
        "🔵 Ultra High": "• Archival storage\n• Professional photography\n• Critical quality applications"
    }
    return recommendations.get(preset_label, "General use")


def _calculate_composite_quality_score(psnr: float, ssim: float) -> float:
    """Calculate composite quality score from PSNR and SSIM."""
    
    # Normalize PSNR to 0-100 scale (assuming max useful PSNR is 50)
    psnr_normalized = min(psnr / 50 * 100, 100)
    # SSIM is already 0-1, convert to 0-100
    ssim_normalized = ssim * 100
    # Weighted average (SSIM is more perceptually relevant)
    return (psnr_normalized * 0.4 + ssim_normalized * 0.6)


def _analyze_image_and_recommend_k(image: np.ndarray) -> int:
    """Analyze image content and recommend optimal k value."""
    
    try:
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gray_image = image
        
        # Use smaller image for faster analysis
        if gray_image.shape[0] > 256:
            step = gray_image.shape[0] // 256
            analysis_img = gray_image[::step, ::step]
        else:
            analysis_img = gray_image
        
        # Analyze image characteristics
        
        # 1. Calculate image complexity (edge density)
        from scipy import ndimage
        edges = ndimage.sobel(analysis_img)
        edge_density = np.mean(np.abs(edges))
        
        # 2. Calculate texture complexity (local variance)
        kernel = np.ones((3, 3)) / 9
        local_mean = ndimage.convolve(analysis_img, kernel)
        local_variance = ndimage.convolve((analysis_img - local_mean)**2, kernel)
        texture_complexity = np.mean(local_variance)
        
        # 3. Calculate dynamic range
        dynamic_range = np.std(analysis_img)
        
        # 4. SVD analysis for energy distribution
        U, s, Vt = np.linalg.svd(analysis_img, full_matrices=False)
        
        # Calculate energy concentration (how quickly energy drops off)
        total_energy = np.sum(s**2)
        cumulative_energy = np.cumsum(s**2) / total_energy
        
        # Find where 90% and 95% energy is reached
        k_90 = np.argmax(cumulative_energy >= 0.9) + 1
        k_95 = np.argmax(cumulative_energy >= 0.95) + 1
        
        # Recommendation logic based on image characteristics
        if edge_density > 0.3 and texture_complexity > 0.1:
            # High detail image - needs higher k
            recommended_k = max(k_95, 40)
            reason = "high detail/texture"
        elif edge_density < 0.1 and texture_complexity < 0.05:
            # Smooth image - can use lower k
            recommended_k = max(k_90, 15)
            reason = "smooth/low detail"
        elif dynamic_range < 0.2:
            # Low contrast image - can use lower k
            recommended_k = max(k_90, 20)
            reason = "low contrast"
        else:
            # Balanced image - use energy-based recommendation
            recommended_k = k_95
            reason = "balanced content"
        
        # Scale recommendation based on original image size
        if image.shape[0] > 512:
            recommended_k = min(recommended_k * 1.2, min(256, image.shape[0]))
        
        # Store analysis results for display
        st.session_state.image_analysis = {
            'edge_density': edge_density,
            'texture_complexity': texture_complexity,
            'dynamic_range': dynamic_range,
            'k_90': k_90,
            'k_95': k_95,
            'recommended_k': int(recommended_k),
            'reason': reason
        }
        
        return int(recommended_k)
        
    except ImportError:
        # Fallback if scipy is not available
        return _calculate_energy_based_k(image, energy_threshold=0.92)
    except Exception:
        # Fallback to safe default
        return 25


def _generate_quick_preview(image: np.ndarray, k_value: int, mode: str) -> np.ndarray:
    """Generate a quick preview of compressed image for real-time feedback."""
    
    try:
        # Use smaller image for faster preview
        preview_size = 128
        if image.shape[0] > preview_size:
            step = image.shape[0] // preview_size
            preview_img = image[::step, ::step]
        else:
            preview_img = image.copy()
        
        # Simple SVD compression for preview
        if mode == "Grayscale":
            if len(preview_img.shape) == 3:
                gray_img = np.dot(preview_img[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray_img = preview_img
            
            # SVD compression
            U, s, Vt = np.linalg.svd(gray_img, full_matrices=False)
            k_use = min(k_value, len(s))
            compressed = U[:, :k_use] @ np.diag(s[:k_use]) @ Vt[:k_use, :]
            
            # Convert back to RGB for display
            compressed = np.clip(compressed, 0, 1)
            if len(image.shape) == 3:
                compressed = np.stack([compressed] * 3, axis=-1)
        else:
            # RGB mode - compress each channel
            if len(preview_img.shape) == 3:
                compressed = np.zeros_like(preview_img)
                for c in range(preview_img.shape[2]):
                    U, s, Vt = np.linalg.svd(preview_img[:, :, c], full_matrices=False)
                    k_use = min(k_value, len(s))
                    compressed[:, :, c] = U[:, :k_use] @ np.diag(s[:k_use]) @ Vt[:k_use, :]
            else:
                U, s, Vt = np.linalg.svd(preview_img, full_matrices=False)
                k_use = min(k_value, len(s))
                compressed = U[:, :k_use] @ np.diag(s[:k_use]) @ Vt[:k_use, :]
        
        return np.clip(compressed, 0, 1)
        
    except Exception as e:
        # Return original image if preview fails
        return image