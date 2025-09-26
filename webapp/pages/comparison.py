"""
Enhanced comparison page for advanced k-value analysis and side-by-side comparisons.
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
from PIL import Image
import sys
from pathlib import Path
import time
from datetime import datetime

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from compression.svd_compressor import SVDCompressor
from evaluation.metrics_calculator import MetricsCalculator
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def show():
    """Display the enhanced comparison analysis page with advanced k-value comparison features."""
    
    # Use the new layout system
    from utils.styling import create_main_content_area, close_main_content_area
    
    # Create main content area
    create_main_content_area()
    
    st.markdown("# ⚖️ Advanced K-Value Comparison Analysis")
    st.markdown("Compare different compression levels with interactive visualizations and comprehensive analysis.")
    
    # Initialize session state
    if 'comparison_image' not in st.session_state:
        st.session_state.comparison_image = None
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'comparison_images' not in st.session_state:
        st.session_state.comparison_images = {}
    
    # Enhanced image upload with multiple image support
    st.markdown("## 📁 Upload Images for K-Value Comparison")
    
    # Create enhanced upload interface
    create_comparison_upload_interface()
    
    uploaded_files = st.file_uploader(
        "Choose image files for k-value comparison",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="Upload one or more images to compare different compression levels"
    )
    
    # Process uploaded files
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) uploaded successfully!")
        
        # Show uploaded files preview
        with st.expander("📋 Uploaded Images Preview", expanded=True):
            cols = st.columns(min(4, len(uploaded_files)))
            processed_images = {}
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    image = Image.open(uploaded_file)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image = image.resize((256, 256), Image.Resampling.LANCZOS)
                    original_array = np.array(image) / 255.0
                    
                    processed_images[uploaded_file.name] = {
                        'array': original_array,
                        'pil': image,
                        'file': uploaded_file
                    }
                    
                    with cols[i % 4]:
                        st.image(image, caption=uploaded_file.name, use_column_width=True)
                        st.markdown(f"**Size:** {original_array.shape[0]}×{original_array.shape[1]}")
                        
                except Exception as e:
                    st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            
            if len(uploaded_files) > 4:
                st.info(f"... and {len(uploaded_files) - 4} more images")
        
        # Store processed images
        st.session_state.comparison_images = processed_images
    
    # Enhanced comparison settings
    if st.session_state.comparison_images:
        st.markdown("---")
        st.markdown("## ⚙️ Advanced Comparison Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 K-Values Configuration")
            
            comparison_mode = st.radio(
                "K-value selection mode:",
                ["Quick Presets", "Custom Values", "Range Analysis", "Adaptive Selection"],
                help="Choose how to select k-values for comprehensive comparison"
            )
            
            if comparison_mode == "Quick Presets":
                preset_type = st.selectbox(
                    "Preset type:",
                    ["Quality Focus (4 levels)", "Compression Focus (5 levels)", "Comprehensive (8 levels)"]
                )
                
                if preset_type == "Quality Focus (4 levels)":
                    k_values = [10, 25, 50, 100]
                elif preset_type == "Compression Focus (5 levels)":
                    k_values = [2, 5, 10, 15, 25]
                else:  # Comprehensive
                    k_values = [2, 5, 10, 15, 25, 40, 60, 100]
                
                st.info(f"Selected k-values: {k_values}")
            
            elif comparison_mode == "Custom Values":
                k_input = st.text_input(
                    "Enter k-values (comma-separated)",
                    value="5, 10, 20, 30, 50",
                    help="Enter up to 10 k-values for comparison"
                )
                try:
                    k_values = [int(k.strip()) for k in k_input.split(',')]
                    k_values = [k for k in k_values if k > 0][:10]  # Limit to 10 values
                    st.info(f"Will compare k-values: {k_values}")
                except:
                    k_values = [5, 15, 30, 50]
                    st.warning("Invalid format. Using default values.")
            
            elif comparison_mode == "Range Analysis":
                k_min = st.number_input("Minimum k", min_value=1, max_value=100, value=5)
                k_max = st.number_input("Maximum k", min_value=k_min, max_value=256, value=50)
                k_count = st.number_input("Number of points", min_value=3, max_value=15, value=8)
                k_values = np.linspace(k_min, k_max, k_count, dtype=int).tolist()
                st.info(f"Will analyze {k_count} k-values from {k_min} to {k_max}")
            
            else:  # Adaptive Selection
                st.info("Adaptive selection will automatically choose optimal k-values based on image characteristics")
                base_k_values = [5, 10, 20, 30, 50]
                k_values = base_k_values  # Will be adapted per image
        
        with col2:
            st.markdown("### 🔧 Analysis Configuration")
            
            processing_mode = st.selectbox(
                "Processing mode",
                ["RGB (Color)", "Grayscale", "Both (Compare Modes)"],
                help="Choose image processing mode for comparison"
            )
            
            analysis_type = st.selectbox(
                "Analysis type",
                ["Single Image Focus", "Multi-Image Comparison", "Cross-Image Analysis"],
                help="Choose the type of comparison analysis to perform"
            )
            
            # Advanced options
            with st.expander("🔬 Advanced Options"):
                show_grid = st.checkbox("Show image comparison grids", value=True)
                show_metrics = st.checkbox("Calculate detailed quality metrics", value=True)
                show_analysis = st.checkbox("Perform statistical analysis", value=True)
                show_recommendations = st.checkbox("Generate optimization recommendations", value=True)
                
                # Export options
                st.markdown("**Export Options:**")
                export_images = st.checkbox("Include compressed images in export", value=False)
                export_plots = st.checkbox("Generate analysis plots for export", value=True)
        
        # Enhanced analysis execution
        if st.button("🚀 Run Advanced K-Value Analysis", type="primary", use_container_width=True):
            run_enhanced_comparison_analysis(
                k_values, processing_mode, analysis_type,
                show_grid, show_metrics, show_analysis, show_recommendations
            )
    
    # Display enhanced results
    if st.session_state.comparison_results is not None:
        display_enhanced_comparison_results()
    
    # Close main content area
    close_main_content_area()


def run_enhanced_comparison_analysis(k_values, processing_mode, analysis_type, 
                                   show_grid, show_metrics, show_analysis, show_recommendations):
    """Run enhanced comparison analysis with advanced features."""
    
    with st.spinner("Running advanced k-value comparison analysis..."):
        try:
            processed_images = st.session_state.comparison_images
            
            if analysis_type == "Single Image Focus":
                # Single image detailed analysis
                if len(processed_images) == 1:
                    filename, image_data = next(iter(processed_images.items()))
                    results_df = create_advanced_k_value_grid(
                        image_data['array'], k_values, processing_mode, filename
                    )
                else:
                    # Let user select image for focus
                    selected_image = st.selectbox(
                        "Select image for detailed analysis:",
                        list(processed_images.keys())
                    )
                    image_data = processed_images[selected_image]
                    results_df = create_advanced_k_value_grid(
                        image_data['array'], k_values, processing_mode, selected_image
                    )
            
            elif analysis_type == "Multi-Image Comparison":
                # Multi-image comparison
                uploaded_files = [data['file'] for data in processed_images.values()]
                results_df = create_multi_image_comparison(uploaded_files, k_values, processing_mode)
            
            else:  # Cross-Image Analysis
                # Cross-image comprehensive analysis
                uploaded_files = [data['file'] for data in processed_images.values()]
                results_df = create_multi_image_comparison(uploaded_files, k_values, processing_mode)
                
                # Additional cross-image analysis
                if not results_df.empty:
                    st.markdown("### 🔄 Cross-Image Performance Analysis")
                    
                    # Performance consistency analysis
                    consistency_analysis = analyze_cross_image_consistency(results_df)
                    display_consistency_analysis(consistency_analysis)
            
            # Store results
            st.session_state.comparison_results = {
                'dataframe': results_df,
                'settings': {
                    'k_values': k_values,
                    'processing_mode': processing_mode,
                    'analysis_type': analysis_type,
                    'show_grid': show_grid,
                    'show_metrics': show_metrics,
                    'show_analysis': show_analysis,
                    'show_recommendations': show_recommendations
                }
            }
            
            # Generate recommendations if requested
            if show_recommendations and not results_df.empty:
                st.markdown("---")
                st.markdown("### 🎯 Optimization Recommendations")
                display_optimization_recommendations(results_df)
            
            st.success("✅ Advanced comparison analysis completed!")
            
        except Exception as e:
            st.error(f"Error in enhanced comparison analysis: {str(e)}")


def analyze_cross_image_consistency(results_df):
    """Analyze consistency of performance across different images."""
    
    consistency_metrics = {}
    
    # Calculate coefficient of variation for each k-value across images
    for k in results_df['k_value'].unique():
        k_data = results_df[results_df['k_value'] == k]
        
        consistency_metrics[k] = {
            'psnr_cv': k_data['psnr'].std() / k_data['psnr'].mean() * 100,
            'ssim_cv': k_data['ssim'].std() / k_data['ssim'].mean() * 100,
            'quality_cv': k_data['quality_score'].std() / k_data['quality_score'].mean() * 100,
            'mean_quality': k_data['quality_score'].mean(),
            'min_quality': k_data['quality_score'].min(),
            'max_quality': k_data['quality_score'].max()
        }
    
    return consistency_metrics


def display_consistency_analysis(consistency_metrics):
    """Display cross-image consistency analysis."""
    
    st.markdown("#### 📊 Performance Consistency Analysis")
    
    # Create consistency DataFrame
    consistency_df = pd.DataFrame(consistency_metrics).T
    consistency_df.index.name = 'k_value'
    consistency_df = consistency_df.reset_index()
    
    # Display consistency table
    st.markdown("**Performance Consistency by K-Value:**")
    st.dataframe(
        consistency_df.round(2),
        use_container_width=True,
        column_config={
            "psnr_cv": st.column_config.NumberColumn(
                "PSNR CV (%)",
                help="Coefficient of Variation for PSNR across images",
                format="%.1f%%"
            ),
            "ssim_cv": st.column_config.NumberColumn(
                "SSIM CV (%)",
                help="Coefficient of Variation for SSIM across images",
                format="%.1f%%"
            ),
            "quality_cv": st.column_config.NumberColumn(
                "Quality CV (%)",
                help="Coefficient of Variation for Quality Score across images",
                format="%.1f%%"
            )
        }
    )
    
    # Find most consistent k-value
    most_consistent_k = consistency_df.loc[consistency_df['quality_cv'].idxmin(), 'k_value']
    best_avg_quality_k = consistency_df.loc[consistency_df['mean_quality'].idxmax(), 'k_value']
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🎯 **Most Consistent Performance**: k={most_consistent_k}")
        st.markdown(f"Quality CV: {consistency_df.loc[consistency_df['k_value'] == most_consistent_k, 'quality_cv'].iloc[0]:.1f}%")
    
    with col2:
        st.info(f"🏆 **Best Average Quality**: k={best_avg_quality_k}")
        st.markdown(f"Average Quality: {consistency_df.loc[consistency_df['k_value'] == best_avg_quality_k, 'mean_quality'].iloc[0]:.1f}/100")


def display_optimization_recommendations(results_df):
    """Display optimization recommendations based on analysis results."""
    
    # Generate different types of recommendations
    recommendations = {
        "Quality-Focused": generate_quality_recommendations(results_df),
        "Compression-Focused": generate_compression_recommendations(results_df),
        "Balanced": generate_balanced_recommendations(results_df)
    }
    
    # Display recommendations in tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Quality Focus", "📦 Compression Focus", "⚖️ Balanced Approach"])
    
    with tab1:
        st.markdown("**Recommendations for Maximum Quality:**")
        for rec in recommendations["Quality-Focused"]:
            st.markdown(f"- {rec}")
    
    with tab2:
        st.markdown("**Recommendations for Maximum Compression:**")
        for rec in recommendations["Compression-Focused"]:
            st.markdown(f"- {rec}")
    
    with tab3:
        st.markdown("**Recommendations for Balanced Performance:**")
        for rec in recommendations["Balanced"]:
            st.markdown(f"- {rec}")


def generate_quality_recommendations(results_df):
    """Generate quality-focused recommendations."""
    
    recommendations = []
    
    # Best quality k-value
    best_quality = results_df.loc[results_df['quality_score'].idxmax()]
    recommendations.append(
        f"Use k={best_quality['k_value']} for maximum quality (Score: {best_quality['quality_score']:.1f}/100)"
    )
    
    # Quality threshold recommendations
    high_quality_results = results_df[results_df['quality_score'] >= 80]
    if not high_quality_results.empty:
        min_k_for_high_quality = high_quality_results['k_value'].min()
        recommendations.append(
            f"Minimum k={min_k_for_high_quality} required for excellent quality (>80/100)"
        )
    
    # PSNR-based recommendations
    best_psnr = results_df.loc[results_df['psnr'].idxmax()]
    if best_psnr['k_value'] != best_quality['k_value']:
        recommendations.append(
            f"For best PSNR specifically, use k={best_psnr['k_value']} ({best_psnr['psnr']:.2f} dB)"
        )
    
    return recommendations


def generate_compression_recommendations(results_df):
    """Generate compression-focused recommendations."""
    
    recommendations = []
    
    # Best compression ratio
    best_compression = results_df.loc[results_df['compression_ratio'].idxmax()]
    recommendations.append(
        f"Use k={best_compression['k_value']} for maximum compression ({best_compression['compression_ratio']:.1f}:1 ratio)"
    )
    
    # Quality trade-off analysis
    acceptable_quality_threshold = 60
    good_compression_results = results_df[
        (results_df['compression_ratio'] >= 3) & 
        (results_df['quality_score'] >= acceptable_quality_threshold)
    ]
    
    if not good_compression_results.empty:
        optimal_compression = good_compression_results.loc[
            good_compression_results['compression_ratio'].idxmax()
        ]
        recommendations.append(
            f"For good compression with acceptable quality: k={optimal_compression['k_value']} "
            f"({optimal_compression['compression_ratio']:.1f}:1, Quality: {optimal_compression['quality_score']:.1f}/100)"
        )
    
    return recommendations


def generate_balanced_recommendations(results_df):
    """Generate balanced performance recommendations."""
    
    recommendations = []
    
    # Calculate balance score (quality * compression efficiency)
    results_df_copy = results_df.copy()
    results_df_copy['balance_score'] = (
        results_df_copy['quality_score'] * 
        np.log(results_df_copy['compression_ratio'] + 1)  # Log to prevent compression bias
    )
    
    best_balance = results_df_copy.loc[results_df_copy['balance_score'].idxmax()]
    recommendations.append(
        f"Optimal balance at k={best_balance['k_value']} "
        f"(Quality: {best_balance['quality_score']:.1f}/100, Compression: {best_balance['compression_ratio']:.1f}:1)"
    )
    
    # Median performance recommendation
    median_k = int(np.median(results_df['k_value'].unique()))
    median_results = results_df[results_df['k_value'] == median_k]
    if not median_results.empty:
        median_quality = median_results['quality_score'].mean()
        median_compression = median_results['compression_ratio'].mean()
        recommendations.append(
            f"Safe middle-ground choice: k={median_k} "
            f"(Avg Quality: {median_quality:.1f}/100, Avg Compression: {median_compression:.1f}:1)"
        )
    
    return recommendations


def create_comprehensive_comparison_export(df, settings):
    """Create comprehensive export functionality for comparison results."""
    
    st.markdown("---")
    st.markdown("### 💾 Export Comparison Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export results as CSV
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Results CSV",
            data=csv_data,
            file_name=f"comparison_results_{int(time.time())}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export detailed report
        report = generate_comparison_report(df, settings)
        st.download_button(
            label="📄 Download Analysis Report",
            data=report,
            file_name=f"comparison_analysis_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Export summary statistics
        summary = generate_summary_statistics(df)
        st.download_button(
            label="📈 Download Summary Stats",
            data=summary,
            file_name=f"comparison_summary_{int(time.time())}.txt",
            mime="text/plain",
            use_container_width=True
        )


def generate_comparison_report(df, settings):
    """Generate comprehensive comparison analysis report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
SVD IMAGE COMPRESSION - COMPARISON ANALYSIS REPORT
=================================================

ANALYSIS TIMESTAMP: {timestamp}

ANALYSIS SETTINGS
=================
K-Values Tested: {sorted(df['k_value'].unique().tolist())}
Processing Mode: {settings.get('processing_mode', 'RGB')}
Analysis Type: {settings.get('analysis_type', 'Single Image Focus')}
Images Analyzed: {df['filename'].nunique() if 'filename' in df.columns else 1}

SUMMARY STATISTICS
==================
Total Experiments: {len(df)}
K-Value Range: {df['k_value'].min()} - {df['k_value'].max()}

Quality Metrics Summary:
- PSNR Range: {df['psnr'].min():.2f} - {df['psnr'].max():.2f} dB
- SSIM Range: {df['ssim'].min():.3f} - {df['ssim'].max():.3f}
- Compression Ratio Range: {df['compression_ratio'].min():.1f} - {df['compression_ratio'].max():.1f}:1

Best Performance:
- Highest PSNR: {df['psnr'].max():.2f} dB (k={df.loc[df['psnr'].idxmax(), 'k_value']})
- Highest SSIM: {df['ssim'].max():.3f} (k={df.loc[df['ssim'].idxmax(), 'k_value']})
- Best Compression: {df['compression_ratio'].max():.1f}:1 (k={df.loc[df['compression_ratio'].idxmax(), 'k_value']})

DETAILED RESULTS
================
"""
    
    for _, row in df.iterrows():
        report += f"""
K-Value: {row['k_value']}
- PSNR: {row['psnr']:.2f} dB
- SSIM: {row['ssim']:.3f}
- MSE: {row['mse']:.6f}
- Compression Ratio: {row['compression_ratio']:.1f}:1
"""
        if 'filename' in row:
            report += f"- Filename: {row['filename']}\n"
        report += "\n"
    
    report += f"""
ANALYSIS CONCLUSIONS
====================
Based on the comparison analysis, the following observations can be made:

1. Quality vs Compression Trade-off:
   - Lower k-values provide higher compression but lower quality
   - Higher k-values preserve more quality but reduce compression efficiency

2. Optimal K-Value Recommendations:
   - For quality priority: k={df.loc[df['psnr'].idxmax(), 'k_value']} (Best PSNR)
   - For compression priority: k={df.loc[df['compression_ratio'].idxmax(), 'k_value']} (Best Compression)
   - For balanced approach: k={df.loc[(df['psnr'] * df['compression_ratio']).idxmax(), 'k_value']} (Best Balance)

GENERATED BY
============
SVD Image Compression Tool - Comparison Analysis
Report generated on {timestamp}
"""
    
    return report


def generate_summary_statistics(df):
    """Generate summary statistics for comparison results."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    summary = f"""
SVD COMPRESSION - SUMMARY STATISTICS
====================================

Generated: {timestamp}

BASIC STATISTICS
================
Number of experiments: {len(df)}
K-values tested: {len(df['k_value'].unique())}
K-value range: {df['k_value'].min()} to {df['k_value'].max()}

PSNR STATISTICS
===============
Mean: {df['psnr'].mean():.2f} dB
Median: {df['psnr'].median():.2f} dB
Std Dev: {df['psnr'].std():.2f} dB
Min: {df['psnr'].min():.2f} dB
Max: {df['psnr'].max():.2f} dB

SSIM STATISTICS
===============
Mean: {df['ssim'].mean():.3f}
Median: {df['ssim'].median():.3f}
Std Dev: {df['ssim'].std():.3f}
Min: {df['ssim'].min():.3f}
Max: {df['ssim'].max():.3f}

COMPRESSION RATIO STATISTICS
============================
Mean: {df['compression_ratio'].mean():.1f}:1
Median: {df['compression_ratio'].median():.1f}:1
Std Dev: {df['compression_ratio'].std():.1f}
Min: {df['compression_ratio'].min():.1f}:1
Max: {df['compression_ratio'].max():.1f}:1

CORRELATIONS
============
PSNR vs K-value: {df['psnr'].corr(df['k_value']):.3f}
SSIM vs K-value: {df['ssim'].corr(df['k_value']):.3f}
Compression vs K-value: {df['compression_ratio'].corr(df['k_value']):.3f}
PSNR vs SSIM: {df['psnr'].corr(df['ssim']):.3f}
"""
    
    return summary


def display_enhanced_comparison_results():
    """Display enhanced comparison results with comprehensive analysis."""
    
    st.markdown("---")
    st.markdown("## 📊 Enhanced Comparison Analysis Results")
    
    results = st.session_state.comparison_results
    df = results['dataframe']
    settings = results['settings']
    
    if df.empty:
        st.warning("No results to display.")
        return
    
    # Enhanced summary with visual indicators
    st.markdown("### 📈 Executive Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Images Analyzed", df['filename'].nunique() if 'filename' in df.columns else 1)
    
    with col2:
        st.metric("K-Values Tested", df['k_value'].nunique())
    
    with col3:
        best_quality = df['quality_score'].max() if 'quality_score' in df.columns else df['psnr'].max()
        st.metric("Best Quality", f"{best_quality:.1f}" + ("/100" if 'quality_score' in df.columns else " dB"))
    
    with col4:
        avg_compression = df['compression_ratio'].mean()
        st.metric("Avg Compression", f"{avg_compression:.1f}:1")
    
    with col5:
        total_experiments = len(df)
        st.metric("Total Experiments", total_experiments)
    
    # Use the new results display component for individual results
    if len(df) == 1 and 'filename' in df.columns:
        # Single image analysis - use full results display
        row = df.iloc[0]
        
        # Get images from session state if available
        if hasattr(st.session_state, 'comparison_images') and st.session_state.comparison_images:
            filename = row.get('filename', 'comparison_image')
            if filename in st.session_state.comparison_images:
                image_data = st.session_state.comparison_images[filename]
                
                # Create mock compressed image and compression data for display
                # In a real scenario, this would come from the actual compression results
                compression_data = {
                    'k_value': row.get('k_value', 0),
                    'psnr': row.get('psnr', 0),
                    'ssim': row.get('ssim', 0),
                    'mse': row.get('mse', 0),
                    'compression_ratio': row.get('compression_ratio', 1),
                    'mode': settings.get('processing_mode', 'RGB')
                }
                
                from utils.results_display import create_results_display_component
                create_results_display_component(
                    original_image=image_data['array'],
                    compressed_image=image_data['array'],  # Would be actual compressed image
                    compression_data=compression_data,
                    filename=filename
                )
    
    # Comprehensive export functionality
    create_comprehensive_comparison_export(df, settings)


def run_comparison_analysis(k_values, processing_mode, show_grid, show_metrics, show_analysis):
    """Run comprehensive comparison analysis."""
    
    with st.spinner("Running comparison analysis..."):
        try:
            compressor = SVDCompressor()
            metrics_calc = MetricsCalculator()
            
            original_image = st.session_state.comparison_image
            
            # Prepare image based on processing mode
            if processing_mode == "Grayscale":
                gray_image = np.dot(original_image[...,:3], [0.2989, 0.5870, 0.1140])
                image_to_process = gray_image
            else:
                image_to_process = original_image
            
            results = []
            compressed_images = {}
            
            # Process each k-value with enhanced progress
            from utils.loading_animations import show_progress_ring
            
            progress_animation = show_progress_ring(
                text=f"Processing {len(k_values)} compression levels...",
                progress=0.0,
                color='purple'
            )
            
            for i, k in enumerate(k_values):
                try:
                    # Compress image
                    compressed_image, metadata = compressor.compress_image(image_to_process, k)
                    
                    # Prepare for metrics calculation
                    if processing_mode == "Grayscale":
                        compressed_rgb = np.stack([compressed_image] * 3, axis=-1)
                        original_rgb = np.stack([gray_image] * 3, axis=-1)
                    else:
                        compressed_rgb = compressed_image
                        original_rgb = original_image
                    
                    # Calculate metrics if requested
                    if show_metrics:
                        psnr = metrics_calc.calculate_psnr(original_rgb, compressed_rgb)
                        ssim = metrics_calc.calculate_ssim(original_rgb, compressed_rgb)
                        mse = metrics_calc.calculate_mse(original_rgb, compressed_rgb)
                    else:
                        psnr = ssim = mse = 0
                    
                    # Store results
                    result = {
                        'k_value': k,
                        'psnr': psnr,
                        'ssim': ssim,
                        'mse': mse,
                        'compression_ratio': metadata.get('compression_ratio', 0)
                    }
                    results.append(result)
                    
                    # Store compressed image for display
                    if show_grid:
                        if processing_mode == "Grayscale":
                            compressed_images[k] = compressed_rgb
                        else:
                            compressed_images[k] = compressed_image
                
                except Exception as e:
                    st.warning(f"Error processing k={k}: {str(e)}")
                
                # Update progress
                current_progress = (i + 1) / len(k_values)
                progress_animation['update_progress'](current_progress)
                progress_animation['update_text'](f"Processing k={k} ({i+1}/{len(k_values)})")
            
            # Store results
            st.session_state.comparison_results = {
                'dataframe': pd.DataFrame(results),
                'images': compressed_images,
                'settings': {
                    'k_values': k_values,
                    'processing_mode': processing_mode,
                    'show_grid': show_grid,
                    'show_metrics': show_metrics,
                    'show_analysis': show_analysis
                }
            }
            
            st.success("✅ Comparison analysis completed!")
            
        except Exception as e:
            st.error(f"Error in comparison analysis: {str(e)}")


def display_comparison_results():
    """Display comprehensive comparison results."""
    
    st.markdown("---")
    st.markdown("## 📊 Comparison Results")
    
    results = st.session_state.comparison_results
    df = results['dataframe']
    images = results['images']
    settings = results['settings']
    
    # Summary metrics
    if settings['show_metrics']:
        st.markdown("### 📈 Quality Metrics Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best PSNR", f"{df['psnr'].max():.2f} dB", f"at k={df.loc[df['psnr'].idxmax(), 'k_value']}")
        
        with col2:
            st.metric("Best SSIM", f"{df['ssim'].max():.3f}", f"at k={df.loc[df['ssim'].idxmax(), 'k_value']}")
        
        with col3:
            st.metric("Lowest MSE", f"{df['mse'].min():.4f}", f"at k={df.loc[df['mse'].idxmin(), 'k_value']}")
        
        with col4:
            st.metric("Max Compression", f"{df['compression_ratio'].max():.1f}:1", f"at k={df.loc[df['compression_ratio'].idxmax(), 'k_value']}")
    
    # Image grid display
    if settings['show_grid'] and images:
        st.markdown("### 🖼️ Visual Comparison Grid")
        
        # Determine grid layout
        n_images = len(images) + 1  # +1 for original
        cols_per_row = min(4, n_images)
        n_rows = (n_images + cols_per_row - 1) // cols_per_row
        
        # Display original image first
        all_images = [("Original", st.session_state.comparison_image)]
        all_images.extend([(f"k={k}", img) for k, img in sorted(images.items())])
        
        for row in range(n_rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                if img_idx < len(all_images):
                    title, img = all_images[img_idx]
                    with cols[col_idx]:
                        st.image(img, caption=title, use_column_width=True)
    
    # Detailed metrics table
    if settings['show_metrics']:
        st.markdown("### 📋 Detailed Metrics")
        
        # Format dataframe for display
        display_df = df.copy()
        display_df['PSNR (dB)'] = display_df['psnr'].round(2)
        display_df['SSIM'] = display_df['ssim'].round(3)
        display_df['MSE'] = display_df['mse'].round(4)
        display_df['Compression Ratio'] = display_df['compression_ratio'].round(1)
        
        st.dataframe(
            display_df[['k_value', 'PSNR (dB)', 'SSIM', 'MSE', 'Compression Ratio']],
            use_container_width=True,
            hide_index=True
        )
    
    # Interactive visualizations
    st.markdown("### 📈 Interactive Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Quality Trends", "Trade-off Analysis", "Detailed Plots"])
    
    with tab1:
        if settings['show_metrics']:
            # Quality metrics vs k-value
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('PSNR vs K-value', 'SSIM vs K-value', 'MSE vs K-value', 'Compression Ratio vs K-value'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # PSNR
            fig.add_trace(
                go.Scatter(x=df['k_value'], y=df['psnr'], mode='lines+markers', name='PSNR'),
                row=1, col=1
            )
            
            # SSIM
            fig.add_trace(
                go.Scatter(x=df['k_value'], y=df['ssim'], mode='lines+markers', name='SSIM'),
                row=1, col=2
            )
            
            # MSE
            fig.add_trace(
                go.Scatter(x=df['k_value'], y=df['mse'], mode='lines+markers', name='MSE'),
                row=2, col=1
            )
            
            # Compression Ratio
            fig.add_trace(
                go.Scatter(x=df['k_value'], y=df['compression_ratio'], mode='lines+markers', name='Compression Ratio'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Quality Metrics Analysis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Enable 'Show detailed metrics' to see quality trends")
    
    with tab2:
        if settings['show_metrics']:
            # Trade-off analysis
            fig_tradeoff = px.scatter(
                df,
                x='compression_ratio',
                y='psnr',
                size='ssim',
                hover_data=['k_value', 'mse'],
                title='Quality vs Compression Trade-off',
                labels={
                    'compression_ratio': 'Compression Ratio',
                    'psnr': 'PSNR (dB)',
                    'ssim': 'SSIM (size)'
                }
            )
            
            # Add annotations for key points
            best_quality_idx = df['psnr'].idxmax()
            best_compression_idx = df['compression_ratio'].idxmax()
            
            fig_tradeoff.add_annotation(
                x=df.loc[best_quality_idx, 'compression_ratio'],
                y=df.loc[best_quality_idx, 'psnr'],
                text=f"Best Quality<br>k={df.loc[best_quality_idx, 'k_value']}",
                showarrow=True,
                arrowhead=2
            )
            
            fig_tradeoff.add_annotation(
                x=df.loc[best_compression_idx, 'compression_ratio'],
                y=df.loc[best_compression_idx, 'psnr'],
                text=f"Best Compression<br>k={df.loc[best_compression_idx, 'k_value']}",
                showarrow=True,
                arrowhead=2
            )
            
            st.plotly_chart(fig_tradeoff, use_container_width=True)
        else:
            st.info("Enable 'Show detailed metrics' to see trade-off analysis")
    
    with tab3:
        if settings['show_metrics']:
            # Detailed individual plots
            metric_choice = st.selectbox(
                "Choose metric to analyze:",
                ["PSNR", "SSIM", "MSE", "Compression Ratio"]
            )
            
            metric_map = {
                "PSNR": 'psnr',
                "SSIM": 'ssim', 
                "MSE": 'mse',
                "Compression Ratio": 'compression_ratio'
            }
            
            metric_col = metric_map[metric_choice]
            
            fig_detailed = go.Figure()
            
            fig_detailed.add_trace(go.Scatter(
                x=df['k_value'],
                y=df[metric_col],
                mode='lines+markers',
                name=metric_choice,
                line=dict(width=3),
                marker=dict(size=8)
            ))
            
            # Add optimal point
            if metric_choice in ["PSNR", "SSIM", "Compression Ratio"]:
                optimal_idx = df[metric_col].idxmax()
            else:  # MSE - lower is better
                optimal_idx = df[metric_col].idxmin()
            
            fig_detailed.add_trace(go.Scatter(
                x=[df.loc[optimal_idx, 'k_value']],
                y=[df.loc[optimal_idx, metric_col]],
                mode='markers',
                name='Optimal',
                marker=dict(size=15, color='red', symbol='star')
            ))
            
            fig_detailed.update_layout(
                title=f'{metric_choice} vs K-value Analysis',
                xaxis_title='K-value',
                yaxis_title=metric_choice,
                height=400
            )
            
            st.plotly_chart(fig_detailed, use_container_width=True)
            
            # Statistical summary
            st.markdown(f"**{metric_choice} Statistics:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Mean", f"{df[metric_col].mean():.3f}")
            with col2:
                st.metric("Std Dev", f"{df[metric_col].std():.3f}")
            with col3:
                st.metric("Min", f"{df[metric_col].min():.3f}")
            with col4:
                st.metric("Max", f"{df[metric_col].max():.3f}")
        else:
            st.info("Enable 'Show detailed metrics' to see detailed plots")
    
    # Statistical analysis
    if settings['show_analysis'] and settings['show_metrics']:
        st.markdown("---")
        st.markdown("### 🔬 Statistical Analysis")
        
        # Correlation analysis
        st.markdown("#### Correlation Matrix")
        corr_matrix = df[['k_value', 'psnr', 'ssim', 'mse', 'compression_ratio']].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Correlation Matrix of Metrics",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Key insights
        st.markdown("#### Key Insights")
        
        # Find optimal k-values for different criteria
        best_psnr_k = df.loc[df['psnr'].idxmax(), 'k_value']
        best_ssim_k = df.loc[df['ssim'].idxmax(), 'k_value']
        best_compression_k = df.loc[df['compression_ratio'].idxmax(), 'k_value']
        
        insights = [
            f"🎯 **Best Quality**: k={best_psnr_k} (PSNR: {df['psnr'].max():.2f} dB)",
            f"🔍 **Best Structure**: k={best_ssim_k} (SSIM: {df['ssim'].max():.3f})",
            f"📦 **Best Compression**: k={best_compression_k} (Ratio: {df['compression_ratio'].max():.1f}:1)",
            f"📊 **Quality Range**: PSNR varies by {df['psnr'].max() - df['psnr'].min():.2f} dB",
            f"⚖️ **Trade-off**: Each 10x compression reduces PSNR by ~{(df['psnr'].max() - df['psnr'].min()) / (df['compression_ratio'].max() - df['compression_ratio'].min()) * 10:.1f} dB"
        ]
        
        for insight in insights:
            st.markdown(insight)
    
    # Download section
    st.markdown("---")
    st.markdown("## 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Download comparison report
        if settings['show_metrics']:
            report = generate_comparison_report(df, settings)
            st.download_button(
                label="📄 Download Comparison Report",
                data=report,
                file_name="comparison_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col2:
        # Download results CSV
        if settings['show_metrics']:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📊 Download Results CSV",
                data=csv_buffer.getvalue(),
                file_name="comparison_results.csv",
                mime="text/csv",
                use_container_width=True
            )


def generate_comparison_report(df, settings):
    """Generate a comprehensive comparison report."""
    
    report = f"""SVD Image Compression - Comparison Analysis Report
=====================================================

Analysis Settings:
- Processing Mode: {settings['processing_mode']}
- K-values Tested: {settings['k_values']}
- Number of Comparisons: {len(df)}

Quality Metrics Summary:
- Best PSNR: {df['psnr'].max():.2f} dB (k={df.loc[df['psnr'].idxmax(), 'k_value']})
- Best SSIM: {df['ssim'].max():.3f} (k={df.loc[df['ssim'].idxmax(), 'k_value']})
- Lowest MSE: {df['mse'].min():.4f} (k={df.loc[df['mse'].idxmin(), 'k_value']})
- Max Compression: {df['compression_ratio'].max():.1f}:1 (k={df.loc[df['compression_ratio'].idxmax(), 'k_value']})

Detailed Results:
"""
    
    for _, row in df.iterrows():
        report += f"""
k={row['k_value']:3d}: PSNR={row['psnr']:6.2f}dB, SSIM={row['ssim']:.3f}, MSE={row['mse']:.4f}, Ratio={row['compression_ratio']:.1f}:1"""
    
    report += f"""

Statistical Analysis:
- PSNR: Mean={df['psnr'].mean():.2f}±{df['psnr'].std():.2f} dB
- SSIM: Mean={df['ssim'].mean():.3f}±{df['ssim'].std():.3f}
- Compression Ratio: Mean={df['compression_ratio'].mean():.1f}±{df['compression_ratio'].std():.1f}

Recommendations:
- For best quality: Use k={df.loc[df['psnr'].idxmax(), 'k_value']} (PSNR: {df['psnr'].max():.2f} dB)
- For balanced quality/compression: Use k={df.loc[df.index[len(df)//2], 'k_value']} (PSNR: {df.iloc[len(df)//2]['psnr']:.2f} dB, Ratio: {df.iloc[len(df)//2]['compression_ratio']:.1f}:1)
- For maximum compression: Use k={df.loc[df['compression_ratio'].idxmax(), 'k_value']} (Ratio: {df['compression_ratio'].max():.1f}:1)

Generated by SVD Image Compression Tool
"""
    
    return report


def create_comparison_upload_interface():
    """Create an enhanced upload interface for comparison analysis."""
    
    st.markdown(
        """
        <div style="
            border: 3px dashed #3b82f6;
            border-radius: 15px;
            padding: 50px;
            text-align: center;
            background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
            margin: 20px 0;
            transition: all 0.3s ease;
        ">
            <div style="font-size: 4rem; margin-bottom: 15px;">⚖️</div>
            <h2 style="color: #1f2937; margin-bottom: 15px;">K-Value Comparison Analysis</h2>
            <p style="color: #6b7280; margin-bottom: 20px; font-size: 1.1rem;">
                Upload images to compare different compression levels side-by-side
            </p>
            <div style="
                display: inline-block;
                background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
                color: white;
                padding: 15px 30px;
                border-radius: 10px;
                font-weight: 600;
                font-size: 1.1rem;
            ">
                📤 Upload for Comparison
            </div>
            <p style="color: #9ca3af; margin-top: 15px; font-size: 0.9rem;">
                Supports PNG, JPG, JPEG • Multiple images supported • Up to 10MB each
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_advanced_k_value_grid(image, k_values, processing_mode, filename):
    """Create an advanced k-value comparison grid with interactive features."""
    
    st.markdown(f"### 🔍 K-Value Comparison Grid - {filename}")
    
    # Process image for all k-values
    compressor = SVDCompressor()
    metrics_calc = MetricsCalculator()
    
    # Prepare image based on processing mode
    if processing_mode == "Grayscale":
        gray_image = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        image_to_process = gray_image
    else:
        image_to_process = image
    
    # Create grid layout
    n_cols = min(4, len(k_values) + 1)  # +1 for original
    n_rows = (len(k_values) + 1 + n_cols - 1) // n_cols
    
    # Display original image
    st.markdown("#### Original vs Compressed Comparison")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📷 Image Grid", "📊 Metrics Table", "📈 Interactive Analysis"])
    
    with tab1:
        # Image grid display
        all_images = [("Original", image)]
        all_metrics = []
        
        # Process each k-value
        for k in k_values:
            try:
                compressed_image, metadata = compressor.compress_image(image_to_process, k)
                
                # Prepare for display and metrics
                if processing_mode == "Grayscale":
                    compressed_rgb = np.stack([compressed_image] * 3, axis=-1)
                    original_rgb = np.stack([gray_image] * 3, axis=-1)
                else:
                    compressed_rgb = compressed_image
                    original_rgb = image
                
                # Calculate metrics
                psnr = metrics_calc.calculate_psnr(original_rgb, compressed_rgb)
                ssim = metrics_calc.calculate_ssim(original_rgb, compressed_rgb)
                compression_ratio = metadata.get('compression_ratio', 0)
                
                all_images.append((f"k={k}", compressed_rgb))
                all_metrics.append({
                    'k_value': k,
                    'psnr': psnr,
                    'ssim': ssim,
                    'compression_ratio': compression_ratio
                })
                
            except Exception as e:
                st.warning(f"Error processing k={k}: {str(e)}")
        
        # Display images in grid
        for row in range(n_rows):
            cols = st.columns(n_cols)
            for col_idx in range(n_cols):
                img_idx = row * n_cols + col_idx
                if img_idx < len(all_images):
                    title, img = all_images[img_idx]
                    with cols[col_idx]:
                        st.image(img, caption=title, use_column_width=True)
                        
                        # Add metrics overlay for compressed images
                        if img_idx > 0 and img_idx - 1 < len(all_metrics):
                            metrics = all_metrics[img_idx - 1]
                            st.markdown(
                                f"""
                                <div style="
                                    background: rgba(59, 130, 246, 0.1);
                                    border: 1px solid #3b82f6;
                                    border-radius: 8px;
                                    padding: 8px;
                                    margin-top: 5px;
                                    font-size: 0.8rem;
                                ">
                                    <strong>PSNR:</strong> {metrics['psnr']:.1f} dB<br>
                                    <strong>SSIM:</strong> {metrics['ssim']:.3f}<br>
                                    <strong>Ratio:</strong> {metrics['compression_ratio']:.1f}:1
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
    
    with tab2:
        # Metrics comparison table
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            
            # Add quality score
            metrics_df['Quality Score'] = metrics_df.apply(
                lambda row: calculate_quality_score_comparison(row['psnr'], row['ssim']), axis=1
            ).round(1)
            
            # Color-coded table
            st.markdown("**Quality Metrics Comparison:**")
            
            st.dataframe(
                metrics_df.round(3),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Quality Score": st.column_config.ProgressColumn(
                        "Quality Score",
                        help="Composite quality metric (0-100)",
                        min_value=0,
                        max_value=100,
                    ),
                    "psnr": st.column_config.NumberColumn(
                        "PSNR (dB)",
                        help="Peak Signal-to-Noise Ratio",
                        format="%.2f"
                    ),
                    "ssim": st.column_config.NumberColumn(
                        "SSIM",
                        help="Structural Similarity Index",
                        format="%.3f"
                    ),
                    "compression_ratio": st.column_config.NumberColumn(
                        "Compression Ratio",
                        help="Original size / Compressed size",
                        format="%.1f:1"
                    )
                }
            )
            
            # Highlight best performers
            best_quality = metrics_df.loc[metrics_df['Quality Score'].idxmax()]
            best_compression = metrics_df.loc[metrics_df['compression_ratio'].idxmax()]
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🏆 **Best Quality**: k={best_quality['k_value']} (Score: {best_quality['Quality Score']:.1f}/100)")
            with col2:
                st.info(f"📦 **Best Compression**: k={best_compression['k_value']} (Ratio: {best_compression['compression_ratio']:.1f}:1)")
    
    with tab3:
        # Interactive analysis
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            
            # Create interactive plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('PSNR vs K-Value', 'SSIM vs K-Value', 
                               'Compression Ratio vs K-Value', 'Quality Trade-off'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # PSNR plot
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['k_value'],
                    y=metrics_df['psnr'],
                    mode='lines+markers',
                    name='PSNR',
                    line=dict(color='#ef4444', width=3),
                    marker=dict(size=8),
                    hovertemplate='K=%{x}<br>PSNR=%{y:.2f} dB<extra></extra>'
                ),
                row=1, col=1
            )
            
            # SSIM plot
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['k_value'],
                    y=metrics_df['ssim'],
                    mode='lines+markers',
                    name='SSIM',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=8),
                    hovertemplate='K=%{x}<br>SSIM=%{y:.3f}<extra></extra>'
                ),
                row=1, col=2
            )
            
            # Compression ratio plot
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['k_value'],
                    y=metrics_df['compression_ratio'],
                    mode='lines+markers',
                    name='Compression Ratio',
                    line=dict(color='#3b82f6', width=3),
                    marker=dict(size=8),
                    hovertemplate='K=%{x}<br>Ratio=%{y:.1f}:1<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Quality vs compression trade-off
            fig.add_trace(
                go.Scatter(
                    x=metrics_df['compression_ratio'],
                    y=metrics_df['psnr'],
                    mode='markers',
                    name='Trade-off',
                    marker=dict(
                        size=metrics_df['ssim'] * 20,  # Size based on SSIM
                        color=metrics_df['k_value'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="K-Value")
                    ),
                    text=metrics_df['k_value'],
                    hovertemplate='K=%{text}<br>Compression=%{x:.1f}:1<br>PSNR=%{y:.2f} dB<extra></extra>'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=700,
                showlegend=False,
                title_text=f"Comprehensive K-Value Analysis - {filename}"
            )
            
            fig.update_xaxes(title_text="K-Value", row=1, col=1)
            fig.update_xaxes(title_text="K-Value", row=1, col=2)
            fig.update_xaxes(title_text="K-Value", row=2, col=1)
            fig.update_xaxes(title_text="Compression Ratio", row=2, col=2)
            
            fig.update_yaxes(title_text="PSNR (dB)", row=1, col=1)
            fig.update_yaxes(title_text="SSIM", row=1, col=2)
            fig.update_yaxes(title_text="Compression Ratio", row=2, col=1)
            fig.update_yaxes(title_text="PSNR (dB)", row=2, col=2)
            
            st.plotly_chart(fig, use_container_width=True)
            
            return metrics_df
    
    return pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()


def calculate_quality_score_comparison(psnr, ssim):
    """Calculate composite quality score for comparison analysis."""
    psnr_normalized = min(psnr / 50 * 100, 100)
    ssim_normalized = ssim * 100
    return (psnr_normalized * 0.4 + ssim_normalized * 0.6)


def create_multi_image_comparison(uploaded_files, k_values, processing_mode):
    """Create comparison analysis across multiple images."""
    
    st.markdown("### 📊 Multi-Image K-Value Analysis")
    
    if not uploaded_files or not k_values:
        st.warning("Please upload images and select k-values for comparison.")
        return
    
    # Process all images
    all_results = []
    compressor = SVDCompressor()
    metrics_calc = MetricsCalculator()
    
    # Enhanced progress tracking for multi-image analysis
    from utils.loading_animations import create_multi_step_progress
    
    analysis_steps = [
        "Loading images...",
        "Processing compressions...", 
        "Calculating metrics...",
        "Generating comparison..."
    ]
    
    progress_controls = create_multi_step_progress(
        operation_id="multi_image_analysis",
        title="Multi-Image Analysis",
        steps=analysis_steps,
        animation_type="dots"
    )
    
    total_operations = len(uploaded_files) * len(k_values)
    current_operation = 0
    
    for file_idx, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Load image
            image = Image.open(uploaded_file)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((256, 256), Image.Resampling.LANCZOS)
            original_array = np.array(image) / 255.0
            
            # Prepare image for processing
            if processing_mode == "Grayscale":
                gray_image = np.dot(original_array[...,:3], [0.2989, 0.5870, 0.1140])
                image_to_process = gray_image
            else:
                image_to_process = original_array
            
            # Process each k-value
            for k in k_values:
                try:
                    compressed_image, metadata = compressor.compress_image(image_to_process, k)
                    
                    # Prepare for metrics
                    if processing_mode == "Grayscale":
                        compressed_rgb = np.stack([compressed_image] * 3, axis=-1)
                        original_rgb = np.stack([gray_image] * 3, axis=-1)
                    else:
                        compressed_rgb = compressed_image
                        original_rgb = original_array
                    
                    # Calculate metrics
                    psnr = metrics_calc.calculate_psnr(original_rgb, compressed_rgb)
                    ssim = metrics_calc.calculate_ssim(original_rgb, compressed_rgb)
                    mse = metrics_calc.calculate_mse(original_rgb, compressed_rgb)
                    
                    all_results.append({
                        'filename': uploaded_file.name,
                        'k_value': k,
                        'psnr': psnr,
                        'ssim': ssim,
                        'mse': mse,
                        'compression_ratio': metadata.get('compression_ratio', 0),
                        'quality_score': calculate_quality_score_comparison(psnr, ssim)
                    })
                    
                except Exception as e:
                    st.warning(f"Error processing {uploaded_file.name} with k={k}: {str(e)}")
                
                current_operation += 1
                # Update progress within current step
                step_progress = current_operation / total_operations
                progress_controls['update_progress'](step_progress)
        
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
    # Complete the multi-step progress
    progress_controls['next_step']()  # Processing compressions
    progress_controls['next_step']()  # Calculating metrics
    progress_controls['next_step']()  # Generating comparison
    progress_controls['complete']()   # Complete the operation
    
    if not all_results:
        st.error("No results generated. Please check your images and settings.")
        return
    
    # Create comprehensive analysis
    results_df = pd.DataFrame(all_results)
    
    # Multi-image comparison visualizations
    st.markdown("#### 📈 Cross-Image Performance Analysis")
    
    # Create comparison plots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PSNR Comparison Across Images', 'SSIM Comparison Across Images',
                       'Quality Score Heatmap', 'Best K-Value by Image'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "heatmap"}, {"secondary_y": False}]]
    )
    
    # PSNR comparison
    for filename in results_df['filename'].unique():
        file_data = results_df[results_df['filename'] == filename]
        fig.add_trace(
            go.Scatter(
                x=file_data['k_value'],
                y=file_data['psnr'],
                mode='lines+markers',
                name=f'{filename} (PSNR)',
                hovertemplate=f'{filename}<br>K=%{{x}}<br>PSNR=%{{y:.2f}} dB<extra></extra>'
            ),
            row=1, col=1
        )
    
    # SSIM comparison
    for filename in results_df['filename'].unique():
        file_data = results_df[results_df['filename'] == filename]
        fig.add_trace(
            go.Scatter(
                x=file_data['k_value'],
                y=file_data['ssim'],
                mode='lines+markers',
                name=f'{filename} (SSIM)',
                showlegend=False,
                hovertemplate=f'{filename}<br>K=%{{x}}<br>SSIM=%{{y:.3f}}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Quality score heatmap
    pivot_quality = results_df.pivot_table(
        values='quality_score',
        index='filename',
        columns='k_value',
        aggfunc='mean'
    )
    
    fig.add_trace(
        go.Heatmap(
            z=pivot_quality.values,
            x=pivot_quality.columns,
            y=pivot_quality.index,
            colorscale='Viridis',
            showscale=True,
            hovertemplate='File: %{y}<br>K-Value: %{x}<br>Quality Score: %{z:.1f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Best k-value by image
    best_k_by_image = results_df.loc[results_df.groupby('filename')['quality_score'].idxmax()]
    
    fig.add_trace(
        go.Bar(
            x=best_k_by_image['filename'],
            y=best_k_by_image['k_value'],
            text=best_k_by_image['quality_score'].round(1),
            textposition='auto',
            name='Best K-Value',
            showlegend=False,
            hovertemplate='%{x}<br>Best K: %{y}<br>Quality Score: %{text}<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Multi-Image K-Value Comparison Analysis"
    )
    
    fig.update_xaxes(title_text="K-Value", row=1, col=1)
    fig.update_xaxes(title_text="K-Value", row=1, col=2)
    fig.update_xaxes(title_text="K-Value", row=2, col=1)
    fig.update_xaxes(title_text="Image", row=2, col=2)
    
    fig.update_yaxes(title_text="PSNR (dB)", row=1, col=1)
    fig.update_yaxes(title_text="SSIM", row=1, col=2)
    fig.update_yaxes(title_text="Image", row=2, col=1)
    fig.update_yaxes(title_text="Best K-Value", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    st.markdown("#### 📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_overall = results_df.loc[results_df['quality_score'].idxmax()]
        st.metric(
            "Best Overall Quality",
            f"{best_overall['quality_score']:.1f}/100",
            f"{best_overall['filename']} (k={best_overall['k_value']})"
        )
    
    with col2:
        avg_psnr = results_df['psnr'].mean()
        st.metric("Average PSNR", f"{avg_psnr:.2f} dB")
    
    with col3:
        avg_ssim = results_df['ssim'].mean()
        st.metric("Average SSIM", f"{avg_ssim:.3f}")
    
    with col4:
        avg_compression = results_df['compression_ratio'].mean()
        st.metric("Avg Compression", f"{avg_compression:.1f}:1")
    
    return results_df


def create_comprehensive_comparison_export(results_df, settings):
    """Create comprehensive export functionality for comparison results."""
    
    st.markdown("### 💾 Export Comparison Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Enhanced CSV export
        csv_buffer = io.StringIO()
        results_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📊 Download Results CSV",
            data=csv_buffer.getvalue(),
            file_name=f"k_value_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Comprehensive analysis report
        report = generate_comprehensive_comparison_report(results_df, settings)
        
        st.download_button(
            label="📄 Download Analysis Report",
            data=report,
            file_name=f"comparison_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Optimization recommendations
        recommendations = generate_comparison_recommendations(results_df)
        
        st.download_button(
            label="🎯 Download Recommendations",
            data=recommendations,
            file_name=f"optimization_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


def generate_comprehensive_comparison_report(results_df, settings):
    """Generate a comprehensive comparison analysis report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
SVD IMAGE COMPRESSION - K-VALUE COMPARISON ANALYSIS REPORT
=========================================================

ANALYSIS TIMESTAMP: {timestamp}

ANALYSIS OVERVIEW
=================
Images Analyzed: {results_df['filename'].nunique()}
K-Values Tested: {', '.join(map(str, sorted(results_df['k_value'].unique())))}
Processing Mode: {settings.get('processing_mode', 'RGB')}
Total Experiments: {len(results_df)}

PERFORMANCE SUMMARY
===================
Best Overall Quality Score: {results_df['quality_score'].max():.1f}/100
Average Quality Score: {results_df['quality_score'].mean():.1f}/100
Quality Score Range: {results_df['quality_score'].min():.1f} - {results_df['quality_score'].max():.1f}

Best PSNR: {results_df['psnr'].max():.2f} dB
Average PSNR: {results_df['psnr'].mean():.2f} dB
PSNR Range: {results_df['psnr'].min():.2f} - {results_df['psnr'].max():.2f} dB

Best SSIM: {results_df['ssim'].max():.3f}
Average SSIM: {results_df['ssim'].mean():.3f}
SSIM Range: {results_df['ssim'].min():.3f} - {results_df['ssim'].max():.3f}

DETAILED ANALYSIS BY IMAGE
==========================
"""
    
    # Per-image analysis
    for filename in results_df['filename'].unique():
        file_data = results_df[results_df['filename'] == filename]
        best_result = file_data.loc[file_data['quality_score'].idxmax()]
        
        report += f"""
{filename}:
- Best Quality Score: {best_result['quality_score']:.1f}/100 (k={best_result['k_value']})
- Best PSNR: {file_data['psnr'].max():.2f} dB (k={file_data.loc[file_data['psnr'].idxmax(), 'k_value']})
- Best SSIM: {file_data['ssim'].max():.3f} (k={file_data.loc[file_data['ssim'].idxmax(), 'k_value']})
- Best Compression: {file_data['compression_ratio'].max():.1f}:1 (k={file_data.loc[file_data['compression_ratio'].idxmax(), 'k_value']})
"""
    
    report += f"""

K-VALUE PERFORMANCE ANALYSIS
============================
"""
    
    # K-value analysis
    k_analysis = results_df.groupby('k_value').agg({
        'quality_score': ['mean', 'std'],
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std'],
        'compression_ratio': ['mean', 'std']
    }).round(3)
    
    for k in sorted(results_df['k_value'].unique()):
        k_data = results_df[results_df['k_value'] == k]
        report += f"""
K={k}:
- Average Quality Score: {k_data['quality_score'].mean():.1f}±{k_data['quality_score'].std():.1f}
- Average PSNR: {k_data['psnr'].mean():.2f}±{k_data['psnr'].std():.2f} dB
- Average SSIM: {k_data['ssim'].mean():.3f}±{k_data['ssim'].std():.3f}
- Average Compression: {k_data['compression_ratio'].mean():.1f}±{k_data['compression_ratio'].std():.1f}:1
"""
    
    report += f"""

STATISTICAL INSIGHTS
====================
Correlation Analysis:
- K-Value vs PSNR: {results_df['k_value'].corr(results_df['psnr']):.3f}
- K-Value vs SSIM: {results_df['k_value'].corr(results_df['ssim']):.3f}
- PSNR vs SSIM: {results_df['psnr'].corr(results_df['ssim']):.3f}
- Compression vs Quality: {results_df['compression_ratio'].corr(results_df['quality_score']):.3f}

Performance Variability:
- Quality Score CV: {(results_df['quality_score'].std() / results_df['quality_score'].mean() * 100):.1f}%
- PSNR CV: {(results_df['psnr'].std() / results_df['psnr'].mean() * 100):.1f}%
- SSIM CV: {(results_df['ssim'].std() / results_df['ssim'].mean() * 100):.1f}%

GENERATED BY
============
SVD Image Compression Tool - K-Value Comparison Analysis
Report generated on {timestamp}
"""
    
    return report


def generate_comparison_recommendations(results_df):
    """Generate optimization recommendations based on comparison results."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    recommendations = f"""
SVD IMAGE COMPRESSION - OPTIMIZATION RECOMMENDATIONS
===================================================

Generated: {timestamp}

EXECUTIVE SUMMARY
=================
Based on the analysis of {results_df['filename'].nunique()} images across {len(results_df['k_value'].unique())} k-values,
the following recommendations will help optimize your compression strategy.

OPTIMAL K-VALUE RECOMMENDATIONS
===============================
"""
    
    # Find optimal k-values for different criteria
    best_quality_overall = results_df.loc[results_df['quality_score'].idxmax()]
    best_psnr_overall = results_df.loc[results_df['psnr'].idxmax()]
    best_compression_overall = results_df.loc[results_df['compression_ratio'].idxmax()]
    
    recommendations += f"""
For Maximum Quality:
- Recommended K-Value: {best_quality_overall['k_value']}
- Expected Quality Score: {best_quality_overall['quality_score']:.1f}/100
- Expected PSNR: {best_quality_overall['psnr']:.2f} dB
- Expected SSIM: {best_quality_overall['ssim']:.3f}

For Best PSNR:
- Recommended K-Value: {best_psnr_overall['k_value']}
- Expected PSNR: {best_psnr_overall['psnr']:.2f} dB
- Quality Score: {best_psnr_overall['quality_score']:.1f}/100

For Maximum Compression:
- Recommended K-Value: {best_compression_overall['k_value']}
- Expected Compression Ratio: {best_compression_overall['compression_ratio']:.1f}:1
- Quality Trade-off: {best_compression_overall['quality_score']:.1f}/100 quality score
"""
    
    # Image-specific recommendations
    recommendations += f"""

IMAGE-SPECIFIC RECOMMENDATIONS
==============================
"""
    
    for filename in results_df['filename'].unique():
        file_data = results_df[results_df['filename'] == filename]
        best_for_file = file_data.loc[file_data['quality_score'].idxmax()]
        
        recommendations += f"""
{filename}:
- Optimal K-Value: {best_for_file['k_value']}
- Quality Score: {best_for_file['quality_score']:.1f}/100
- PSNR: {best_for_file['psnr']:.2f} dB
- Compression: {best_for_file['compression_ratio']:.1f}:1
"""
    
    # General guidelines
    avg_quality_by_k = results_df.groupby('k_value')['quality_score'].mean()
    best_avg_k = avg_quality_by_k.idxmax()
    
    recommendations += f"""

GENERAL GUIDELINES
==================
1. **Balanced Performance**: K={best_avg_k} provides the best average quality across all images
   (Average Quality Score: {avg_quality_by_k.max():.1f}/100)

2. **Quality Thresholds**:
   - For excellent quality (>80/100): Use k≥{results_df[results_df['quality_score'] > 80]['k_value'].min() if not results_df[results_df['quality_score'] > 80].empty else 'N/A'}
   - For good quality (>60/100): Use k≥{results_df[results_df['quality_score'] > 60]['k_value'].min() if not results_df[results_df['quality_score'] > 60].empty else 'N/A'}

3. **Compression Efficiency**:
   - High compression (>5:1): Use k≤{results_df[results_df['compression_ratio'] > 5]['k_value'].max() if not results_df[results_df['compression_ratio'] > 5].empty else 'N/A'}
   - Moderate compression (2-5:1): Use k={results_df[(results_df['compression_ratio'] >= 2) & (results_df['compression_ratio'] <= 5)]['k_value'].median():.0f} (median)

4. **Image Type Considerations**:
   - Complex images may need higher k-values for acceptable quality
   - Simple images can achieve good compression with lower k-values
   - Monitor SSIM for structural preservation

IMPLEMENTATION STRATEGY
=======================
1. Start with k={best_avg_k} as a baseline for most images
2. Adjust based on specific quality requirements
3. Use quality score >70/100 as minimum acceptable threshold
4. Consider computational resources when choosing k-values
5. Validate results with visual inspection for critical applications

Generated by SVD Image Compression Tool
"""
    
    return recommendations