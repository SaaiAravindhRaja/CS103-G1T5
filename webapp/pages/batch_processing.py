"""
Enhanced batch processing page for multiple image analysis with advanced features.
"""

import streamlit as st
import numpy as np
import pandas as pd
import io
import zipfile
from PIL import Image
import sys
from pathlib import Path
import time
import json
from datetime import datetime

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from compression.svd_compressor import SVDCompressor
from evaluation.metrics_calculator import MetricsCalculator
from utils.styling import show_loading_animation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots


def show():
    """Display the enhanced batch processing interface within single page design."""
    
    # Use the new layout system
    from utils.styling import create_main_content_area, close_main_content_area
    
    # Create main content area
    create_main_content_area()
    
    st.markdown("# 📊 Batch Processing")
    st.markdown("Process multiple images simultaneously with comprehensive analysis and batch download capabilities.")
    
    # Create the enhanced batch processing interface
    render_batch_processing_interface()
    

    
    # Close main content area
    close_main_content_area()


def render_batch_processing_interface():
    """Render the enhanced batch processing interface within single page design."""
    
    # Initialize session state for batch processing
    if 'batch_state' not in st.session_state:
        st.session_state.batch_state = {
            'uploaded_files': [],
            'processing_status': 'idle',
            'results': None,
            'processed_images': {},
            'selected_files': [],
            'processing_config': {}
        }
    
    batch_state = st.session_state.batch_state
    
    # Create main interface sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Multiple file upload and management interface
        render_batch_upload_interface(batch_state)
        
        # Batch processing controls and progress tracking
        if batch_state['uploaded_files']:
            render_batch_processing_controls(batch_state)
    
    with col2:
        # File management and selection interface
        if batch_state['uploaded_files']:
            render_file_management_interface(batch_state)
    
    # Results and batch download functionality
    if batch_state['results'] is not None:
        st.markdown("---")
        render_batch_results_interface(batch_state)


def render_batch_upload_interface(batch_state):
    """Render the multiple file upload interface with enhanced management."""
    
    st.markdown("### 📁 Multiple File Upload")
    
    # Enhanced upload component for batch processing
    from utils.upload_component import create_enhanced_upload_component
    
    upload_data = create_enhanced_upload_component(
        key="batch_upload",
        multiple=True,
        show_preview=True,
        show_progress=True
    )
    
    if upload_data and 'all_files' in upload_data:
        # Update batch state with uploaded files
        batch_state['uploaded_files'] = upload_data['all_files']
        
        # Show upload summary
        st.success(f"✅ {len(batch_state['uploaded_files'])} images uploaded successfully!")
        
        # Quick stats
        total_size = sum(file_data['metadata']['file_size_mb'] for file_data in batch_state['uploaded_files'])
        avg_dimensions = []
        for file_data in batch_state['uploaded_files']:
            dims = file_data['metadata']['dimensions']
            avg_dimensions.append(dims[0] * dims[1])
        
        avg_pixels = sum(avg_dimensions) / len(avg_dimensions) if avg_dimensions else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(batch_state['uploaded_files']))
        with col2:
            st.metric("Total Size", f"{total_size:.1f} MB")
        with col3:
            st.metric("Avg Resolution", f"{int(avg_pixels**0.5)}×{int(avg_pixels**0.5)}")


def render_file_management_interface(batch_state):
    """Render the file management and selection interface."""
    
    st.markdown("### 📋 File Management")
    
    uploaded_files = batch_state['uploaded_files']
    
    # File selection for processing
    st.markdown("#### Select Files to Process")
    
    # Select all/none buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All", use_container_width=True):
            batch_state['selected_files'] = list(range(len(uploaded_files)))
            st.rerun()
    
    with col2:
        if st.button("Clear Selection", use_container_width=True):
            batch_state['selected_files'] = []
            st.rerun()
    
    # Individual file selection with thumbnails
    st.markdown("#### File List")
    
    for idx, file_data in enumerate(uploaded_files):
        metadata = file_data['metadata']
        
        # Create checkbox for file selection
        is_selected = idx in batch_state.get('selected_files', [])
        
        with st.container():
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Thumbnail
                thumbnail = file_data['pil'].copy()
                thumbnail.thumbnail((80, 80))
                st.image(thumbnail, width=80)
            
            with col2:
                # File info and selection
                selected = st.checkbox(
                    f"**{metadata['filename']}**",
                    value=is_selected,
                    key=f"file_select_{idx}",
                    help=f"Size: {metadata['file_size_mb']:.1f}MB, Dimensions: {metadata['dimensions'][0]}×{metadata['dimensions'][1]}"
                )
                
                if selected and idx not in batch_state['selected_files']:
                    batch_state['selected_files'].append(idx)
                elif not selected and idx in batch_state['selected_files']:
                    batch_state['selected_files'].remove(idx)
                
                # File details
                st.caption(f"📐 {metadata['dimensions'][0]}×{metadata['dimensions'][1]} | 💾 {metadata['file_size_mb']:.1f}MB")
    
    # Show selection summary
    if batch_state['selected_files']:
        selected_count = len(batch_state['selected_files'])
        st.info(f"📊 {selected_count} file{'s' if selected_count != 1 else ''} selected for processing")


def render_batch_processing_controls(batch_state):
    """Render batch processing controls and progress tracking."""
    
    st.markdown("### ⚙️ Batch Processing Controls")
    
    if not batch_state['selected_files']:
        st.warning("⚠️ Please select files to process from the file management panel.")
        return
    
    # Processing configuration
    with st.expander("🔧 Processing Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # K-values configuration
            st.markdown("**Compression Levels (K-values)**")
            
            k_mode = st.radio(
                "K-value selection:",
                ["Quick Test (3 values)", "Standard Range", "Custom Values"],
                help="Choose compression levels to test"
            )
            
            if k_mode == "Quick Test (3 values)":
                k_values = [10, 25, 50]
            elif k_mode == "Standard Range":
                k_min = st.number_input("Min K", min_value=1, max_value=100, value=5)
                k_max = st.number_input("Max K", min_value=k_min, max_value=256, value=50)
                k_step = st.number_input("Step", min_value=1, max_value=20, value=10)
                k_values = list(range(k_min, k_max + 1, k_step))
            else:
                k_input = st.text_input("Custom K-values (comma-separated)", value="5,10,20,30,50")
                try:
                    k_values = [int(k.strip()) for k in k_input.split(',') if k.strip().isdigit()]
                except:
                    k_values = [5, 10, 20, 30, 50]
            
            st.info(f"Will test: {k_values}")
        
        with col2:
            # Processing options
            st.markdown("**Processing Options**")
            
            processing_mode = st.selectbox(
                "Color mode:",
                ["RGB (Color)", "Grayscale", "Both"],
                help="Choose how to process images"
            )
            
            resize_images = st.checkbox(
                "Resize to 256×256",
                value=True,
                help="Standardize image sizes for consistent processing"
            )
            
            save_compressed = st.checkbox(
                "Save compressed images",
                value=True,
                help="Enable batch download of compressed images"
            )
    
    # Store configuration
    batch_state['processing_config'] = {
        'k_values': k_values,
        'processing_mode': processing_mode,
        'resize_images': resize_images,
        'save_compressed': save_compressed
    }
    
    # Processing controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Start processing button
        if batch_state['processing_status'] == 'idle':
            if st.button(
                f"🚀 Start Batch Processing ({len(batch_state['selected_files'])} files)",
                type="primary",
                use_container_width=True
            ):
                start_batch_processing(batch_state)
        
        elif batch_state['processing_status'] == 'processing':
            st.info("🔄 Processing in progress...")
            if st.button("⏹️ Stop Processing", use_container_width=True):
                batch_state['processing_status'] = 'idle'
                st.rerun()
    
    with col2:
        # Processing status
        if batch_state['processing_status'] == 'processing':
            st.markdown("**Status:** 🔄 Processing")
        elif batch_state['processing_status'] == 'completed':
            st.markdown("**Status:** ✅ Complete")
        else:
            st.markdown("**Status:** ⏸️ Ready")


def start_batch_processing(batch_state):
    """Start the batch processing operation with progress tracking."""
    
    batch_state['processing_status'] = 'processing'
    
    # Get selected files and configuration
    uploaded_files = batch_state['uploaded_files']
    selected_indices = batch_state['selected_files']
    config = batch_state['processing_config']
    
    selected_files = [uploaded_files[i] for i in selected_indices]
    
    # Create progress tracking interface
    st.markdown("### 🔄 Processing Progress")
    
    # Overall progress
    overall_progress = st.progress(0)
    status_text = st.empty()
    
    # Detailed progress
    progress_container = st.container()
    
    results = []
    processed_images = {}
    
    # Import required modules
    from compression.svd_compressor import SVDCompressor
    from evaluation.metrics_calculator import MetricsCalculator
    
    compressor = SVDCompressor()
    metrics_calc = MetricsCalculator()
    
    total_operations = len(selected_files) * len(config['k_values'])
    if config['processing_mode'] == "Both":
        total_operations *= 2
    
    current_operation = 0
    
    try:
        for file_idx, file_data in enumerate(selected_files):
            file_obj = file_data['file']
            metadata = file_data['metadata']
            
            status_text.text(f"Processing {metadata['filename']}... ({file_idx + 1}/{len(selected_files)})")
            
            # Load and preprocess image
            try:
                image_array = file_data['array']
                
                if config['resize_images'] and image_array.shape[:2] != (256, 256):
                    from PIL import Image
                    pil_image = Image.fromarray((image_array * 255).astype('uint8'))
                    pil_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
                    image_array = np.array(pil_image) / 255.0
                
                # Process for each mode
                modes_to_process = []
                if config['processing_mode'] in ["RGB (Color)", "Both"]:
                    modes_to_process.append(("RGB", image_array))
                if config['processing_mode'] in ["Grayscale", "Both"]:
                    gray_array = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
                    modes_to_process.append(("Grayscale", gray_array))
                
                for mode_name, img_array in modes_to_process:
                    for k in config['k_values']:
                        try:
                            # Compress image
                            compressed_image, compression_metadata = compressor.compress_image(img_array, k)
                            
                            # Prepare for metrics calculation
                            if mode_name == "Grayscale":
                                compressed_rgb = np.stack([compressed_image] * 3, axis=-1)
                                original_rgb = np.stack([img_array] * 3, axis=-1)
                            else:
                                compressed_rgb = compressed_image
                                original_rgb = img_array
                            
                            # Calculate metrics
                            psnr = metrics_calc.calculate_psnr(original_rgb, compressed_rgb)
                            ssim = metrics_calc.calculate_ssim(original_rgb, compressed_rgb)
                            mse = metrics_calc.calculate_mse(original_rgb, compressed_rgb)
                            
                            # Store results
                            result = {
                                'filename': metadata['filename'],
                                'mode': mode_name,
                                'k_value': k,
                                'psnr': psnr,
                                'ssim': ssim,
                                'mse': mse,
                                'compression_ratio': compression_metadata.get('compression_ratio', 0),
                                'file_size_kb': metadata['file_size_mb'] * 1024
                            }
                            results.append(result)
                            
                            # Store compressed image if requested
                            if config['save_compressed']:
                                key = f"{metadata['filename']}_{mode_name}_k{k}"
                                processed_images[key] = compressed_rgb
                            
                        except Exception as e:
                            st.warning(f"Error processing {metadata['filename']} with k={k}: {str(e)}")
                        
                        current_operation += 1
                        overall_progress.progress(current_operation / total_operations)
                
            except Exception as e:
                st.error(f"Error loading {metadata['filename']}: {str(e)}")
                continue
        
        # Store results
        batch_state['results'] = pd.DataFrame(results)
        batch_state['processed_images'] = processed_images
        batch_state['processing_status'] = 'completed'
        
        status_text.text("✅ Batch processing completed successfully!")
        st.success(f"Successfully processed {len(selected_files)} images with {len(config['k_values'])} k-values each!")
        
        # Auto-refresh to show results
        time.sleep(2)
        st.rerun()
        
    except Exception as e:
        batch_state['processing_status'] = 'idle'
        st.error(f"Batch processing failed: {str(e)}")


def render_batch_results_interface(batch_state):
    """Render batch results with enhanced download functionality."""
    
    st.markdown("### 📈 Batch Processing Results")
    
    results_df = batch_state['results']
    processed_images = batch_state['processed_images']
    
    if results_df.empty:
        st.warning("No results to display.")
        return
    
    # Results summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Images Processed", results_df['filename'].nunique())
    
    with col2:
        st.metric("Total Experiments", len(results_df))
    
    with col3:
        avg_psnr = results_df['psnr'].mean()
        st.metric("Avg PSNR", f"{avg_psnr:.2f} dB")
    
    with col4:
        avg_ssim = results_df['ssim'].mean()
        st.metric("Avg SSIM", f"{avg_ssim:.3f}")
    
    # Enhanced batch download functionality
    st.markdown("---")
    render_batch_download_interface(batch_state)
    
    # Results visualization
    st.markdown("### 📊 Results Analysis")
    
    # Quick results table
    with st.expander("📋 Detailed Results Table", expanded=False):
        # Add quality score
        display_df = results_df.copy()
        display_df['Quality Score'] = display_df.apply(
            lambda row: calculate_quality_score_batch(row['psnr'], row['ssim']), axis=1
        ).round(1)
        
        st.dataframe(
            display_df.round(3),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Quality Score": st.column_config.ProgressColumn(
                    "Quality Score",
                    help="Composite quality metric (0-100)",
                    min_value=0,
                    max_value=100,
                ),
                "psnr": st.column_config.NumberColumn("PSNR (dB)", format="%.2f"),
                "ssim": st.column_config.NumberColumn("SSIM", format="%.3f"),
                "compression_ratio": st.column_config.NumberColumn("Compression Ratio", format="%.1f:1")
            }
        )
    
    # Interactive analysis
    create_interactive_batch_plots(results_df)


def render_batch_download_interface(batch_state):
    """Render the enhanced batch download functionality."""
    
    st.markdown("### 📥 Batch Download Options")
    
    results_df = batch_state['results']
    processed_images = batch_state['processed_images']
    
    if results_df.empty:
        st.info("No results available for download.")
        return
    
    # Download configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Download Configuration")
        
        # Download format options
        download_format = st.selectbox(
            "Image format:",
            ["PNG", "JPEG", "TIFF"],
            help="Choose output format for compressed images"
        )
        
        # Quality selection for downloads
        quality_filter = st.selectbox(
            "Quality filter:",
            ["All Results", "High Quality Only (PSNR > 25dB)", "Best Results Only (Top 25%)", "Custom Filter"],
            help="Filter results by quality metrics"
        )
        
        # Custom filter options
        if quality_filter == "Custom Filter":
            min_psnr = st.slider("Minimum PSNR (dB)", 0.0, 50.0, 20.0)
            min_ssim = st.slider("Minimum SSIM", 0.0, 1.0, 0.5)
            filtered_df = results_df[(results_df['psnr'] >= min_psnr) & (results_df['ssim'] >= min_ssim)]
        elif quality_filter == "High Quality Only (PSNR > 25dB)":
            filtered_df = results_df[results_df['psnr'] > 25]
        elif quality_filter == "Best Results Only (Top 25%)":
            quality_scores = results_df.apply(lambda row: calculate_quality_score_batch(row['psnr'], row['ssim']), axis=1)
            threshold = quality_scores.quantile(0.75)
            filtered_df = results_df[quality_scores >= threshold]
        else:
            filtered_df = results_df
        
        st.info(f"📊 {len(filtered_df)} results match the selected criteria")
    
    with col2:
        st.markdown("#### Download Actions")
        
        # Individual download options
        if st.button("📄 Download Results CSV", use_container_width=True):
            csv_data = create_results_csv(filtered_df)
            st.download_button(
                label="💾 Download CSV File",
                data=csv_data,
                file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Compressed images download
        if processed_images and st.button("🖼️ Download Compressed Images", use_container_width=True):
            # Filter images based on quality criteria
            filtered_keys = []
            for _, row in filtered_df.iterrows():
                key = f"{row['filename']}_{row['mode']}_k{row['k_value']}"
                if key in processed_images:
                    filtered_keys.append(key)
            
            if filtered_keys:
                zip_data = create_images_zip_filtered(processed_images, filtered_keys, download_format)
                if zip_data:
                    st.download_button(
                        label="📦 Download Images ZIP",
                        data=zip_data,
                        file_name=f"compressed_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
            else:
                st.warning("No compressed images available for the selected criteria.")
        
        # Comprehensive report download
        if st.button("📊 Download Full Report", use_container_width=True):
            report_data = create_comprehensive_batch_report(filtered_df)
            st.download_button(
                label="📋 Download Report",
                data=report_data,
                file_name=f"batch_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Batch download summary
    st.markdown("---")
    st.markdown("#### 📋 Download Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("Available Results", len(filtered_df))
    
    with summary_col2:
        available_images = len([k for k in processed_images.keys() 
                              if any(f"{row['filename']}_{row['mode']}_k{row['k_value']}" == k 
                                    for _, row in filtered_df.iterrows())])
        st.metric("Available Images", available_images)
    
    with summary_col3:
        total_size_mb = sum(len(str(processed_images[k])) for k in processed_images.keys()) / (1024 * 1024)
        st.metric("Est. Download Size", f"{total_size_mb:.1f} MB")


def create_results_csv(df):
    """Create CSV data from results dataframe."""
    return df.to_csv(index=False)


def create_images_zip_filtered(processed_images, filtered_keys, format_type):
    """Create a ZIP file containing filtered compressed images."""
    
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for key in filtered_keys:
                if key in processed_images:
                    image_array = processed_images[key]
                    
                    # Convert to PIL Image
                    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
                    
                    # Save to buffer
                    img_buffer = io.BytesIO()
                    image_pil.save(img_buffer, format=format_type)
                    img_buffer.seek(0)
                    
                    # Add to ZIP with proper extension
                    extension = format_type.lower()
                    if extension == 'jpeg':
                        extension = 'jpg'
                    
                    zip_file.writestr(f"{key}.{extension}", img_buffer.getvalue())
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating ZIP file: {str(e)}")
        return None


def create_comprehensive_batch_report(df):
    """Create a comprehensive text report of batch processing results."""
    
    report = f"""
SVD IMAGE COMPRESSION - BATCH ANALYSIS REPORT
=============================================
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

EXECUTIVE SUMMARY
=================
Total Images Processed: {df['filename'].nunique()}
Total Experiments: {len(df)}
Processing Modes: {', '.join(df['mode'].unique())}
K-Values Tested: {sorted(df['k_value'].unique())}

QUALITY METRICS OVERVIEW
========================
Average PSNR: {df['psnr'].mean():.2f} dB (Range: {df['psnr'].min():.2f} - {df['psnr'].max():.2f})
Average SSIM: {df['ssim'].mean():.3f} (Range: {df['ssim'].min():.3f} - {df['ssim'].max():.3f})
Average Compression Ratio: {df['compression_ratio'].mean():.1f}:1

BEST PERFORMING CONFIGURATIONS
==============================
"""
    
    # Find best results
    best_psnr = df.loc[df['psnr'].idxmax()]
    best_ssim = df.loc[df['ssim'].idxmax()]
    best_compression = df.loc[df['compression_ratio'].idxmax()]
    
    report += f"""
Best PSNR: {best_psnr['psnr']:.2f} dB
- File: {best_psnr['filename']}
- Mode: {best_psnr['mode']}
- K-Value: {best_psnr['k_value']}

Best SSIM: {best_ssim['ssim']:.3f}
- File: {best_ssim['filename']}
- Mode: {best_ssim['mode']}
- K-Value: {best_ssim['k_value']}

Best Compression: {best_compression['compression_ratio']:.1f}:1
- File: {best_compression['filename']}
- Mode: {best_compression['mode']}
- K-Value: {best_compression['k_value']}

DETAILED RESULTS BY IMAGE
=========================
"""
    
    # Add per-image analysis
    for filename in df['filename'].unique():
        file_data = df[df['filename'] == filename]
        best_result = file_data.loc[file_data['psnr'].idxmax()]
        
        report += f"""
{filename}:
- Best PSNR: {best_result['psnr']:.2f} dB (k={best_result['k_value']})
- Best SSIM: {file_data['ssim'].max():.3f} (k={file_data.loc[file_data['ssim'].idxmax(), 'k_value']})
- File Size: {file_data['file_size_kb'].iloc[0]:.1f} KB
"""
    
    report += f"""

RECOMMENDATIONS
===============
"""
    
    # Add recommendations
    recommendations = generate_optimization_recommendations(df)
    for category, recs in recommendations.items():
        report += f"\n{category}:\n"
        for rec in recs:
            report += f"- {rec}\n"
    
    report += f"""

TECHNICAL DETAILS
=================
SVD Compression Method:
- Singular Value Decomposition factorizes image matrix A = U * Σ * V^T
- Compression achieved by keeping only top k singular values
- Quality vs compression trade-off controlled by k parameter

Processing Parameters:
- Standard image size: 256×256 pixels
- Pixel value normalization: [0, 1] range
- Color space: RGB and/or Grayscale as specified

GENERATED BY
============
SVD Image Compression Tool - Batch Analysis Module
Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    return report


# Legacy function - replaced by new batch processing interface
# This function is kept for backward compatibility but is no longer used


# Legacy function - replaced by new batch results interface
# This function is kept for backward compatibility but is no longer used


# Legacy function - replaced by create_images_zip_filtered
# This function is kept for backward compatibility but is no longer used





def create_batch_comparison_grid(df, selected_files, k_values):
    """Create an interactive comparison grid for different k-values."""
    
    st.markdown("### 🔍 K-Value Comparison Grid")
    
    if len(selected_files) == 0 or len(k_values) == 0:
        st.warning("Please select files and k-values to display comparison grid.")
        return
    
    # Select file for comparison
    selected_file = st.selectbox(
        "Select image for k-value comparison:",
        selected_files,
        help="Choose which image to show across different k-values"
    )
    
    # Filter data for selected file
    file_data = df[df['filename'] == selected_file]
    
    if file_data.empty:
        st.warning("No data available for selected file.")
        return
    
    # Create comparison grid
    st.markdown(f"#### Compression Comparison for: {selected_file}")
    
    # Show original image info
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("**Original Image Properties:**")
        file_size = file_data['file_size_kb'].iloc[0]
        st.markdown(f"- File size: {file_size:.1f} KB")
        st.markdown(f"- Dimensions: 256×256 pixels")
    
    with col_info2:
        st.markdown("**Comparison Overview:**")
        st.markdown(f"- K-values tested: {len(k_values)}")
        st.markdown(f"- Processing modes: {', '.join(file_data['mode'].unique())}")
    
    # Create interactive comparison table
    comparison_data = []
    for _, row in file_data.iterrows():
        comparison_data.append({
            'K-Value': row['k_value'],
            'Mode': row['mode'],
            'PSNR (dB)': f"{row['psnr']:.2f}",
            'SSIM': f"{row['ssim']:.3f}",
            'Compression Ratio': f"{row['compression_ratio']:.1f}:1",
            'Quality Score': f"{calculate_quality_score_batch(row['psnr'], row['ssim']):.1f}/100"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display with color coding
    st.markdown("**Quality Metrics Comparison:**")
    
    # Color code the dataframe
    def color_quality(val):
        if 'PSNR' in val.name:
            num_val = float(val.replace(' (dB)', ''))
            if num_val > 30:
                return 'background-color: #dcfce7'  # Green
            elif num_val > 20:
                return 'background-color: #fef3c7'  # Yellow
            else:
                return 'background-color: #fee2e2'  # Red
        elif 'SSIM' in val.name:
            num_val = float(val)
            if num_val > 0.8:
                return 'background-color: #dcfce7'
            elif num_val > 0.6:
                return 'background-color: #fef3c7'
            else:
                return 'background-color: #fee2e2'
        return ''
    
    styled_df = comparison_df.style.applymap(color_quality, subset=['PSNR (dB)', 'SSIM'])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Interactive quality progression chart
    st.markdown("**Quality Progression Chart:**")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('PSNR vs K-Value', 'SSIM vs K-Value'),
        vertical_spacing=0.1
    )
    
    for mode in file_data['mode'].unique():
        mode_data = file_data[file_data['mode'] == mode].sort_values('k_value')
        
        # PSNR plot
        fig.add_trace(
            go.Scatter(
                x=mode_data['k_value'],
                y=mode_data['psnr'],
                mode='lines+markers',
                name=f'PSNR ({mode})',
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate='K=%{x}<br>PSNR=%{y:.2f} dB<extra></extra>'
            ),
            row=1, col=1
        )
        
        # SSIM plot
        fig.add_trace(
            go.Scatter(
                x=mode_data['k_value'],
                y=mode_data['ssim'],
                mode='lines+markers',
                name=f'SSIM ({mode})',
                line=dict(width=3),
                marker=dict(size=8),
                hovertemplate='K=%{x}<br>SSIM=%{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        height=600,
        title_text=f"Quality Metrics Progression - {selected_file}",
        showlegend=True
    )
    
    fig.update_xaxes(title_text="K-Value", row=2, col=1)
    fig.update_yaxes(title_text="PSNR (dB)", row=1, col=1)
    fig.update_yaxes(title_text="SSIM", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def calculate_quality_score_batch(psnr, ssim):
    """Calculate composite quality score for batch processing."""
    psnr_normalized = min(psnr / 50 * 100, 100)
    ssim_normalized = ssim * 100
    return (psnr_normalized * 0.4 + ssim_normalized * 0.6)


def create_interactive_batch_plots(df):
    """Create interactive plots for PSNR/SSIM analysis across images."""
    
    st.markdown("### 📈 Interactive Analysis Dashboard")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Quality Overview", 
        "🔍 Detailed Analysis", 
        "📈 Trends & Patterns",
        "🎯 Optimization Guide"
    ])
    
    with tab1:
        # Quality overview dashboard
        st.markdown("#### Quality Metrics Overview")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_psnr = df['psnr'].mean()
            st.metric("Average PSNR", f"{avg_psnr:.2f} dB", 
                     delta=f"{avg_psnr - 25:.1f}" if avg_psnr > 25 else None)
        
        with col2:
            avg_ssim = df['ssim'].mean()
            st.metric("Average SSIM", f"{avg_ssim:.3f}",
                     delta=f"{avg_ssim - 0.7:.2f}" if avg_ssim > 0.7 else None)
        
        with col3:
            avg_compression = df['compression_ratio'].mean()
            st.metric("Avg Compression", f"{avg_compression:.1f}:1")
        
        with col4:
            best_quality = df.apply(lambda row: calculate_quality_score_batch(row['psnr'], row['ssim']), axis=1).max()
            st.metric("Best Quality Score", f"{best_quality:.1f}/100")
        
        # Quality distribution
        fig_dist = make_subplots(
            rows=1, cols=2,
            subplot_titles=('PSNR Distribution', 'SSIM Distribution')
        )
        
        fig_dist.add_trace(
            go.Histogram(x=df['psnr'], nbinsx=20, name='PSNR', opacity=0.7),
            row=1, col=1
        )
        
        fig_dist.add_trace(
            go.Histogram(x=df['ssim'], nbinsx=20, name='SSIM', opacity=0.7),
            row=1, col=2
        )
        
        fig_dist.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with tab2:
        # Detailed analysis with filtering
        st.markdown("#### Detailed Quality Analysis")
        
        # Interactive filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_modes = st.multiselect(
                "Processing Modes:",
                df['mode'].unique(),
                default=df['mode'].unique()
            )
        
        with col2:
            k_range = st.slider(
                "K-Value Range:",
                int(df['k_value'].min()),
                int(df['k_value'].max()),
                (int(df['k_value'].min()), int(df['k_value'].max()))
            )
        
        with col3:
            quality_threshold = st.slider(
                "Min Quality Score:",
                0, 100, 60
            )
        
        # Filter data
        filtered_df = df[
            (df['mode'].isin(selected_modes)) &
            (df['k_value'] >= k_range[0]) &
            (df['k_value'] <= k_range[1])
        ]
        
        # Add quality score column
        filtered_df = filtered_df.copy()
        filtered_df['quality_score'] = filtered_df.apply(
            lambda row: calculate_quality_score_batch(row['psnr'], row['ssim']), axis=1
        )
        
        # Filter by quality threshold
        high_quality_df = filtered_df[filtered_df['quality_score'] >= quality_threshold]
        
        # 3D scatter plot
        fig_3d = go.Figure(data=go.Scatter3d(
            x=high_quality_df['k_value'],
            y=high_quality_df['psnr'],
            z=high_quality_df['ssim'],
            mode='markers',
            marker=dict(
                size=8,
                color=high_quality_df['compression_ratio'],
                colorscale='Viridis',
                colorbar=dict(title="Compression Ratio"),
                opacity=0.8
            ),
            text=high_quality_df['filename'],
            hovertemplate='<b>%{text}</b><br>' +
                         'K-Value: %{x}<br>' +
                         'PSNR: %{y:.2f} dB<br>' +
                         'SSIM: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        fig_3d.update_layout(
            title="3D Quality Analysis (K-Value vs PSNR vs SSIM)",
            scene=dict(
                xaxis_title="K-Value",
                yaxis_title="PSNR (dB)",
                zaxis_title="SSIM"
            ),
            height=600
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    with tab3:
        # Trends and patterns analysis
        st.markdown("#### Trends & Patterns Analysis")
        
        # Performance trends by file
        fig_trends = px.line(
            df,
            x='k_value',
            y='psnr',
            color='filename',
            facet_col='mode',
            title='PSNR Trends Across Files and Modes',
            labels={'psnr': 'PSNR (dB)', 'k_value': 'K-Value'}
        )
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Correlation analysis
        st.markdown("**Correlation Analysis:**")
        
        correlation_data = df[['k_value', 'psnr', 'ssim', 'compression_ratio']].corr()
        
        fig_corr = px.imshow(
            correlation_data,
            title='Metrics Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Statistical insights
        st.markdown("**Key Insights:**")
        
        insights = []
        
        # K-value vs quality relationship
        k_psnr_corr = df['k_value'].corr(df['psnr'])
        if k_psnr_corr > 0.7:
            insights.append("🔍 Strong positive correlation between K-value and PSNR quality")
        
        # Best performing files
        best_file = df.groupby('filename')['psnr'].mean().idxmax()
        insights.append(f"🏆 Best performing image: {best_file}")
        
        # Optimal k-value range
        high_quality_k = df[df['psnr'] > df['psnr'].quantile(0.8)]['k_value']
        if not high_quality_k.empty:
            insights.append(f"🎯 Optimal K-value range: {high_quality_k.min()}-{high_quality_k.max()}")
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    with tab4:
        # Optimization recommendations
        st.markdown("#### Optimization Guide & Recommendations")
        
        # Generate recommendations based on data
        recommendations = generate_optimization_recommendations(df)
        
        for category, recs in recommendations.items():
            st.markdown(f"**{category}:**")
            for rec in recs:
                st.markdown(f"- {rec}")
            st.markdown("")


def generate_optimization_recommendations(df):
    """Generate optimization recommendations based on batch results."""
    
    recommendations = {
        "Quality Optimization": [],
        "Compression Efficiency": [],
        "Processing Strategy": []
    }
    
    # Quality analysis
    avg_psnr = df['psnr'].mean()
    avg_ssim = df['ssim'].mean()
    
    if avg_psnr < 25:
        recommendations["Quality Optimization"].append(
            "Consider increasing K-values to improve PSNR (currently below 25 dB threshold)"
        )
    
    if avg_ssim < 0.7:
        recommendations["Quality Optimization"].append(
            "SSIM values are low - try higher K-values for better structural similarity"
        )
    
    # Find optimal k-values
    quality_scores = df.apply(lambda row: calculate_quality_score_batch(row['psnr'], row['ssim']), axis=1)
    best_k = df.loc[quality_scores.idxmax(), 'k_value']
    
    recommendations["Quality Optimization"].append(
        f"Best overall quality achieved at K={best_k}"
    )
    
    # Compression analysis
    avg_compression = df['compression_ratio'].mean()
    
    if avg_compression < 3:
        recommendations["Compression Efficiency"].append(
            "Low compression ratios - consider decreasing K-values for better space savings"
        )
    
    # Processing strategy
    if 'RGB' in df['mode'].values and 'Grayscale' in df['mode'].values:
        rgb_psnr = df[df['mode'] == 'RGB']['psnr'].mean()
        gray_psnr = df[df['mode'] == 'Grayscale']['psnr'].mean()
        
        if abs(rgb_psnr - gray_psnr) < 2:
            recommendations["Processing Strategy"].append(
                "RGB and Grayscale modes show similar quality - consider Grayscale for faster processing"
            )
    
    # File-specific recommendations
    file_performance = df.groupby('filename')['psnr'].mean()
    worst_file = file_performance.idxmin()
    best_file = file_performance.idxmax()
    
    recommendations["Processing Strategy"].append(
        f"Image '{worst_file}' may need special attention (lowest average PSNR)"
    )
    
    recommendations["Processing Strategy"].append(
        f"Image '{best_file}' compresses very well (highest average PSNR)"
    )
    
    return recommendations


def create_comprehensive_export_functionality(df, processed_images):
    """Create comprehensive export functionality for batch results."""
    
    st.markdown("### 💾 Comprehensive Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Data Export**")
        
        # Enhanced CSV export with additional metrics
        enhanced_df = df.copy()
        enhanced_df['quality_score'] = enhanced_df.apply(
            lambda row: calculate_quality_score_batch(row['psnr'], row['ssim']), axis=1
        )
        enhanced_df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        csv_buffer = io.StringIO()
        enhanced_df.to_csv(csv_buffer, index=False)
        
        st.download_button(
            label="📈 Download Enhanced CSV",
            data=csv_buffer.getvalue(),
            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # JSON export for programmatic use
        json_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_images': df['filename'].nunique(),
                'total_experiments': len(df),
                'k_values_tested': sorted(df['k_value'].unique().tolist()),
                'processing_modes': df['mode'].unique().tolist()
            },
            'summary_statistics': {
                'avg_psnr': float(df['psnr'].mean()),
                'avg_ssim': float(df['ssim'].mean()),
                'avg_compression_ratio': float(df['compression_ratio'].mean()),
                'best_quality_score': float(enhanced_df['quality_score'].max())
            },
            'detailed_results': enhanced_df.to_dict('records')
        }
        
        json_str = json.dumps(json_data, indent=2)
        
        st.download_button(
            label="📋 Download JSON Report",
            data=json_str,
            file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        st.markdown("**🖼️ Image Export**")
        
        if processed_images:
            # Create organized ZIP with folders
            if st.button("📦 Create Organized ZIP", use_container_width=True):
                zip_buffer = create_organized_images_zip(processed_images, df)
                if zip_buffer:
                    st.download_button(
                        label="📥 Download Organized Images",
                        data=zip_buffer,
                        file_name=f"batch_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
        else:
            st.info("No processed images available for download")
    
    with col3:
        st.markdown("**📄 Analysis Report**")
        
        # Generate comprehensive analysis report
        report = generate_batch_analysis_report(df, processed_images)
        
        st.download_button(
            label="📋 Download Analysis Report",
            data=report,
            file_name=f"batch_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )


def create_organized_images_zip(processed_images, df):
    """Create an organized ZIP file with folders for different k-values and modes."""
    
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Create folder structure: mode/k_value/filename
            for key, image_array in processed_images.items():
                # Parse key: filename_mode_kX
                parts = key.rsplit('_', 2)  # Split from right to get mode and k
                if len(parts) >= 3:
                    filename_base = parts[0]
                    mode = parts[1]
                    k_part = parts[2]  # kX format
                    
                    # Create folder path
                    folder_path = f"{mode}/{k_part}/"
                    file_path = f"{folder_path}{filename_base}.png"
                    
                    # Convert to PIL Image
                    image_pil = Image.fromarray((image_array * 255).astype(np.uint8))
                    
                    # Save to buffer
                    img_buffer = io.BytesIO()
                    image_pil.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    
                    # Add to ZIP with organized structure
                    zip_file.writestr(file_path, img_buffer.getvalue())
            
            # Add summary file
            summary = generate_zip_summary(df)
            zip_file.writestr("README.txt", summary)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error creating organized ZIP file: {str(e)}")
        return None


def generate_zip_summary(df):
    """Generate a summary file for the ZIP archive."""
    
    summary = f"""
SVD Image Compression - Batch Processing Results
===============================================

Export Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

SUMMARY STATISTICS
==================
Total Images Processed: {df['filename'].nunique()}
Total Experiments: {len(df)}
K-Values Tested: {', '.join(map(str, sorted(df['k_value'].unique())))}
Processing Modes: {', '.join(df['mode'].unique())}

Average PSNR: {df['psnr'].mean():.2f} dB
Average SSIM: {df['ssim'].mean():.3f}
Average Compression Ratio: {df['compression_ratio'].mean():.1f}:1

FOLDER STRUCTURE
================
The images are organized as follows:
- RGB/k5/: RGB images compressed with k=5
- RGB/k10/: RGB images compressed with k=10
- Grayscale/k5/: Grayscale images compressed with k=5
- etc.

Each folder contains the compressed images for that specific
processing mode and k-value combination.

QUALITY INTERPRETATION
======================
PSNR (Peak Signal-to-Noise Ratio):
- > 30 dB: Excellent quality
- 25-30 dB: Good quality
- 20-25 dB: Fair quality
- < 20 dB: Poor quality

SSIM (Structural Similarity Index):
- > 0.9: Excellent similarity
- 0.7-0.9: Good similarity
- 0.5-0.7: Fair similarity
- < 0.5: Poor similarity

Generated by SVD Image Compression Tool
"""
    
    return summary


def generate_batch_analysis_report(df, processed_images):
    """Generate a comprehensive batch analysis report."""
    
    enhanced_df = df.copy()
    enhanced_df['quality_score'] = enhanced_df.apply(
        lambda row: calculate_quality_score_batch(row['psnr'], row['ssim']), axis=1
    )
    
    report = f"""
SVD IMAGE COMPRESSION - BATCH ANALYSIS REPORT
============================================

ANALYSIS TIMESTAMP: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

EXECUTIVE SUMMARY
=================
This report presents the results of batch SVD image compression analysis
performed on {df['filename'].nunique()} images with {len(df['k_value'].unique())} different
compression levels (k-values).

DATASET OVERVIEW
================
Images Processed: {df['filename'].nunique()}
Total Experiments: {len(df)}
K-Values Tested: {', '.join(map(str, sorted(df['k_value'].unique())))}
Processing Modes: {', '.join(df['mode'].unique())}

QUALITY METRICS SUMMARY
=======================
Average PSNR: {df['psnr'].mean():.2f} dB (Range: {df['psnr'].min():.2f} - {df['psnr'].max():.2f})
Average SSIM: {df['ssim'].mean():.3f} (Range: {df['ssim'].min():.3f} - {df['ssim'].max():.3f})
Average MSE: {df['mse'].mean():.4f} (Range: {df['mse'].min():.4f} - {df['mse'].max():.4f})
Average Compression Ratio: {df['compression_ratio'].mean():.1f}:1

PERFORMANCE ANALYSIS
====================
Best Overall Quality Score: {enhanced_df['quality_score'].max():.1f}/100
Worst Overall Quality Score: {enhanced_df['quality_score'].min():.1f}/100

Best Performing Image: {df.loc[enhanced_df['quality_score'].idxmax(), 'filename']}
- PSNR: {df.loc[enhanced_df['quality_score'].idxmax(), 'psnr']:.2f} dB
- SSIM: {df.loc[enhanced_df['quality_score'].idxmax(), 'ssim']:.3f}
- K-Value: {df.loc[enhanced_df['quality_score'].idxmax(), 'k_value']}

DETAILED RESULTS BY IMAGE
=========================
"""
    
    # Add per-image analysis
    for filename in df['filename'].unique():
        file_data = df[df['filename'] == filename]
        best_result = file_data.loc[file_data['psnr'].idxmax()]
        
        report += f"""
{filename}:
- Best PSNR: {best_result['psnr']:.2f} dB (k={best_result['k_value']})
- Best SSIM: {file_data['ssim'].max():.3f} (k={file_data.loc[file_data['ssim'].idxmax(), 'k_value']})
- File Size: {file_data['file_size_kb'].iloc[0]:.1f} KB
"""
    
    report += f"""

RECOMMENDATIONS
===============
"""
    
    # Add recommendations
    recommendations = generate_optimization_recommendations(df)
    for category, recs in recommendations.items():
        report += f"\n{category}:\n"
        for rec in recs:
            report += f"- {rec}\n"
    
    report += f"""

TECHNICAL DETAILS
=================
SVD Compression Method:
- Singular Value Decomposition factorizes image matrix A = U * Σ * V^T
- Compression achieved by keeping only top k singular values
- Quality vs compression trade-off controlled by k parameter

Processing Parameters:
- Standard image size: 256×256 pixels
- Pixel value normalization: [0, 1] range
- Color space: RGB and/or Grayscale as specified

GENERATED BY
============
SVD Image Compression Tool - Batch Analysis Module
Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    return report