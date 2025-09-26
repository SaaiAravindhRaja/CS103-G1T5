"""
Enhanced image upload component with drag-and-drop, validation, progress indicators,
animations, and accessibility features.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import time
from typing import List, Dict, Optional, Tuple, Any
from .animations import animation_manager, show_loading_state, create_notification
from .accessibility import create_accessible_image, announce_to_screen_reader


class ImageUploadComponent:
    """Enhanced image upload component with modern UI, animations, and accessibility features."""
    
    def __init__(self):
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.supported_formats = ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
        self.max_files = 10
        self.animation_manager = animation_manager
        
    def render(self, 
               key: str = "upload", 
               multiple: bool = True,
               show_preview: bool = True,
               show_progress: bool = True) -> Optional[Dict[str, Any]]:
        """
        Render the enhanced upload component.
        
        Args:
            key: Unique key for the component
            multiple: Allow multiple file uploads
            show_preview: Show image preview thumbnails
            show_progress: Show upload progress indicators
            
        Returns:
            Dictionary containing upload results or None
        """
        
        # Initialize session state for this component
        upload_key = f"upload_state_{key}"
        if upload_key not in st.session_state:
            st.session_state[upload_key] = {
                'files': [],
                'selected_index': 0,
                'upload_progress': {},
                'validation_errors': {},
                'processing_status': 'idle'
            }
        
        upload_state = st.session_state[upload_key]
        
        # Enhanced upload zone with animations and accessibility
        self._render_enhanced_upload_zone(key, multiple, upload_state)
        
        # Render drag-and-drop zone
        self._render_drag_drop_zone(key)
        
        # File uploader with enhanced styling
        uploaded_files = st.file_uploader(
            "Choose image files" if multiple else "Choose an image file",
            type=self.supported_formats,
            accept_multiple_files=multiple,
            help=f"Supported formats: {', '.join(self.supported_formats).upper()}. Max size: {self.max_file_size // (1024*1024)}MB per file.",
            key=f"file_uploader_{key}"
        )
        
        if uploaded_files:
            # Handle single file vs multiple files
            files_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
            
            # Validate and process files
            validation_results = self._validate_files(files_list)
            
            if validation_results['valid_files']:
                # Show upload progress if enabled
                if show_progress:
                    self._show_upload_progress(validation_results['valid_files'], key)
                
                # Process valid files
                processed_files = self._process_files(validation_results['valid_files'], key)
                upload_state['files'] = processed_files
                
                # Show file management interface
                if len(processed_files) > 1:
                    selected_file = self._render_file_selector(processed_files, key)
                    upload_state['selected_index'] = selected_file
                
                # Show preview if enabled
                if show_preview and processed_files:
                    self._render_image_preview(processed_files, upload_state['selected_index'])
                
                # Show validation errors for invalid files
                if validation_results['invalid_files']:
                    self._show_validation_errors(validation_results['invalid_files'])
                
                # Return the selected file data
                if processed_files:
                    selected_file_data = processed_files[upload_state['selected_index']]
                    return {
                        'file': selected_file_data['file'],
                        'array': selected_file_data['array'],
                        'pil': selected_file_data['pil'],
                        'metadata': selected_file_data['metadata'],
                        'all_files': processed_files,
                        'selected_index': upload_state['selected_index']
                    }
            else:
                # Show validation errors
                self._show_validation_errors(validation_results['invalid_files'])
        
        return None
    
    def _render_drag_drop_zone(self, key: str):
        """Render the drag-and-drop upload zone with visual feedback."""
        
        st.markdown(
            f"""
            <div class="enhanced-upload-zone" id="upload_zone_{key}">
                <div class="upload-zone-content">
                    <div class="upload-icon">📁</div>
                    <h3 class="upload-title">Drag & Drop Images Here</h3>
                    <p class="upload-subtitle">or click the upload button below</p>
                    <div class="upload-features">
                        <span class="feature-item">✓ Multiple files</span>
                        <span class="feature-item">✓ Instant preview</span>
                        <span class="feature-item">✓ Auto validation</span>
                    </div>
                    <div class="upload-formats">
                        Supports: {', '.join(self.supported_formats).upper()}
                    </div>
                </div>
                <div class="upload-overlay" id="overlay_{key}">
                    <div class="overlay-content">
                        <div class="overlay-icon">⬇️</div>
                        <div class="overlay-text">Drop files here</div>
                    </div>
                </div>
            </div>
            
            <style>
            .enhanced-upload-zone {{
                position: relative;
                border: 3px dashed #3b82f6;
                border-radius: 20px;
                padding: 3rem 2rem;
                text-align: center;
                background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                margin: 1.5rem 0;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                cursor: pointer;
                animation: fadeIn 0.5s ease-in-out;
                min-height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            
            .enhanced-upload-zone:hover {{
                border-color: #2563eb;
                background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
                transform: translateY(-4px) scale(1.01);
                box-shadow: 0 12px 35px rgba(59, 130, 246, 0.2);
            }}
            
            .upload-zone-content {{
                position: relative;
                z-index: 2;
                width: 100%;
            }}
            
            .upload-icon {{
                font-size: 4rem;
                margin-bottom: 1rem;
                opacity: 0.8;
            }}
            
            .upload-title {{
                color: #1f2937;
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }}
            
            .upload-subtitle {{
                color: #6b7280;
                font-size: 1rem;
                margin-bottom: 1.5rem;
            }}
            
            .upload-features {{
                display: flex;
                justify-content: center;
                gap: 1.5rem;
                margin-bottom: 1rem;
                flex-wrap: wrap;
            }}
            
            .feature-item {{
                background: rgba(59, 130, 246, 0.1);
                color: #2563eb;
                padding: 0.25rem 0.75rem;
                border-radius: 15px;
                font-size: 0.875rem;
                font-weight: 500;
            }}
            
            .upload-formats {{
                color: #9ca3af;
                font-size: 0.875rem;
                font-weight: 500;
            }}
            
            .upload-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(59, 130, 246, 0.95);
                border-radius: 17px;
                display: flex;
                align-items: center;
                justify-content: center;
                opacity: 0;
                transform: scale(0.95);
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                pointer-events: none;
            }}
            
            .upload-overlay.active {{
                opacity: 1;
                transform: scale(1);
            }}
            
            .overlay-content {{
                text-align: center;
                color: white;
            }}
            
            .overlay-icon {{
                font-size: 3rem;
                margin-bottom: 0.5rem;
                animation: bounce 1s infinite;
            }}
            
            .overlay-text {{
                font-size: 1.25rem;
                font-weight: 600;
            }}
            
            @keyframes bounce {{
                0%, 20%, 50%, 80%, 100% {{
                    transform: translateY(0);
                }}
                40% {{
                    transform: translateY(-10px);
                }}
                60% {{
                    transform: translateY(-5px);
                }}
            }}
            
            /* Tablet Styles */
            @media (min-width: 768px) and (max-width: 1024px) {{
                .enhanced-upload-zone {{
                    padding: 2.5rem 1.5rem;
                    margin: 1rem 0;
                }}
                
                .upload-icon {{
                    font-size: 3.5rem;
                }}
                
                .upload-title {{
                    font-size: 1.375rem;
                }}
                
                .upload-subtitle {{
                    font-size: 0.9375rem;
                }}
                
                .upload-features {{
                    gap: 1rem;
                }}
            }}
            
            /* Mobile Styles */
            @media (max-width: 768px) {{
                .enhanced-upload-zone {{
                    padding: 2rem 1rem;
                    margin: 1rem 0.5rem;
                    min-height: 160px;
                }}
                
                .upload-icon {{
                    font-size: 3rem;
                }}
                
                .upload-title {{
                    font-size: 1.25rem;
                    line-height: 1.3;
                }}
                
                .upload-subtitle {{
                    font-size: 0.875rem;
                    margin-bottom: 1rem;
                }}
                
                .upload-features {{
                    flex-direction: column;
                    gap: 0.5rem;
                    align-items: center;
                }}
                
                .feature-item {{
                    font-size: 0.8125rem;
                    padding: 0.1875rem 0.625rem;
                }}
                
                .upload-formats {{
                    font-size: 0.8125rem;
                }}
            }}
            
            /* Small Mobile Styles */
            @media (max-width: 480px) {{
                .enhanced-upload-zone {{
                    padding: 1.5rem 0.75rem;
                    margin: 0.75rem 0.25rem;
                    min-height: 140px;
                    border-radius: 15px;
                }}
                
                .upload-icon {{
                    font-size: 2.5rem;
                    margin-bottom: 0.75rem;
                }}
                
                .upload-title {{
                    font-size: 1.125rem;
                    margin-bottom: 0.375rem;
                }}
                
                .upload-subtitle {{
                    font-size: 0.8125rem;
                    margin-bottom: 0.75rem;
                }}
                
                .upload-features {{
                    gap: 0.375rem;
                }}
                
                .feature-item {{
                    font-size: 0.75rem;
                    padding: 0.125rem 0.5rem;
                }}
                
                .upload-formats {{
                    font-size: 0.75rem;
                }}
                
                .overlay-icon {{
                    font-size: 2.5rem;
                }}
                
                .overlay-text {{
                    font-size: 1.125rem;
                }}
            }}
            
            /* Touch device optimizations */
            @media (hover: none) and (pointer: coarse) {{
                .enhanced-upload-zone {{
                    min-height: 120px;
                }}
                
                .enhanced-upload-zone:hover {{
                    transform: none;
                }}
                
                .enhanced-upload-zone:active {{
                    transform: scale(0.98);
                    transition: transform 0.1s;
                }}
            }}
            
            /* Landscape mobile optimizations */
            @media (max-width: 768px) and (orientation: landscape) {{
                .enhanced-upload-zone {{
                    padding: 1.5rem 1rem;
                    min-height: 100px;
                }}
                
                .upload-icon {{
                    font-size: 2rem;
                    margin-bottom: 0.5rem;
                }}
                
                .upload-title {{
                    font-size: 1rem;
                    margin-bottom: 0.25rem;
                }}
                
                .upload-subtitle {{
                    font-size: 0.75rem;
                    margin-bottom: 0.5rem;
                }}
                
                .upload-features {{
                    flex-direction: row;
                    gap: 0.75rem;
                }}
            }}
            </style>
            
            <script>
            (function() {{
                const uploadZone = document.getElementById('upload_zone_{key}');
                const overlay = document.getElementById('overlay_{key}');
                
                if (uploadZone && overlay) {{
                    let dragCounter = 0;
                    
                    uploadZone.addEventListener('dragenter', function(e) {{
                        e.preventDefault();
                        dragCounter++;
                        overlay.classList.add('active');
                    }});
                    
                    uploadZone.addEventListener('dragleave', function(e) {{
                        e.preventDefault();
                        dragCounter--;
                        if (dragCounter === 0) {{
                            overlay.classList.remove('active');
                        }}
                    }});
                    
                    uploadZone.addEventListener('dragover', function(e) {{
                        e.preventDefault();
                    }});
                    
                    uploadZone.addEventListener('drop', function(e) {{
                        e.preventDefault();
                        dragCounter = 0;
                        overlay.classList.remove('active');
                        
                        // Note: Streamlit file_uploader handles the actual file processing
                        // This is just for visual feedback
                    }});
                }}
            }})();
            </script>
            """,
            unsafe_allow_html=True
        )
    
    def _validate_files(self, files: List) -> Dict[str, List]:
        """Validate uploaded files and return categorized results."""
        
        valid_files = []
        invalid_files = []
        
        for file in files:
            errors = []
            
            # Check file size
            if file.size > self.max_file_size:
                errors.append(f"File too large ({file.size / (1024*1024):.1f}MB > {self.max_file_size / (1024*1024)}MB)")
            
            # Check file format
            file_extension = file.name.split('.')[-1].lower()
            if file_extension not in self.supported_formats:
                errors.append(f"Unsupported format (.{file_extension})")
            
            # Try to open as image
            try:
                file.seek(0)  # Reset file pointer
                Image.open(file)
                file.seek(0)  # Reset again for later use
            except Exception as e:
                errors.append(f"Invalid image file: {str(e)}")
            
            if errors:
                invalid_files.append({
                    'file': file,
                    'errors': errors
                })
            else:
                valid_files.append(file)
        
        return {
            'valid_files': valid_files,
            'invalid_files': invalid_files
        }
    
    def _show_upload_progress(self, files: List, key: str):
        """Show upload progress indicators."""
        
        if len(files) == 1:
            # Single file progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(101):
                progress_bar.progress(i)
                if i < 30:
                    status_text.text("📤 Uploading file...")
                elif i < 60:
                    status_text.text("🔍 Validating image...")
                elif i < 90:
                    status_text.text("🖼️ Processing image...")
                else:
                    status_text.text("✅ Upload complete!")
                
                time.sleep(0.01)  # Simulate processing time
            
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
        
        else:
            # Multiple files progress
            st.markdown("### 📤 Upload Progress")
            
            overall_progress = st.progress(0)
            
            for idx, file in enumerate(files):
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(f"📁 {file.name}")
                
                with col2:
                    file_progress = st.progress(0)
                
                with col3:
                    status = st.empty()
                
                # Simulate file processing
                for i in range(101):
                    file_progress.progress(i)
                    if i < 50:
                        status.text("⏳")
                    else:
                        status.text("✅")
                    time.sleep(0.005)
                
                overall_progress.progress((idx + 1) / len(files))
            
            st.success(f"✅ Successfully uploaded {len(files)} files!")
            time.sleep(1)
            
            # Clean up progress indicators
            overall_progress.empty()
    
    def _process_files(self, files: List, key: str) -> List[Dict[str, Any]]:
        """Process valid files and extract image data."""
        
        processed_files = []
        
        for file in files:
            try:
                # Reset file pointer
                file.seek(0)
                
                # Open with PIL
                pil_image = Image.open(file)
                
                # Convert to RGB if necessary
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')
                
                # Convert to numpy array
                image_array = np.array(pil_image) / 255.0
                
                # Extract metadata
                metadata = {
                    'filename': file.name,
                    'size': file.size,
                    'dimensions': pil_image.size,
                    'mode': pil_image.mode,
                    'format': pil_image.format,
                    'file_size_mb': file.size / (1024 * 1024)
                }
                
                processed_files.append({
                    'file': file,
                    'pil': pil_image,
                    'array': image_array,
                    'metadata': metadata
                })
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
        
        return processed_files
    
    def _render_file_selector(self, files: List[Dict], key: str) -> int:
        """Render file selector for multiple uploads."""
        
        st.markdown("### 📁 Uploaded Files")
        
        # Create file selection interface
        file_options = []
        for idx, file_data in enumerate(files):
            metadata = file_data['metadata']
            size_mb = metadata['file_size_mb']
            dimensions = metadata['dimensions']
            file_options.append(
                f"{metadata['filename']} ({dimensions[0]}×{dimensions[1]}, {size_mb:.1f}MB)"
            )
        
        selected_index = st.selectbox(
            "Select file to work with:",
            range(len(file_options)),
            format_func=lambda x: file_options[x],
            key=f"file_selector_{key}"
        )
        
        # Show file grid for visual selection
        cols = st.columns(min(4, len(files)))
        
        for idx, file_data in enumerate(files):
            with cols[idx % len(cols)]:
                metadata = file_data['metadata']
                
                # Create clickable thumbnail
                if st.button(
                    f"📷 {metadata['filename'][:15]}...",
                    key=f"thumb_{key}_{idx}",
                    help=f"Select {metadata['filename']}",
                    use_container_width=True,
                    type="primary" if idx == selected_index else "secondary"
                ):
                    selected_index = idx
                    st.rerun()
                
                # Show thumbnail
                thumbnail = file_data['pil'].copy()
                thumbnail.thumbnail((150, 150))
                st.image(thumbnail, use_column_width=True)
                
                # Show file info
                st.caption(f"{metadata['dimensions'][0]}×{metadata['dimensions'][1]}")
                st.caption(f"{metadata['file_size_mb']:.1f}MB")
        
        return selected_index
    
    def _render_image_preview(self, files: List[Dict], selected_index: int):
        """Render image preview with detailed information."""
        
        if not files or selected_index >= len(files):
            return
        
        selected_file = files[selected_index]
        metadata = selected_file['metadata']
        
        st.markdown("### 🖼️ Image Preview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(
                selected_file['pil'],
                caption=f"Preview: {metadata['filename']}",
                use_column_width=True
            )
        
        with col2:
            st.markdown("#### 📋 Image Details")
            
            # Create info cards
            info_items = [
                ("📁 Filename", metadata['filename']),
                ("📐 Dimensions", f"{metadata['dimensions'][0]} × {metadata['dimensions'][1]} pixels"),
                ("💾 File Size", f"{metadata['file_size_mb']:.2f} MB"),
                ("🎨 Color Mode", metadata['mode']),
                ("📄 Format", metadata.get('format', 'Unknown')),
                ("📊 Aspect Ratio", f"{metadata['dimensions'][0]/metadata['dimensions'][1]:.2f}:1")
            ]
            
            for label, value in info_items:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                        border: 1px solid #e2e8f0;
                        border-radius: 8px;
                        padding: 0.75rem;
                        margin: 0.5rem 0;
                    ">
                        <div style="font-weight: 600; color: #374151; margin-bottom: 0.25rem;">
                            {label}
                        </div>
                        <div style="color: #6b7280; font-size: 0.9rem;">
                            {value}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Image statistics
            image_array = selected_file['array']
            st.markdown("#### 📈 Image Statistics")
            
            stats_items = [
                ("📊 Mean Intensity", f"{image_array.mean():.3f}"),
                ("📏 Std Deviation", f"{image_array.std():.3f}"),
                ("⬇️ Min Value", f"{image_array.min():.3f}"),
                ("⬆️ Max Value", f"{image_array.max():.3f}")
            ]
            
            for label, value in stats_items:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
                        border: 1px solid #10b981;
                        border-radius: 8px;
                        padding: 0.5rem;
                        margin: 0.25rem 0;
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    ">
                        <span style="font-weight: 500; color: #065f46;">{label}</span>
                        <span style="font-weight: 600; color: #047857; font-family: monospace;">{value}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    def _show_validation_errors(self, invalid_files: List[Dict]):
        """Show validation errors for invalid files."""
        
        if not invalid_files:
            return
        
        st.markdown("### ⚠️ Upload Issues")
        
        for file_info in invalid_files:
            file = file_info['file']
            errors = file_info['errors']
            
            with st.expander(f"❌ {file.name}", expanded=True):
                st.error("This file could not be uploaded due to the following issues:")
                
                for error in errors:
                    st.markdown(f"• {error}")
                
                # Show file info if available
                st.markdown("**File Information:**")
                st.markdown(f"• Size: {file.size / (1024*1024):.2f} MB")
                st.markdown(f"• Type: {file.type if hasattr(file, 'type') else 'Unknown'}")
                
                # Suggestions
                st.markdown("**Suggestions:**")
                if any("too large" in error for error in errors):
                    st.markdown("• Reduce image size or compress the file")
                if any("Unsupported format" in error for error in errors):
                    st.markdown(f"• Convert to supported format: {', '.join(self.supported_formats).upper()}")
                if any("Invalid image" in error for error in errors):
                    st.markdown("• Ensure the file is a valid image")


def create_enhanced_upload_component(key: str = "upload", 
                                   multiple: bool = True,
                                   show_preview: bool = True,
                                   show_progress: bool = True) -> Optional[Dict[str, Any]]:
    """
    Create and render an enhanced image upload component.
    
    Args:
        key: Unique key for the component
        multiple: Allow multiple file uploads
        show_preview: Show image preview thumbnails
        show_progress: Show upload progress indicators
        
    Returns:
        Dictionary containing upload results or None
    """
    
    upload_component = ImageUploadComponent()
    return upload_component.render(
        key=key,
        multiple=multiple,
        show_preview=show_preview,
        show_progress=show_progress
    )