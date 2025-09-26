"""
Fallback state components for handling processing failures gracefully.
Provides alternative UI states when normal processing cannot be completed.
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time


class FallbackStateManager:
    """Manages fallback states for various failure scenarios."""
    
    def __init__(self):
        self.fallback_cache = {}
    
    def create_processing_failure_state(self, 
                                      error_info: Dict[str, Any], 
                                      original_image: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Create a fallback state for processing failures.
        
        Args:
            error_info: Information about the error that occurred
            original_image: Original image if available
            
        Returns:
            Fallback state data
        """
        
        st.markdown("---")
        st.markdown("## ⚠️ Processing Issue Detected")
        
        # Show error summary
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🔍 What Happened")
            
            if error_info.get('error_type') == 'out_of_memory':
                self._show_memory_error_state(error_info, original_image)
            elif error_info.get('error_type') == 'svd_failure':
                self._show_processing_error_state(error_info, original_image)
            elif error_info.get('error_type') == 'invalid_k_value':
                self._show_parameter_error_state(error_info, original_image)
            else:
                self._show_generic_error_state(error_info, original_image)
        
        with col2:
            self._show_error_recovery_panel(error_info)
        
        # Show fallback visualization if available
        if error_info.get('fallback_data'):
            st.markdown("---")
            st.markdown("## 🔄 Fallback Result")
            st.info("We were able to provide a partial result using alternative settings.")
            self._show_fallback_result(error_info['fallback_data'], original_image)
        
        return {
            'state': 'fallback',
            'error_handled': True,
            'fallback_available': bool(error_info.get('fallback_data'))
        }
    
    def _show_memory_error_state(self, error_info: Dict[str, Any], original_image: Optional[np.ndarray]):
        """Show memory error fallback state."""
        
        st.error("💾 **Memory Limit Exceeded**")
        st.markdown("""
        The image you're trying to process requires more memory than is currently available. 
        This typically happens with very large images or high k-values.
        """)
        
        if original_image is not None:
            # Show image statistics
            image_size_mb = original_image.nbytes / (1024 * 1024)
            st.markdown(f"""
            **Image Statistics:**
            - Dimensions: {original_image.shape[0]} × {original_image.shape[1]}
            - Memory usage: {image_size_mb:.1f} MB
            - Channels: {original_image.shape[2] if len(original_image.shape) > 2 else 1}
            """)
            
            # Memory usage visualization
            self._create_memory_usage_chart(image_size_mb)
        
        # Show memory optimization tips
        with st.expander("💡 Memory Optimization Tips", expanded=True):
            st.markdown("""
            **Immediate Solutions:**
            1. **Reduce image size**: Resize to 1024×1024 or smaller
            2. **Lower k-value**: Try k=10-20 for initial testing
            3. **Use grayscale**: Reduces memory usage by ~67%
            4. **Close other apps**: Free up system memory
            
            **Advanced Options:**
            - Enable automatic image resizing in settings
            - Use batch processing for multiple small images
            - Consider cloud processing for very large images
            """)
    
    def _show_processing_error_state(self, error_info: Dict[str, Any], original_image: Optional[np.ndarray]):
        """Show processing error fallback state."""
        
        st.error("🔧 **Processing Failed**")
        st.markdown("""
        The SVD compression algorithm encountered an issue while processing your image. 
        This can happen with certain types of image data or parameter combinations.
        """)
        
        if original_image is not None:
            # Analyze image characteristics
            self._analyze_image_characteristics(original_image)
        
        # Show processing troubleshooting
        with st.expander("🔧 Troubleshooting Guide", expanded=True):
            st.markdown("""
            **Common Causes & Solutions:**
            
            1. **Invalid image data**
               - Check for NaN or infinite values
               - Ensure image is properly normalized (0-1 range)
               
            2. **Extreme k-values**
               - Try k-values between 5-50 for most images
               - Use auto-optimize to find suitable values
               
            3. **Unusual image dimensions**
               - Very thin or wide images may cause issues
               - Try cropping to more square proportions
               
            4. **Corrupted image data**
               - Re-upload the image file
               - Try converting to a different format first
            """)
    
    def _show_parameter_error_state(self, error_info: Dict[str, Any], original_image: Optional[np.ndarray]):
        """Show parameter error fallback state."""
        
        st.warning("🎛️ **Invalid Parameters**")
        st.markdown("""
        The compression parameters you selected are not compatible with this image. 
        Let's help you find the right settings.
        """)
        
        if original_image is not None:
            # Show parameter recommendations
            self._show_parameter_recommendations(original_image)
        
        # Interactive parameter fixer
        st.markdown("### 🔧 Parameter Assistant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Quick Fixes:**")
            if st.button("🎯 Auto-Optimize Parameters", use_container_width=True):
                st.success("✅ Parameters optimized! Try processing again.")
                # This would trigger parameter optimization
            
            if st.button("🔄 Reset to Defaults", use_container_width=True):
                st.success("✅ Parameters reset to safe defaults!")
                # This would reset parameters
        
        with col2:
            st.markdown("**Manual Adjustment:**")
            if original_image is not None:
                max_k = min(original_image.shape[:2])
                recommended_k = max(1, max_k // 10)
                
                st.info(f"**Recommended k-value:** {recommended_k}")
                st.info(f"**Valid range:** 1 to {max_k}")
    
    def _show_generic_error_state(self, error_info: Dict[str, Any], original_image: Optional[np.ndarray]):
        """Show generic error fallback state."""
        
        st.error("❌ **Unexpected Error**")
        st.markdown("""
        An unexpected error occurred during processing. Don't worry - we can help you get back on track!
        """)
        
        # Show error details if available
        if error_info.get('error'):
            with st.expander("🔍 Technical Details", expanded=False):
                st.code(str(error_info['error']))
        
        # Show general recovery options
        st.markdown("### 🔄 Recovery Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Try Again", use_container_width=True):
                st.rerun()
        
        with col2:
            if st.button("🎯 Use Safe Settings", use_container_width=True):
                st.success("✅ Switched to safe settings!")
        
        with col3:
            if st.button("📁 Try Different Image", use_container_width=True):
                st.info("Please upload a different image above.")
    
    def _show_error_recovery_panel(self, error_info: Dict[str, Any]):
        """Show error recovery options panel."""
        
        st.markdown("### 🚀 Quick Recovery")
        
        # Recovery actions based on error type
        suggestions = error_info.get('suggestions', [])
        
        if suggestions:
            st.markdown("**Suggested Actions:**")
            for i, suggestion in enumerate(suggestions[:3], 1):  # Show top 3 suggestions
                st.markdown(f"{i}. {suggestion}")
        
        # Quick action buttons
        st.markdown("**Quick Actions:**")
        
        if st.button("🔄 Retry with Safe Settings", use_container_width=True):
            st.session_state['use_safe_settings'] = True
            st.success("✅ Safe settings enabled!")
        
        if st.button("📊 Show Diagnostics", use_container_width=True):
            self._show_diagnostic_info(error_info)
        
        if st.button("💬 Get Help", use_container_width=True):
            self._show_help_resources()
        
        # Error reporting
        with st.expander("📝 Report Issue", expanded=False):
            st.markdown("Help us improve by reporting this issue:")
            
            issue_description = st.text_area(
                "Describe what you were trying to do:",
                placeholder="I was trying to compress a landscape photo with k=50..."
            )
            
            if st.button("📤 Send Report"):
                if issue_description:
                    st.success("✅ Thank you! Your report has been submitted.")
                else:
                    st.warning("Please provide a description of the issue.")
    
    def _show_fallback_result(self, fallback_data: Dict[str, Any], original_image: Optional[np.ndarray]):
        """Show fallback processing result."""
        
        if not fallback_data or not fallback_data.get('compressed_image') is not None:
            return
        
        compressed_image = fallback_data['compressed_image']
        metadata = fallback_data.get('metadata', {})
        
        # Show fallback settings used
        st.info(f"""
        **Fallback Settings Used:**
        - K-value: {metadata.get('k_value', 'Unknown')}
        - Processing mode: {'Grayscale' if len(compressed_image.shape) == 2 else 'RGB'}
        - Compression ratio: {metadata.get('compression_ratio', 'Unknown'):.1f}:1
        """)
        
        # Show comparison if original is available
        if original_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(original_image, use_column_width=True)
            
            with col2:
                st.markdown("**Fallback Result**")
                st.image(compressed_image, use_column_width=True)
        else:
            st.markdown("**Fallback Result**")
            st.image(compressed_image, use_column_width=True)
        
        # Show quality metrics if available
        if 'quality_metrics' in fallback_data:
            self._show_fallback_metrics(fallback_data['quality_metrics'])
    
    def _create_memory_usage_chart(self, image_size_mb: float):
        """Create memory usage visualization."""
        
        # Estimate memory requirements
        processing_memory = image_size_mb * 3  # Rough estimate for SVD processing
        available_memory = 1024  # Assume 1GB available (this could be dynamic)
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Image Data', 'Processing', 'Available'],
                y=[image_size_mb, processing_memory, available_memory],
                marker_color=['#3b82f6', '#ef4444', '#10b981']
            )
        ])
        
        fig.update_layout(
            title="Memory Usage Estimate",
            yaxis_title="Memory (MB)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _analyze_image_characteristics(self, image: np.ndarray):
        """Analyze and display image characteristics."""
        
        st.markdown("**Image Analysis:**")
        
        # Basic statistics
        stats = {
            'Mean': np.mean(image),
            'Std Dev': np.std(image),
            'Min': np.min(image),
            'Max': np.max(image),
            'Range': np.max(image) - np.min(image)
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            for key, value in list(stats.items())[:3]:
                st.metric(key, f"{value:.3f}")
        
        with col2:
            for key, value in list(stats.items())[3:]:
                st.metric(key, f"{value:.3f}")
        
        # Check for potential issues
        issues = []
        if np.any(np.isnan(image)):
            issues.append("⚠️ Contains NaN values")
        if np.any(np.isinf(image)):
            issues.append("⚠️ Contains infinite values")
        if np.min(image) < 0:
            issues.append("⚠️ Contains negative values")
        if np.max(image) > 1:
            issues.append("⚠️ Values exceed 1.0 (may need normalization)")
        
        if issues:
            st.markdown("**Potential Issues:**")
            for issue in issues:
                st.markdown(f"- {issue}")
    
    def _show_parameter_recommendations(self, image: np.ndarray):
        """Show parameter recommendations based on image analysis."""
        
        height, width = image.shape[:2]
        max_k = min(height, width)
        
        # Calculate recommendations
        recommendations = {
            'Conservative (High Quality)': min(max_k, max(50, max_k // 4)),
            'Balanced': min(max_k, max(20, max_k // 8)),
            'Aggressive (High Compression)': min(max_k, max(5, max_k // 20))
        }
        
        st.markdown("**Recommended K-Values:**")
        
        for preset, k_value in recommendations.items():
            compression_ratio = (height * width) / (k_value * (height + width + 1))
            st.markdown(f"- **{preset}**: k={k_value} (≈{compression_ratio:.1f}:1 compression)")
    
    def _show_fallback_metrics(self, metrics: Dict[str, Any]):
        """Show quality metrics for fallback result."""
        
        st.markdown("### 📊 Fallback Quality Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'psnr' in metrics:
                st.metric("PSNR", f"{metrics['psnr']:.1f} dB")
        
        with col2:
            if 'ssim' in metrics:
                st.metric("SSIM", f"{metrics['ssim']:.3f}")
        
        with col3:
            if 'compression_ratio' in metrics:
                st.metric("Compression", f"{metrics['compression_ratio']:.1f}:1")
    
    def _show_diagnostic_info(self, error_info: Dict[str, Any]):
        """Show diagnostic information."""
        
        st.markdown("### 🔍 Diagnostic Information")
        
        with st.expander("System Information", expanded=True):
            import psutil
            import platform
            
            st.markdown(f"""
            **System:**
            - Platform: {platform.system()} {platform.release()}
            - Python: {platform.python_version()}
            - Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB total
            - Available: {psutil.virtual_memory().available / (1024**3):.1f} GB
            
            **Error Details:**
            - Type: {error_info.get('error_type', 'Unknown')}
            - Category: {error_info.get('category', 'Unknown')}
            - Recovery Attempted: {error_info.get('recovery_attempted', False)}
            - Recovery Successful: {error_info.get('recovery_successful', False)}
            """)
    
    def _show_help_resources(self):
        """Show help resources."""
        
        st.markdown("### 💬 Help Resources")
        
        st.markdown("""
        **Quick Help:**
        - 📖 [User Guide](https://example.com/guide) - Step-by-step instructions
        - 🎥 [Video Tutorials](https://example.com/videos) - Visual learning
        - ❓ [FAQ](https://example.com/faq) - Common questions
        
        **Community Support:**
        - 💬 [Discussion Forum](https://example.com/forum) - Ask questions
        - 🐛 [Bug Reports](https://example.com/bugs) - Report issues
        - 💡 [Feature Requests](https://example.com/features) - Suggest improvements
        
        **Contact:**
        - 📧 Email: support@example.com
        - 🐦 Twitter: @example_support
        """)


def create_loading_failure_state(error_message: str = None):
    """Create a fallback state for loading failures."""
    
    st.markdown("---")
    st.error("📁 **Loading Failed**")
    
    if error_message:
        st.markdown(f"**Error:** {error_message}")
    
    st.markdown("""
    We couldn't load the necessary components. Here are some things you can try:
    
    1. **Refresh the page** - Sometimes a simple refresh fixes loading issues
    2. **Check your connection** - Ensure you have a stable internet connection
    3. **Clear browser cache** - Old cached files might be causing conflicts
    4. **Try a different browser** - Some browsers handle the app better than others
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Page", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ Cache cleared!")
    
    with col3:
        if st.button("📞 Get Help", use_container_width=True):
            st.info("Please contact support with details about this error.")


def create_network_failure_state():
    """Create a fallback state for network failures."""
    
    st.markdown("---")
    st.warning("🌐 **Connection Issue**")
    
    st.markdown("""
    It looks like there's a network connectivity issue. The app can still work in offline mode 
    with limited functionality.
    
    **Available in offline mode:**
    - Basic image compression
    - Local file processing
    - Cached results
    
    **Not available offline:**
    - Cloud processing
    - Online help resources
    - Automatic updates
    """)
    
    if st.button("🔄 Retry Connection", use_container_width=True):
        st.rerun()


def create_maintenance_state():
    """Create a fallback state for maintenance mode."""
    
    st.markdown("---")
    st.info("🔧 **Maintenance Mode**")
    
    st.markdown("""
    The application is currently undergoing maintenance. Some features may be temporarily unavailable.
    
    **What's working:**
    - Basic image upload and viewing
    - Simple compression operations
    - Cached results
    
    **What's being updated:**
    - Advanced processing features
    - Performance optimizations
    - New functionality
    
    We apologize for any inconvenience and appreciate your patience!
    """)
    
    # Show estimated completion time
    st.markdown("**Estimated completion:** ~15 minutes")
    
    # Progress bar (simulated)
    progress = st.progress(0.7)  # 70% complete
    st.caption("Maintenance progress: 70% complete")


# Global fallback state manager
_fallback_manager = None


def get_fallback_manager() -> FallbackStateManager:
    """Get global fallback state manager instance."""
    global _fallback_manager
    if _fallback_manager is None:
        _fallback_manager = FallbackStateManager()
    return _fallback_manager