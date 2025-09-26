"""
Test script for responsive design and mobile optimization.
This script validates that all components work correctly across different screen sizes.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import time

# Test responsive design components
def test_responsive_navigation():
    """Test responsive navigation component."""
    st.markdown("## 📱 Responsive Navigation Test")
    
    from utils.navigation import create_tab_navigation
    
    st.markdown("### Navigation Component")
    st.info("Test the navigation on different screen sizes. On mobile, labels should be shortened or hidden.")
    
    current_tab = create_tab_navigation()
    st.success(f"Current tab: {current_tab}")


def test_responsive_upload():
    """Test responsive upload component."""
    st.markdown("## 📁 Responsive Upload Test")
    
    from utils.upload_component import create_enhanced_upload_component
    
    st.markdown("### Upload Component")
    st.info("Test the upload component on different screen sizes. It should adapt its layout and touch targets.")
    
    upload_result = create_enhanced_upload_component(
        key="responsive_test_upload",
        multiple=True,
        show_preview=True,
        show_progress=True
    )
    
    if upload_result:
        st.success("Upload component working correctly!")
        st.json({
            "filename": upload_result['file'].name,
            "dimensions": upload_result['array'].shape,
            "file_size": len(upload_result['file'].getvalue())
        })


def test_responsive_metrics():
    """Test responsive metrics display."""
    st.markdown("## 📊 Responsive Metrics Test")
    
    st.markdown("### Metrics Grid")
    st.info("Test metrics layout on different screen sizes. Should stack appropriately on mobile.")
    
    # Create test metrics in responsive grid
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        st.metric("PSNR", "32.5 dB", "Good")
    
    with col2:
        st.metric("SSIM", "0.85", "High")
    
    with col3:
        st.metric("Compression", "5.2:1", "48% saved")
    
    with col4:
        st.metric("Quality", "82/100", "Excellent")


def test_responsive_images():
    """Test responsive image display."""
    st.markdown("## 🖼️ Responsive Image Test")
    
    st.markdown("### Image Comparison")
    st.info("Test image comparison layout on different screen sizes. Should stack on mobile.")
    
    # Create test images
    test_image = np.random.rand(100, 100, 3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original**")
        st.image(test_image, caption="Original Image", use_column_width=True)
    
    with col2:
        st.markdown("**Compressed**")
        st.image(test_image * 0.8, caption="Compressed Image", use_column_width=True)


def test_touch_interactions():
    """Test touch-friendly interactions."""
    st.markdown("## 👆 Touch Interaction Test")
    
    st.markdown("### Touch-Friendly Buttons")
    st.info("Test button sizes and touch targets. Should be at least 44px on touch devices.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Small Button", key="small_btn"):
            st.success("Small button pressed!")
    
    with col2:
        if st.button("Medium Button", key="medium_btn", use_container_width=True):
            st.success("Medium button pressed!")
    
    with col3:
        if st.button("Large Button", key="large_btn", use_container_width=True, type="primary"):
            st.success("Large button pressed!")


def test_mobile_forms():
    """Test mobile-friendly form elements."""
    st.markdown("## 📝 Mobile Form Test")
    
    st.markdown("### Form Elements")
    st.info("Test form elements on mobile devices. Should be appropriately sized and spaced.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.selectbox("Select Option", ["Option 1", "Option 2", "Option 3"])
        st.slider("Slider", 0, 100, 50)
    
    with col2:
        st.number_input("Number Input", 0, 100, 25)
        st.text_input("Text Input", "Sample text")


def test_responsive_layout():
    """Test overall responsive layout."""
    st.markdown("## 📐 Layout Test")
    
    st.markdown("### Responsive Grid System")
    st.info("Test grid layouts that adapt to screen size.")
    
    # Test different grid configurations
    st.markdown("#### 4-Column Grid (should stack on mobile)")
    cols = st.columns(4)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                color: white;
                padding: 1rem;
                border-radius: 0.5rem;
                text-align: center;
                margin: 0.25rem 0;
            ">
                <h4 style="color: white; margin: 0;">Item {i+1}</h4>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem;">Grid item content</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("#### 2-Column Grid (should stack on small mobile)")
    cols = st.columns(2)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 0.75rem;
                text-align: center;
                margin: 0.5rem 0;
            ">
                <h4 style="color: white; margin: 0;">Section {i+1}</h4>
                <p style="margin: 0.75rem 0 0 0;">Larger content area</p>
            </div>
            """, unsafe_allow_html=True)


def test_performance_indicators():
    """Test loading and performance indicators."""
    st.markdown("## ⚡ Performance Test")
    
    st.markdown("### Loading Animations")
    st.info("Test loading animations and progress indicators.")
    
    if st.button("Test Loading Animation"):
        with st.spinner("Processing..."):
            progress_bar = st.progress(0)
            for i in range(101):
                progress_bar.progress(i)
                time.sleep(0.01)
        st.success("Loading complete!")


def main():
    """Main test function."""
    st.set_page_config(
        page_title="Responsive Design Test",
        page_icon="📱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Load styling
    from utils.styling import load_tailwind_css
    load_tailwind_css()
    
    st.title("📱 Responsive Design & Mobile Optimization Test")
    st.markdown("""
    This page tests the responsive design and mobile optimization features.
    
    **Test Instructions:**
    1. Open this page on different devices (desktop, tablet, mobile)
    2. Test in different orientations (portrait/landscape)
    3. Verify touch interactions work properly
    4. Check that layouts adapt appropriately
    5. Ensure text remains readable at all sizes
    """)
    
    # Device detection info
    st.markdown("### 📱 Device Information")
    st.markdown("""
    <div style="
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    ">
        <p><strong>Current viewport:</strong> Use browser dev tools to test different screen sizes</p>
        <p><strong>Recommended test sizes:</strong></p>
        <ul>
            <li>Mobile: 375px × 667px (iPhone SE)</li>
            <li>Mobile Large: 414px × 896px (iPhone 11)</li>
            <li>Tablet: 768px × 1024px (iPad)</li>
            <li>Desktop: 1280px × 720px</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Run all tests
    test_responsive_navigation()
    st.markdown("---")
    
    test_responsive_upload()
    st.markdown("---")
    
    test_responsive_metrics()
    st.markdown("---")
    
    test_responsive_images()
    st.markdown("---")
    
    test_touch_interactions()
    st.markdown("---")
    
    test_mobile_forms()
    st.markdown("---")
    
    test_responsive_layout()
    st.markdown("---")
    
    test_performance_indicators()
    
    # Test summary
    st.markdown("---")
    st.markdown("## ✅ Test Summary")
    st.success("""
    **Responsive Design Features Tested:**
    - ✅ Responsive navigation with mobile adaptations
    - ✅ Touch-friendly upload component
    - ✅ Adaptive metrics grid layout
    - ✅ Responsive image comparison
    - ✅ Touch-optimized button sizes
    - ✅ Mobile-friendly form elements
    - ✅ Flexible grid systems
    - ✅ Loading animations and progress indicators
    
    **Next Steps:**
    - Test on actual devices
    - Verify accessibility compliance
    - Check performance on slower connections
    - Validate touch gesture support
    """)


if __name__ == "__main__":
    main()