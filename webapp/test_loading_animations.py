"""
Test script for the new loading animations and progress feedback components.
"""

import streamlit as st
import time
from utils.loading_animations import (
    LoadingAnimations, 
    ProgressManager, 
    loading_context,
    show_spinner,
    show_progress_ring,
    show_pulse,
    create_multi_step_progress
)


def main():
    """Test the loading animations and progress feedback."""
    
    st.set_page_config(
        page_title="Loading Animations Test",
        page_icon="⏳",
        layout="wide"
    )
    
    st.title("🎨 Loading Animations & Progress Feedback Test")
    st.markdown("Testing the enhanced loading animations and progress feedback components.")
    
    # Test different animation types
    st.header("🔄 Animation Types")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Test Spinner Animation", use_container_width=True):
            test_spinner_animation()
        
        if st.button("Test Pulse Animation", use_container_width=True):
            test_pulse_animation()
        
        if st.button("Test Wave Animation", use_container_width=True):
            test_wave_animation()
        
        if st.button("Test Dots Animation", use_container_width=True):
            test_dots_animation()
    
    with col2:
        if st.button("Test Progress Ring", use_container_width=True):
            test_progress_ring_animation()
        
        if st.button("Test Skeleton Animation", use_container_width=True):
            test_skeleton_animation()
        
        if st.button("Test Multi-Step Progress", use_container_width=True):
            test_multi_step_progress()
        
        if st.button("Test Context Manager", use_container_width=True):
            test_context_manager()
    
    # Test color schemes
    st.header("🎨 Color Schemes")
    
    color_cols = st.columns(4)
    colors = ['blue', 'green', 'purple', 'orange']
    
    for i, color in enumerate(colors):
        with color_cols[i]:
            if st.button(f"Test {color.title()}", key=f"color_{color}", use_container_width=True):
                test_color_scheme(color)
    
    # Test progress tracking
    st.header("📊 Progress Tracking")
    
    if st.button("Test Image Compression Simulation", use_container_width=True):
        test_image_compression_simulation()
    
    if st.button("Test Batch Processing Simulation", use_container_width=True):
        test_batch_processing_simulation()


def test_spinner_animation():
    """Test the spinner animation."""
    st.subheader("🔄 Spinner Animation Test")
    
    animations = LoadingAnimations()
    animation_controls = animations.show_loading_animation(
        animation_type='spinner',
        text="Testing spinner animation...",
        color_scheme='blue'
    )
    
    # Simulate progress
    for i in range(101):
        animation_controls['update_progress'](i / 100)
        time.sleep(0.02)
    
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()


def test_pulse_animation():
    """Test the pulse animation."""
    st.subheader("💓 Pulse Animation Test")
    
    animations = LoadingAnimations()
    animation_controls = animations.show_loading_animation(
        animation_type='pulse',
        text="Testing pulse animation...",
        color_scheme='green'
    )
    
    time.sleep(3)
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()


def test_wave_animation():
    """Test the wave animation."""
    st.subheader("🌊 Wave Animation Test")
    
    animations = LoadingAnimations()
    animation_controls = animations.show_loading_animation(
        animation_type='wave',
        text="Testing wave animation...",
        progress=0.0,
        color_scheme='blue'
    )
    
    # Simulate progress
    for i in range(101):
        animation_controls['update_progress'](i / 100)
        time.sleep(0.03)
    
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()


def test_dots_animation():
    """Test the dots animation."""
    st.subheader("⚫ Dots Animation Test")
    
    animations = LoadingAnimations()
    animation_controls = animations.show_loading_animation(
        animation_type='dots',
        text="Testing dots animation...",
        color_scheme='purple'
    )
    
    time.sleep(3)
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()


def test_progress_ring_animation():
    """Test the progress ring animation."""
    st.subheader("⭕ Progress Ring Test")
    
    animations = LoadingAnimations()
    animation_controls = animations.show_loading_animation(
        animation_type='progress_ring',
        text="Testing progress ring...",
        progress=0.0,
        color_scheme='orange'
    )
    
    # Simulate progress
    for i in range(101):
        animation_controls['update_progress'](i / 100)
        animation_controls['update_text'](f"Progress: {i}%")
        time.sleep(0.02)
    
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()


def test_skeleton_animation():
    """Test the skeleton animation."""
    st.subheader("💀 Skeleton Animation Test")
    
    animations = LoadingAnimations()
    animation_controls = animations.show_loading_animation(
        animation_type='skeleton',
        text="Loading content...",
        color_scheme='blue'
    )
    
    time.sleep(3)
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()


def test_multi_step_progress():
    """Test the multi-step progress manager."""
    st.subheader("📋 Multi-Step Progress Test")
    
    steps = [
        "Initializing process...",
        "Loading data...",
        "Processing information...",
        "Generating results...",
        "Finalizing output..."
    ]
    
    progress_controls = create_multi_step_progress(
        operation_id="test_operation",
        title="Test Operation",
        steps=steps,
        animation_type="progress_ring"
    )
    
    # Simulate step progression
    for i in range(len(steps)):
        time.sleep(1)
        if i < len(steps) - 1:
            progress_controls['next_step']()
        else:
            progress_controls['complete']()


def test_context_manager():
    """Test the loading context manager."""
    st.subheader("🔄 Context Manager Test")
    
    with loading_context("Processing with context manager...", "pulse", "green"):
        time.sleep(3)
    
    st.success("Context manager test completed!")


def test_color_scheme(color: str):
    """Test a specific color scheme."""
    st.subheader(f"🎨 {color.title()} Color Scheme Test")
    
    animations = LoadingAnimations()
    animation_controls = animations.show_loading_animation(
        animation_type='spinner',
        text=f"Testing {color} color scheme...",
        progress=0.0,
        color_scheme=color
    )
    
    # Simulate progress
    for i in range(101):
        animation_controls['update_progress'](i / 100)
        time.sleep(0.01)
    
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()


def test_image_compression_simulation():
    """Simulate image compression with progress tracking."""
    st.subheader("🖼️ Image Compression Simulation")
    
    compression_steps = [
        "Loading image data...",
        "Applying SVD decomposition...",
        "Reconstructing compressed image...",
        "Calculating quality metrics...",
        "Generating comparison..."
    ]
    
    progress_controls = create_multi_step_progress(
        operation_id="image_compression_sim",
        title="Image Compression",
        steps=compression_steps,
        animation_type="progress_ring"
    )
    
    # Simulate compression process
    for i, step in enumerate(compression_steps):
        # Simulate work within each step
        for j in range(20):
            progress_controls['update_progress'](j / 20)
            time.sleep(0.05)
        
        if i < len(compression_steps) - 1:
            progress_controls['next_step']()
        else:
            progress_controls['complete']()


def test_batch_processing_simulation():
    """Simulate batch processing with progress tracking."""
    st.subheader("📦 Batch Processing Simulation")
    
    batch_steps = [
        "Loading batch files...",
        "Processing images...",
        "Calculating metrics...",
        "Generating report..."
    ]
    
    progress_controls = create_multi_step_progress(
        operation_id="batch_processing_sim",
        title="Batch Processing",
        steps=batch_steps,
        animation_type="wave"
    )
    
    # Simulate batch processing
    num_files = 5
    
    for i, step in enumerate(batch_steps):
        if i == 1:  # Processing images step
            for file_idx in range(num_files):
                progress_controls['update_progress'](file_idx / num_files)
                progress_controls['update_text'](f"Processing file {file_idx + 1}/{num_files}...")
                time.sleep(0.3)
        else:
            # Other steps
            for j in range(10):
                progress_controls['update_progress'](j / 10)
                time.sleep(0.1)
        
        if i < len(batch_steps) - 1:
            progress_controls['next_step']()
        else:
            progress_controls['complete']()


if __name__ == "__main__":
    main()