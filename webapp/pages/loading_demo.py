"""
Demo page showcasing the new loading animations and progress feedback.
"""

import streamlit as st
import time
from utils.loading_animations import (
    LoadingAnimations,
    show_spinner,
    show_progress_ring,
    show_pulse,
    create_multi_step_progress,
    loading_context
)


def show():
    """Display the loading animations demo page."""
    
    st.markdown("# 🎨 Loading Animations Demo")
    st.markdown("Showcase of the new enhanced loading animations and progress feedback components.")
    
    # Quick demo section
    st.markdown("## 🚀 Quick Demo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Spinner Demo", use_container_width=True):
            demo_spinner()
    
    with col2:
        if st.button("Progress Ring Demo", use_container_width=True):
            demo_progress_ring()
    
    with col3:
        if st.button("Multi-Step Demo", use_container_width=True):
            demo_multi_step()
    
    # Animation types showcase
    st.markdown("## 🔄 Animation Types")
    
    animation_type = st.selectbox(
        "Choose animation type:",
        ["spinner", "pulse", "wave", "dots", "progress_ring", "skeleton"]
    )
    
    color_scheme = st.selectbox(
        "Choose color scheme:",
        ["blue", "green", "purple", "orange"]
    )
    
    if st.button("Show Animation", use_container_width=True):
        show_animation_demo(animation_type, color_scheme)
    
    # Real-world examples
    st.markdown("## 🌍 Real-World Examples")
    
    example_col1, example_col2 = st.columns(2)
    
    with example_col1:
        if st.button("Image Compression Simulation", use_container_width=True):
            simulate_image_compression()
    
    with example_col2:
        if st.button("Batch Processing Simulation", use_container_width=True):
            simulate_batch_processing()
    
    # Context manager demo
    st.markdown("## 🔧 Context Manager Demo")
    
    if st.button("Test Context Manager", use_container_width=True):
        demo_context_manager()


def demo_spinner():
    """Demo the spinner animation."""
    st.markdown("### 🔄 Spinner Animation")
    
    animation_controls = show_spinner("Processing data...", color='blue')
    
    # Simulate progress
    for i in range(101):
        animation_controls['update_progress'](i / 100)
        time.sleep(0.02)
    
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()
    
    st.success("Spinner demo completed!")


def demo_progress_ring():
    """Demo the progress ring animation."""
    st.markdown("### ⭕ Progress Ring Animation")
    
    animation_controls = show_progress_ring("Analyzing image...", progress=0.0, color='green')
    
    # Simulate progress with text updates
    stages = [
        "Loading image data...",
        "Applying SVD decomposition...",
        "Calculating metrics...",
        "Finalizing results..."
    ]
    
    for i, stage in enumerate(stages):
        for j in range(25):
            progress = (i * 25 + j) / 100
            animation_controls['update_progress'](progress)
            animation_controls['update_text'](stage)
            time.sleep(0.02)
    
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()
    
    st.success("Progress ring demo completed!")


def demo_multi_step():
    """Demo the multi-step progress."""
    st.markdown("### 📋 Multi-Step Progress")
    
    steps = [
        "Initializing process...",
        "Loading data...",
        "Processing information...",
        "Generating results..."
    ]
    
    progress_controls = create_multi_step_progress(
        operation_id="demo_multi_step",
        title="Demo Process",
        steps=steps,
        animation_type="progress_ring"
    )
    
    # Simulate step progression
    for i in range(len(steps)):
        # Simulate work within each step
        for j in range(20):
            progress_controls['update_progress'](j / 20)
            time.sleep(0.05)
        
        if i < len(steps) - 1:
            progress_controls['next_step']()
        else:
            progress_controls['complete']()
    
    st.success("Multi-step demo completed!")


def show_animation_demo(animation_type: str, color_scheme: str):
    """Show a specific animation type and color scheme."""
    st.markdown(f"### {animation_type.title()} Animation ({color_scheme.title()} Theme)")
    
    animations = LoadingAnimations()
    
    if animation_type in ['progress_ring', 'wave', 'skeleton']:
        # Progress-based animations
        animation_controls = animations.show_loading_animation(
            animation_type=animation_type,
            text=f"Testing {animation_type} animation...",
            progress=0.0,
            color_scheme=color_scheme
        )
        
        # Simulate progress
        for i in range(101):
            animation_controls['update_progress'](i / 100)
            time.sleep(0.03)
    else:
        # Non-progress animations
        animation_controls = animations.show_loading_animation(
            animation_type=animation_type,
            text=f"Testing {animation_type} animation...",
            color_scheme=color_scheme
        )
        
        time.sleep(3)
    
    animation_controls['complete']()
    time.sleep(1)
    animation_controls['hide']()
    
    st.success(f"{animation_type.title()} animation demo completed!")


def simulate_image_compression():
    """Simulate image compression process."""
    st.markdown("### 🖼️ Image Compression Simulation")
    
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
        for j in range(15):
            progress_controls['update_progress'](j / 15)
            time.sleep(0.1)
        
        if i < len(compression_steps) - 1:
            progress_controls['next_step']()
        else:
            progress_controls['complete']()
    
    st.success("Image compression simulation completed!")


def simulate_batch_processing():
    """Simulate batch processing."""
    st.markdown("### 📦 Batch Processing Simulation")
    
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
                time.sleep(0.2)
        else:
            # Other steps
            for j in range(10):
                progress_controls['update_progress'](j / 10)
                time.sleep(0.05)
        
        if i < len(batch_steps) - 1:
            progress_controls['next_step']()
        else:
            progress_controls['complete']()
    
    st.success("Batch processing simulation completed!")


def demo_context_manager():
    """Demo the context manager."""
    st.markdown("### 🔧 Context Manager Demo")
    
    with loading_context("Processing with context manager...", "pulse", "purple"):
        time.sleep(3)
    
    st.success("Context manager demo completed!")


if __name__ == "__main__":
    show()