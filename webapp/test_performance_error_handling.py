"""
Test suite for performance optimization and error handling systems.
"""

import streamlit as st
import numpy as np
import time
import sys
from pathlib import Path
from PIL import Image
import tempfile
import os

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils.performance_optimizer import (
    ImageCache, PerformanceMonitor, ImageProcessor,
    get_image_cache, get_performance_monitor, get_image_processor,
    clear_all_caches, get_optimal_k_for_size
)
from utils.error_handler import (
    ErrorHandler, ErrorCategory, ErrorSeverity,
    get_error_handler, handle_error, error_boundary
)
from utils.fallback_states import (
    FallbackStateManager, get_fallback_manager,
    create_loading_failure_state, create_network_failure_state
)


def test_performance_optimization():
    """Test performance optimization features."""
    
    st.markdown("# 🚀 Performance Optimization Tests")
    
    # Test 1: Image Cache
    st.markdown("## 📦 Image Cache Test")
    
    cache = get_image_cache()
    
    # Create test image
    test_image = np.random.rand(100, 100, 3)
    k_value = 20
    mode = "RGB"
    
    # Test cache miss
    start_time = time.time()
    cached_result = cache.get(test_image, k_value, mode)
    cache_miss_time = time.time() - start_time
    
    if cached_result is None:
        st.success(f"✅ Cache miss detected correctly ({cache_miss_time:.4f}s)")
    else:
        st.error("❌ Cache should be empty for new test")
    
    # Test cache put and get
    test_result = {
        'compressed_image': test_image * 0.9,
        'metadata': {'k_value': k_value, 'compression_ratio': 5.0},
        'success': True
    }
    
    cache.put(test_image, k_value, mode, test_result)
    
    start_time = time.time()
    cached_result = cache.get(test_image, k_value, mode)
    cache_hit_time = time.time() - start_time
    
    if cached_result is not None:
        st.success(f"✅ Cache hit successful ({cache_hit_time:.4f}s)")
        st.info(f"Cache speedup: {cache_miss_time / cache_hit_time:.1f}x faster")
    else:
        st.error("❌ Cache should return stored result")
    
    # Show cache statistics
    cache_stats = cache.get_stats()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cache Entries", cache_stats['entries'])
    with col2:
        st.metric("Cache Size", f"{cache_stats['total_size_mb']:.2f} MB")
    with col3:
        st.metric("Max Entries", cache_stats['max_entries'])
    
    # Test 2: Performance Monitor
    st.markdown("## 📊 Performance Monitor Test")
    
    monitor = get_performance_monitor()
    
    # Test timing
    start_time = monitor.start_timing()
    time.sleep(0.1)  # Simulate work
    duration = monitor.end_timing(start_time, "test_operation")
    
    if 0.09 <= duration <= 0.15:  # Allow some tolerance
        st.success(f"✅ Timing measurement accurate: {duration:.3f}s")
    else:
        st.warning(f"⚠️ Timing may be inaccurate: {duration:.3f}s (expected ~0.1s)")
    
    # Test memory monitoring
    memory_usage = monitor.record_memory_usage()
    if memory_usage and memory_usage > 0:
        st.success(f"✅ Memory monitoring working: {memory_usage:.1f} MB")
    else:
        st.warning("⚠️ Memory monitoring may not be available")
    
    # Test cache hit/miss tracking
    monitor.record_cache_hit()
    monitor.record_cache_miss()
    hit_rate = monitor.get_cache_hit_rate()
    
    if hit_rate == 0.5:  # 1 hit, 1 miss = 50%
        st.success(f"✅ Cache hit rate tracking: {hit_rate:.1%}")
    else:
        st.warning(f"⚠️ Cache hit rate unexpected: {hit_rate:.1%}")
    
    # Test 3: Image Processor
    st.markdown("## 🖼️ Image Processor Test")
    
    processor = get_image_processor()
    
    # Test with small image (should not be resized)
    small_image = np.random.rand(256, 256, 3)
    optimized_small, opt_info_small = processor.optimize_image_for_processing(small_image)
    
    if not opt_info_small['was_resized']:
        st.success("✅ Small image not resized (correct)")
    else:
        st.error("❌ Small image should not be resized")
    
    # Test with large image (should be resized)
    large_image = np.random.rand(3000, 3000, 3)
    optimized_large, opt_info_large = processor.optimize_image_for_processing(large_image)
    
    if opt_info_large['was_resized']:
        st.success(f"✅ Large image resized: {opt_info_large['original_shape']} → {opt_info_large['new_shape']}")
        st.info(f"Memory saved: {opt_info_large['memory_saved_mb']:.1f} MB")
    else:
        st.warning("⚠️ Large image should be resized for performance")
    
    # Test processing with fallback
    test_result = processor.process_with_fallback(small_image, 10, "RGB")
    
    if test_result['success']:
        st.success("✅ Image processing successful")
        st.info(f"Processing time: {test_result['processing_time']:.3f}s")
        
        if test_result.get('memory_usage'):
            memory_info = test_result['memory_usage']
            st.info(f"Memory usage: {memory_info.get('before_mb', 0):.1f} → {memory_info.get('after_mb', 0):.1f} MB")
    else:
        st.error(f"❌ Image processing failed: {test_result.get('error')}")
    
    # Test 4: Optimal K Calculation
    st.markdown("## 🎯 Optimal K Calculation Test")
    
    test_cases = [
        (100, 100, 0.1),   # Small image, high compression
        (512, 512, 0.05),  # Medium image, very high compression
        (1024, 1024, 0.2)  # Large image, moderate compression
    ]
    
    for height, width, target_compression in test_cases:
        optimal_k = get_optimal_k_for_size(height, width, target_compression)
        max_k = min(height, width)
        
        if 1 <= optimal_k <= max_k:
            st.success(f"✅ Optimal k for {height}×{width} (target {target_compression:.1%}): k={optimal_k}")
        else:
            st.error(f"❌ Invalid optimal k: {optimal_k} (should be 1-{max_k})")
    
    # Performance summary
    st.markdown("## 📈 Performance Summary")
    
    avg_processing_time = monitor.get_average_processing_time()
    current_memory = monitor.get_current_memory_usage()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Processing Time", f"{avg_processing_time:.3f}s")
    with col2:
        st.metric("Current Memory", f"{current_memory:.1f} MB")
    with col3:
        st.metric("Cache Hit Rate", f"{monitor.get_cache_hit_rate():.1%}")


def test_error_handling():
    """Test error handling features."""
    
    st.markdown("# 🛡️ Error Handling Tests")
    
    error_handler = get_error_handler()
    
    # Test 1: Basic Error Handling
    st.markdown("## ⚠️ Basic Error Handling Test")
    
    try:
        # Simulate different types of errors
        test_errors = [
            (ValueError("Invalid k-value: -5"), ErrorCategory.VALIDATION),
            (MemoryError("Not enough memory"), ErrorCategory.MEMORY),
            (FileNotFoundError("Image file not found"), ErrorCategory.UPLOAD),
            (RuntimeError("SVD decomposition failed"), ErrorCategory.PROCESSING)
        ]
        
        for error, category in test_errors:
            result = error_handler.handle_error(
                error, 
                category, 
                context={'test': True},
                show_user_message=False  # Don't show UI messages in test
            )
            
            if result['error_type']:
                st.success(f"✅ {category.value} error classified as: {result['error_type']}")
            else:
                st.warning(f"⚠️ {category.value} error not classified")
            
            if result['recovery_attempted']:
                st.info(f"🔄 Recovery attempted for {category.value} error")
    
    except Exception as e:
        st.error(f"❌ Error handling test failed: {str(e)}")
    
    # Test 2: Error Statistics
    st.markdown("## 📊 Error Statistics Test")
    
    error_stats = error_handler.get_error_statistics()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Errors", error_stats['total_errors'])
    with col2:
        st.metric("Recent Errors", error_stats['recent_errors'])
    with col3:
        if error_stats['most_common_category']:
            st.metric("Most Common", error_stats['most_common_category'])
    
    if error_stats['category_breakdown']:
        st.markdown("**Error Breakdown:**")
        for category, count in error_stats['category_breakdown'].items():
            st.markdown(f"- {category}: {count}")
    
    # Test 3: Error Boundary Decorator
    st.markdown("## 🚧 Error Boundary Test")
    
    @error_boundary(ErrorCategory.PROCESSING, {'test_function': True})
    def test_function_with_error():
        """Test function that raises an error."""
        raise ValueError("Test error for boundary testing")
    
    @error_boundary(ErrorCategory.PROCESSING, {'test_function': True})
    def test_function_success():
        """Test function that succeeds."""
        return "Success!"
    
    # Test error boundary with error
    result_error = test_function_with_error()
    if isinstance(result_error, dict) and 'error_type' in result_error:
        st.success("✅ Error boundary caught error correctly")
    else:
        st.error("❌ Error boundary did not work")
    
    # Test error boundary with success
    result_success = test_function_success()
    if result_success == "Success!":
        st.success("✅ Error boundary allows success correctly")
    else:
        st.error("❌ Error boundary interfered with success")
    
    # Test 4: Memory Error Simulation
    st.markdown("## 💾 Memory Error Simulation")
    
    if st.button("Simulate Memory Error"):
        try:
            # Create a context that simulates memory error conditions
            large_image = np.random.rand(100, 100, 3)  # Small for testing
            context = {
                'image': large_image,
                'k_value': 50,
                'mode': 'RGB'
            }
            
            # Simulate memory error
            memory_error = MemoryError("Simulated memory error for testing")
            result = handle_error(memory_error, ErrorCategory.MEMORY, context, show_user_message=True)
            
            if result['recovery_attempted']:
                st.success("✅ Memory error recovery attempted")
            
        except Exception as e:
            st.error(f"❌ Memory error simulation failed: {str(e)}")


def test_fallback_states():
    """Test fallback state management."""
    
    st.markdown("# 🔄 Fallback States Tests")
    
    fallback_manager = get_fallback_manager()
    
    # Test 1: Processing Failure State
    st.markdown("## ⚠️ Processing Failure State Test")
    
    if st.button("Test Processing Failure State"):
        error_info = {
            'error_type': 'svd_failure',
            'error': 'Simulated SVD processing failure',
            'recovery_attempted': True,
            'recovery_successful': False,
            'suggestions': [
                'Try a smaller k-value',
                'Switch to grayscale mode',
                'Reduce image size'
            ]
        }
        
        # Create test image
        test_image = np.random.rand(256, 256, 3)
        
        result = fallback_manager.create_processing_failure_state(error_info, test_image)
        
        if result['state'] == 'fallback':
            st.success("✅ Processing failure state created")
    
    # Test 2: Loading Failure State
    st.markdown("## 📁 Loading Failure State Test")
    
    if st.button("Test Loading Failure State"):
        create_loading_failure_state("Simulated loading failure for testing")
        st.success("✅ Loading failure state displayed")
    
    # Test 3: Network Failure State
    st.markdown("## 🌐 Network Failure State Test")
    
    if st.button("Test Network Failure State"):
        create_network_failure_state()
        st.success("✅ Network failure state displayed")
    
    # Test 4: Memory Error Fallback
    st.markdown("## 💾 Memory Error Fallback Test")
    
    if st.button("Test Memory Error Fallback"):
        error_info = {
            'error_type': 'out_of_memory',
            'error': 'Simulated out of memory error',
            'recovery_attempted': True,
            'recovery_successful': False,
            'suggestions': [
                'Reduce image size',
                'Use smaller k-value',
                'Switch to grayscale mode',
                'Close other applications'
            ]
        }
        
        # Create large test image info
        test_image = np.random.rand(512, 512, 3)
        
        result = fallback_manager.create_processing_failure_state(error_info, test_image)
        
        if result['state'] == 'fallback':
            st.success("✅ Memory error fallback state created")


def test_integration():
    """Test integration between performance optimization and error handling."""
    
    st.markdown("# 🔗 Integration Tests")
    
    # Test 1: Performance Monitoring with Error Handling
    st.markdown("## 📊 Performance + Error Integration")
    
    processor = get_image_processor()
    monitor = get_performance_monitor()
    
    # Test successful processing with monitoring
    test_image = np.random.rand(128, 128, 3)
    
    start_time = monitor.start_timing()
    result = processor.process_with_fallback(test_image, 10, "RGB")
    processing_time = monitor.end_timing(start_time, "integration_test")
    
    if result['success']:
        st.success(f"✅ Successful processing monitored: {processing_time:.3f}s")
        
        # Check if result includes performance info
        if 'processing_time' in result:
            st.info(f"Internal timing: {result['processing_time']:.3f}s")
    else:
        st.error(f"❌ Processing failed: {result.get('error')}")
    
    # Test 2: Cache Performance with Error Recovery
    st.markdown("## 🚀 Cache + Error Recovery Integration")
    
    cache = get_image_cache()
    
    # Clear cache for clean test
    cache.clear()
    
    # First processing (cache miss)
    start_time = time.time()
    result1 = processor.process_with_fallback(test_image, 15, "RGB")
    time1 = time.time() - start_time
    
    # Second processing (cache hit)
    start_time = time.time()
    result2 = processor.process_with_fallback(test_image, 15, "RGB")
    time2 = time.time() - start_time
    
    if result1['success'] and result2['success']:
        speedup = time1 / time2 if time2 > 0 else float('inf')
        st.success(f"✅ Cache integration working")
        st.info(f"First run: {time1:.3f}s, Second run: {time2:.3f}s")
        st.info(f"Cache speedup: {speedup:.1f}x")
        
        if speedup > 2:  # Expect significant speedup from cache
            st.success("🚀 Excellent cache performance!")
        elif speedup > 1.5:
            st.info("✅ Good cache performance")
        else:
            st.warning("⚠️ Cache may not be working optimally")
    
    # Test 3: Memory Management Integration
    st.markdown("## 💾 Memory Management Integration")
    
    initial_memory = monitor.get_current_memory_usage()
    
    # Process multiple images to test memory management
    for i in range(3):
        test_img = np.random.rand(200, 200, 3)
        result = processor.process_with_fallback(test_img, 20, "RGB")
        
        if not result['success']:
            st.warning(f"Processing {i+1} failed: {result.get('error')}")
    
    final_memory = monitor.get_current_memory_usage()
    memory_increase = final_memory - initial_memory
    
    st.info(f"Memory usage: {initial_memory:.1f} → {final_memory:.1f} MB")
    st.info(f"Memory increase: {memory_increase:.1f} MB")
    
    if memory_increase < 50:  # Reasonable memory increase
        st.success("✅ Good memory management")
    else:
        st.warning("⚠️ High memory usage - check for memory leaks")


def main():
    """Main test interface."""
    
    st.set_page_config(
        page_title="Performance & Error Handling Tests",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Performance Optimization & Error Handling Test Suite")
    
    st.markdown("""
    This test suite validates the performance optimization and error handling systems
    implemented for the SVD image compression webapp.
    """)
    
    # Test selection
    test_type = st.selectbox(
        "Select Test Suite:",
        [
            "Performance Optimization",
            "Error Handling", 
            "Fallback States",
            "Integration Tests",
            "All Tests"
        ]
    )
    
    if st.button("🚀 Run Tests", type="primary"):
        
        if test_type == "Performance Optimization" or test_type == "All Tests":
            test_performance_optimization()
        
        if test_type == "Error Handling" or test_type == "All Tests":
            test_error_handling()
        
        if test_type == "Fallback States" or test_type == "All Tests":
            test_fallback_states()
        
        if test_type == "Integration Tests" or test_type == "All Tests":
            test_integration()
        
        st.markdown("---")
        st.success("🎉 Test suite completed!")
    
    # Cleanup section
    st.markdown("---")
    st.markdown("## 🧹 Cleanup")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Clear All Caches"):
            clear_all_caches()
            st.success("✅ All caches cleared!")
    
    with col2:
        if st.button("Reset Error Log"):
            error_handler = get_error_handler()
            error_handler.clear_error_log()
            st.success("✅ Error log cleared!")
    
    with col3:
        if st.button("Reset Performance Stats"):
            monitor = get_performance_monitor()
            monitor.metrics = {
                'processing_times': [],
                'memory_usage': [],
                'cache_hits': 0,
                'cache_misses': 0
            }
            st.success("✅ Performance stats reset!")


if __name__ == "__main__":
    main()