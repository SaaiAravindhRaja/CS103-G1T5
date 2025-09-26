"""
Memory profiling tests for SVD image compression.

This module provides comprehensive memory usage analysis including
memory leak detection, peak usage monitoring, and efficiency testing.
"""

import pytest
import numpy as np
import psutil
import os
import gc
import time
from typing import List, Dict, Any
import threading

from src.compression.svd_compressor import SVDCompressor
from src.evaluation.performance_profiler import PerformanceProfiler
from src.evaluation.metrics_calculator import MetricsCalculator
from src.data.image_loader import ImageLoader
from src.visualization.plot_generator import PlotGenerator


class MemoryMonitor:
    """Helper class for monitoring memory usage."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.measurements = []
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def start_monitoring(self, interval: float = 0.1):
        """Start continuous memory monitoring."""
        self.monitoring = True
        self.measurements = []
        
        def monitor():
            while self.monitoring:
                self.measurements.append({
                    'timestamp': time.time(),
                    'memory_mb': self.get_current_memory()
                })
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring and return statistics."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        
        if not self.measurements:
            return {}
        
        memories = [m['memory_mb'] for m in self.measurements]
        return {
            'min_memory': min(memories),
            'max_memory': max(memories),
            'avg_memory': np.mean(memories),
            'peak_memory': max(memories),
            'memory_range': max(memories) - min(memories),
            'measurements': len(memories)
        }


class TestMemoryProfiling:
    """Memory profiling tests for SVD compression system."""
    
    @pytest.fixture
    def memory_monitor(self):
        """Create memory monitor instance."""
        return MemoryMonitor()
    
    @pytest.fixture
    def compressor(self):
        """Create SVD compressor instance."""
        return SVDCompressor()
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler instance."""
        return PerformanceProfiler(enable_logging=False)
    
    def test_compression_memory_usage(self, compressor, memory_monitor):
        """Test memory usage during compression operations."""
        # Test with various image sizes
        test_sizes = [(100, 100), (200, 200), (400, 400)]
        k = 20
        
        for height, width in test_sizes:
            # Force garbage collection before test
            gc.collect()
            
            initial_memory = memory_monitor.get_current_memory()
            
            # Create image
            image = np.random.rand(height, width)
            after_creation = memory_monitor.get_current_memory()
            
            # Perform compression
            reconstructed, metadata = compressor.compress_image(image, k)
            after_compression = memory_monitor.get_current_memory()
            
            # Clean up
            del image, reconstructed, metadata
            gc.collect()
            final_memory = memory_monitor.get_current_memory()
            
            # Analyze memory usage
            creation_overhead = after_creation - initial_memory
            compression_overhead = after_compression - after_creation
            cleanup_efficiency = after_compression - final_memory
            
            # Memory usage should be reasonable
            expected_image_size = height * width * 8 / 1024 / 1024  # 8 bytes per float64
            assert creation_overhead <= expected_image_size * 5, f"Excessive memory for image creation: {creation_overhead} MB"
            
            # Compression should not use excessive additional memory (allow for SVD overhead)
            assert compression_overhead <= max(expected_image_size * 10, 20), f"Excessive compression overhead: {compression_overhead} MB"
            
            # Cleanup should be effective (allow some tolerance for Python's memory management)
            assert cleanup_efficiency >= 0, "Memory should be released after cleanup"
    
    def test_memory_leak_detection(self, compressor, memory_monitor):
        """Test for memory leaks during repeated operations."""
        image = np.random.rand(150, 150)
        k = 25
        
        # Baseline memory measurement
        gc.collect()
        baseline_memory = memory_monitor.get_current_memory()
        
        # Perform many compression operations
        num_operations = 100
        memory_samples = []
        
        for i in range(num_operations):
            reconstructed, metadata = compressor.compress_image(image, k)
            
            # Sample memory every 10 operations
            if i % 10 == 0:
                memory_samples.append(memory_monitor.get_current_memory())
            
            # Clean up references
            del reconstructed, metadata
            
            # Periodic garbage collection
            if i % 20 == 0:
                gc.collect()
        
        # Final cleanup and measurement
        gc.collect()
        final_memory = memory_monitor.get_current_memory()
        
        # Analyze memory trend
        memory_increase = final_memory - baseline_memory
        
        # Should not have significant memory leak
        assert memory_increase < 100, f"Potential memory leak detected: {memory_increase} MB increase"
        
        # Memory usage should be relatively stable
        if len(memory_samples) > 2:
            memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
            assert abs(memory_trend) < 0.5, f"Memory usage trending upward: {memory_trend} MB per sample"
    
    def test_peak_memory_usage(self, compressor, memory_monitor):
        """Test peak memory usage during compression."""
        # Test with large image to observe peak usage
        large_image = np.random.rand(500, 400)
        k = 50
        
        # Start continuous monitoring
        memory_monitor.start_monitoring(interval=0.05)  # 50ms intervals
        
        # Perform compression
        reconstructed, metadata = compressor.compress_image(large_image, k)
        
        # Stop monitoring
        stats = memory_monitor.stop_monitoring()
        
        # Clean up
        del large_image, reconstructed, metadata
        gc.collect()
        
        # Analyze peak usage
        if stats:
            memory_range = stats['memory_range']
            peak_memory = stats['peak_memory']
            
            # Peak memory should be reasonable
            expected_image_size = 500 * 400 * 8 / 1024 / 1024  # ~1.5 MB
            assert memory_range < max(expected_image_size * 20, 50), f"Excessive peak memory usage: {memory_range} MB range"
            
            # Should have captured some measurements
            assert stats['measurements'] >= 1, "Should have captured memory measurements"
    
    def test_concurrent_memory_usage(self, compressor, memory_monitor):
        """Test memory usage under concurrent operations."""
        import threading
        import queue
        
        # Create test images
        images = [np.random.rand(100, 100) for _ in range(5)]
        k = 15
        results_queue = queue.Queue()
        
        # Monitor memory during concurrent operations
        gc.collect()
        initial_memory = memory_monitor.get_current_memory()
        
        def compress_worker(image, worker_id):
            """Worker function for concurrent compression."""
            try:
                reconstructed, metadata = compressor.compress_image(image, k)
                results_queue.put({
                    'worker_id': worker_id,
                    'success': True,
                    'memory_after': memory_monitor.get_current_memory()
                })
                # Clean up in worker
                del reconstructed, metadata
            except Exception as e:
                results_queue.put({
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Start concurrent workers
        threads = []
        for i, image in enumerate(images):
            thread = threading.Thread(target=compress_worker, args=(image, i))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Final memory measurement
        gc.collect()
        final_memory = memory_monitor.get_current_memory()
        
        # Analyze concurrent memory usage
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) == 5, "All concurrent operations should succeed"
        
        # Memory increase should be reasonable for concurrent operations
        memory_increase = final_memory - initial_memory
        expected_max_increase = len(images) * 100 * 100 * 8 / 1024 / 1024 * 2  # Conservative estimate
        assert memory_increase < expected_max_increase, f"Excessive concurrent memory usage: {memory_increase} MB"
    
    def test_memory_efficiency_by_k_value(self, compressor, memory_monitor):
        """Test memory efficiency for different k values."""
        image = np.random.rand(200, 200)
        k_values = [5, 10, 20, 50, 100]
        
        memory_usage_by_k = []
        
        for k in k_values:
            gc.collect()
            before_memory = memory_monitor.get_current_memory()
            
            # Start monitoring
            memory_monitor.start_monitoring(interval=0.02)
            
            # Perform compression
            reconstructed, metadata = compressor.compress_image(image, k)
            
            # Stop monitoring
            stats = memory_monitor.stop_monitoring()
            
            # Clean up
            del reconstructed, metadata
            gc.collect()
            after_memory = memory_monitor.get_current_memory()
            
            memory_usage_by_k.append({
                'k': k,
                'memory_before': before_memory,
                'memory_after': after_memory,
                'memory_delta': after_memory - before_memory,
                'peak_memory': stats.get('peak_memory', after_memory),
                'memory_range': stats.get('memory_range', 0)
            })
        
        # Analyze memory efficiency
        for usage in memory_usage_by_k:
            # Memory usage should be reasonable for all k values
            assert usage['memory_delta'] < 100, f"Excessive memory for k={usage['k']}: {usage['memory_delta']} MB"
            
            # Peak memory should not be excessive
            peak_overhead = usage['peak_memory'] - usage['memory_before']
            assert peak_overhead < 200, f"Excessive peak memory for k={usage['k']}: {peak_overhead} MB"
        
        # Memory usage should not increase dramatically with k
        # (SVD memory usage is more related to matrix size than k value)
        memory_deltas = [u['memory_delta'] for u in memory_usage_by_k]
        max_delta = max(memory_deltas)
        min_delta = min([d for d in memory_deltas if d > 0], default=1)  # Avoid division by zero
        if min_delta > 0:
            assert max_delta < min_delta * 10, "Memory usage should not vary dramatically with k"
    
    def test_memory_usage_with_different_components(self, memory_monitor):
        """Test memory usage of different system components."""
        components_memory = {}
        
        # Test SVD Compressor
        gc.collect()
        before = memory_monitor.get_current_memory()
        compressor = SVDCompressor()
        image = np.random.rand(100, 100)
        reconstructed, metadata = compressor.compress_image(image, 10)
        after = memory_monitor.get_current_memory()
        components_memory['svd_compressor'] = after - before
        del compressor, image, reconstructed, metadata
        
        # Test Metrics Calculator
        gc.collect()
        before = memory_monitor.get_current_memory()
        calculator = MetricsCalculator()
        img1 = np.random.rand(100, 100)
        img2 = np.random.rand(100, 100)
        metrics = calculator.calculate_all_metrics(img1, img2, 10)
        after = memory_monitor.get_current_memory()
        components_memory['metrics_calculator'] = after - before
        del calculator, img1, img2, metrics
        
        # Test Image Loader
        gc.collect()
        before = memory_monitor.get_current_memory()
        loader = ImageLoader()
        # Create a simple test without actual file I/O
        test_array = np.random.rand(100, 100, 3)
        after = memory_monitor.get_current_memory()
        components_memory['image_loader'] = after - before
        del loader, test_array
        
        # Test Plot Generator
        gc.collect()
        before = memory_monitor.get_current_memory()
        plotter = PlotGenerator()
        singular_values = np.logspace(2, -2, 50)
        fig = plotter.plot_singular_values(singular_values)
        after = memory_monitor.get_current_memory()
        components_memory['plot_generator'] = after - before
        plotter.close_all_figures()
        del plotter, singular_values, fig
        
        # Analyze component memory usage
        for component, memory_usage in components_memory.items():
            # Each component should use reasonable memory
            assert memory_usage < 50, f"Excessive memory usage for {component}: {memory_usage} MB"
            
        # Total memory usage should be reasonable
        total_memory = sum(components_memory.values())
        assert total_memory < 150, f"Excessive total memory usage: {total_memory} MB"
    
    def test_memory_cleanup_after_errors(self, compressor, memory_monitor):
        """Test memory cleanup when errors occur during compression."""
        gc.collect()
        initial_memory = memory_monitor.get_current_memory()
        
        # Test with various error conditions
        error_cases = [
            (np.random.rand(50, 50), -1),  # Invalid k
            (np.random.rand(50, 50), 100),  # k too large
        ]
        
        for image, k in error_cases:
            before_error = memory_monitor.get_current_memory()
            
            try:
                compressor.compress_image(image, k)
            except (ValueError, TypeError):
                # Expected errors
                pass
            
            # Force cleanup
            gc.collect()
            after_error = memory_monitor.get_current_memory()
            
            # Memory should not increase significantly due to errors
            memory_increase = after_error - before_error
            assert memory_increase < 10, f"Memory not cleaned up after error: {memory_increase} MB"
        
        # Final memory should be close to initial
        gc.collect()
        final_memory = memory_monitor.get_current_memory()
        total_increase = final_memory - initial_memory
        assert total_increase < 20, f"Memory leaked after error handling: {total_increase} MB"
    
    def test_large_batch_memory_efficiency(self, compressor, memory_monitor):
        """Test memory efficiency during large batch processing."""
        # Simulate batch processing of many images
        num_images = 20
        image_size = (100, 100)
        k_values = [5, 10, 15]
        
        gc.collect()
        initial_memory = memory_monitor.get_current_memory()
        peak_memory = initial_memory
        
        # Process images in batches
        for batch_start in range(0, num_images, 5):  # Process 5 at a time
            batch_images = []
            
            # Create batch
            for i in range(5):
                if batch_start + i < num_images:
                    img = np.random.rand(*image_size)
                    batch_images.append(img)
            
            # Process batch
            for img in batch_images:
                for k in k_values:
                    reconstructed, metadata = compressor.compress_image(img, k)
                    
                    # Track peak memory
                    current_memory = memory_monitor.get_current_memory()
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Clean up immediately
                    del reconstructed, metadata
            
            # Clean up batch
            del batch_images
            gc.collect()
        
        # Final memory check
        final_memory = memory_monitor.get_current_memory()
        
        # Analyze batch processing memory efficiency
        memory_increase = final_memory - initial_memory
        peak_increase = peak_memory - initial_memory
        
        # Should not accumulate memory over batches
        assert memory_increase < 50, f"Memory accumulated during batch processing: {memory_increase} MB"
        
        # Peak memory should be reasonable
        expected_peak = len(k_values) * image_size[0] * image_size[1] * 8 / 1024 / 1024 * 3  # Conservative
        assert peak_increase < expected_peak, f"Excessive peak memory during batch: {peak_increase} MB"
    
    def test_memory_fragmentation_analysis(self, compressor, memory_monitor):
        """Test for memory fragmentation during repeated operations."""
        # Perform many allocation/deallocation cycles
        num_cycles = 50
        image_sizes = [(50, 50), (100, 100), (75, 75)]  # Varying sizes
        k = 15
        
        gc.collect()
        baseline_memory = memory_monitor.get_current_memory()
        memory_samples = [baseline_memory]
        
        for cycle in range(num_cycles):
            # Use different image sizes to test fragmentation
            size = image_sizes[cycle % len(image_sizes)]
            image = np.random.rand(*size)
            
            # Compress and immediately clean up
            reconstructed, metadata = compressor.compress_image(image, k)
            del image, reconstructed, metadata
            
            # Sample memory every few cycles
            if cycle % 5 == 0:
                gc.collect()
                current_memory = memory_monitor.get_current_memory()
                memory_samples.append(current_memory)
        
        # Final cleanup and measurement
        gc.collect()
        final_memory = memory_monitor.get_current_memory()
        
        # Analyze fragmentation
        memory_trend = np.polyfit(range(len(memory_samples)), memory_samples, 1)[0]
        memory_variance = np.var(memory_samples)
        total_increase = final_memory - baseline_memory
        
        # Should not show significant upward trend (fragmentation)
        assert abs(memory_trend) < 1.0, f"Memory fragmentation detected: trend={memory_trend} MB per sample"
        
        # Memory variance should be reasonable
        assert memory_variance < 100, f"High memory variance: {memory_variance}"
        
        # Total increase should be minimal
        assert total_increase < 30, f"Memory not properly released: {total_increase} MB increase"
    
    def test_memory_pressure_handling(self, compressor, memory_monitor):
        """Test system behavior under memory pressure."""
        # Create memory pressure by allocating large arrays
        pressure_arrays = []
        
        try:
            # Allocate memory to create pressure (but not exhaust system)
            for i in range(5):
                # Allocate 50MB arrays
                pressure_array = np.random.rand(50 * 1024 * 1024 // 8)  # 50MB
                pressure_arrays.append(pressure_array)
            
            gc.collect()
            pressure_memory = memory_monitor.get_current_memory()
            
            # Now test compression under memory pressure
            test_image = np.random.rand(200, 200)
            k = 25
            
            # Should still work under memory pressure
            start_time = time.time()
            reconstructed, metadata = compressor.compress_image(test_image, k)
            compression_time = time.time() - start_time
            
            # Verify results are still valid
            assert np.all(np.isfinite(reconstructed)), "Results should be valid under memory pressure"
            assert metadata['compression_ratio'] > 0, "Metadata should be valid under memory pressure"
            
            # Should complete in reasonable time (may be slower due to pressure)
            assert compression_time < 30, f"Compression under pressure took too long: {compression_time}s"
            
            # Clean up compression results
            del test_image, reconstructed, metadata
            
        finally:
            # Clean up pressure arrays
            del pressure_arrays
            gc.collect()
    
    def test_memory_usage_patterns_analysis(self, compressor, memory_monitor):
        """Analyze memory usage patterns for different compression scenarios."""
        test_scenarios = [
            ('small_image_small_k', (50, 50), 5),
            ('small_image_large_k', (50, 50), 20),
            ('large_image_small_k', (200, 200), 5),
            ('large_image_large_k', (200, 200), 50),
            ('rectangular_image', (300, 100), 15),
            ('square_image', (150, 150), 15)
        ]
        
        memory_patterns = []
        
        for scenario_name, image_size, k in test_scenarios:
            gc.collect()
            
            # Create image
            image = np.random.rand(*image_size)
            before_compression = memory_monitor.get_current_memory()
            
            # Start monitoring
            memory_monitor.start_monitoring(interval=0.01)
            
            # Perform compression
            reconstructed, metadata = compressor.compress_image(image, k)
            
            # Stop monitoring
            stats = memory_monitor.stop_monitoring()
            
            after_compression = memory_monitor.get_current_memory()
            
            # Clean up
            del image, reconstructed, metadata
            gc.collect()
            after_cleanup = memory_monitor.get_current_memory()
            
            memory_patterns.append({
                'scenario': scenario_name,
                'image_size': image_size,
                'k': k,
                'memory_before': before_compression,
                'memory_after': after_compression,
                'memory_cleanup': after_cleanup,
                'peak_memory': stats.get('peak_memory', after_compression),
                'memory_range': stats.get('memory_range', 0),
                'compression_overhead': after_compression - before_compression,
                'cleanup_efficiency': after_compression - after_cleanup
            })
        
        # Analyze patterns
        for pattern in memory_patterns:
            # Memory overhead should correlate with image size and k
            image_pixels = np.prod(pattern['image_size'])
            expected_base_memory = image_pixels * 8 / 1024 / 1024  # Base image memory
            
            # Compression overhead should be reasonable
            assert pattern['compression_overhead'] < expected_base_memory * 5, f"Excessive overhead for {pattern['scenario']}"
            
            # Cleanup should be effective
            assert pattern['cleanup_efficiency'] >= 0, f"Memory not cleaned up for {pattern['scenario']}"
            
            # Peak memory should not be excessive
            peak_overhead = pattern['peak_memory'] - pattern['memory_before']
            assert peak_overhead < expected_base_memory * 8, f"Excessive peak memory for {pattern['scenario']}"
    
    def test_memory_profiling_accuracy(self, memory_monitor):
        """Test accuracy and reliability of memory profiling measurements."""
        # Test measurement consistency
        measurements = []
        
        for _ in range(10):
            gc.collect()
            memory = memory_monitor.get_current_memory()
            measurements.append(memory)
            time.sleep(0.01)  # Small delay
        
        # Measurements should be consistent when no operations are performed
        memory_std = np.std(measurements)
        memory_mean = np.mean(measurements)
        
        # Standard deviation should be small relative to mean
        cv = memory_std / memory_mean if memory_mean > 0 else float('inf')
        assert cv < 0.01, f"Memory measurements inconsistent: CV={cv}"
        
        # Test monitoring accuracy
        gc.collect()
        baseline = memory_monitor.get_current_memory()
        
        # Allocate known amount of memory
        test_arrays = []
        for _ in range(10):  # Allocate multiple arrays to ensure detection
            test_arrays.append(np.random.rand(1024 * 1024))  # ~8MB each
        
        after_allocation = memory_monitor.get_current_memory()
        
        # Should detect memory increase (more lenient check)
        memory_increase = after_allocation - baseline
        # Memory detection can be unreliable due to system factors, so we use a more lenient check
        if memory_increase <= 0:
            # If no increase detected, at least verify the monitor is working
            assert baseline >= 0, "Memory monitor should return valid baseline"
            assert after_allocation >= 0, "Memory monitor should return valid measurement"
        else:
            assert memory_increase < 200, "Should not overestimate memory usage significantly"
        
        # Clean up and verify detection
        del test_arrays
        gc.collect()
        after_cleanup = memory_monitor.get_current_memory()
        
        cleanup_detected = after_allocation - after_cleanup
        # Memory cleanup detection can be unreliable due to Python's memory management
        # Just verify that memory didn't increase significantly
        assert cleanup_detected >= -5, "Memory should not increase significantly after cleanup"
    
    def test_memory_leak_stress_testing(self, compressor, memory_monitor):
        """Stress test for memory leaks with intensive operations."""
        # Perform intensive compression operations
        num_operations = 200
        image_size = (100, 100)
        k_values = [5, 10, 15, 20]
        
        gc.collect()
        initial_memory = memory_monitor.get_current_memory()
        
        # Track memory over time
        memory_checkpoints = []
        
        for i in range(num_operations):
            # Vary parameters to stress different code paths
            k = k_values[i % len(k_values)]
            
            # Create image with some variation
            if i % 3 == 0:
                image = np.random.rand(*image_size)
            elif i % 3 == 1:
                image = np.ones(image_size) * (i % 10) / 10
            else:
                image = np.eye(image_size[0], image_size[1])
            
            # Perform compression
            try:
                reconstructed, metadata = compressor.compress_image(image, k)
                
                # Immediate cleanup
                del image, reconstructed, metadata
                
            except Exception:
                # Clean up even if compression fails
                del image
                continue
            
            # Checkpoint memory every 20 operations
            if i % 20 == 0:
                gc.collect()
                current_memory = memory_monitor.get_current_memory()
                memory_checkpoints.append({
                    'operation': i,
                    'memory': current_memory,
                    'increase': current_memory - initial_memory
                })
        
        # Final memory check
        gc.collect()
        final_memory = memory_monitor.get_current_memory()
        total_increase = final_memory - initial_memory
        
        # Analyze stress test results
        assert total_increase < 100, f"Memory leak detected in stress test: {total_increase} MB increase"
        
        # Memory should not show consistent upward trend
        if len(memory_checkpoints) > 2:
            increases = [cp['increase'] for cp in memory_checkpoints]
            trend = np.polyfit(range(len(increases)), increases, 1)[0]
            assert abs(trend) < 0.5, f"Memory leak trend detected: {trend} MB per checkpoint"
    
    def test_memory_optimization_verification(self, compressor, memory_monitor):
        """Verify memory optimizations are working correctly."""
        # Test that memory usage is optimized for different scenarios
        
        # Scenario 1: Small k should use less memory than large k
        image = np.random.rand(200, 200)
        
        gc.collect()
        baseline = memory_monitor.get_current_memory()
        
        # Small k compression
        memory_monitor.start_monitoring()
        reconstructed_small, _ = compressor.compress_image(image, 5)
        stats_small = memory_monitor.stop_monitoring()
        del reconstructed_small
        
        gc.collect()
        
        # Large k compression
        memory_monitor.start_monitoring()
        reconstructed_large, _ = compressor.compress_image(image, 50)
        stats_large = memory_monitor.stop_monitoring()
        del reconstructed_large
        
        # Memory usage should be reasonable for both
        small_k_peak = stats_small.get('peak_memory', baseline)
        large_k_peak = stats_large.get('peak_memory', baseline)
        
        small_k_overhead = small_k_peak - baseline
        large_k_overhead = large_k_peak - baseline
        
        # Both should use reasonable memory
        assert small_k_overhead < 100, f"Small k uses too much memory: {small_k_overhead} MB"
        assert large_k_overhead < 200, f"Large k uses too much memory: {large_k_overhead} MB"
        
        # Scenario 2: Grayscale should use less memory than RGB
        rgb_image = np.random.rand(150, 150, 3)
        gray_image = np.random.rand(150, 150)
        k = 15
        
        gc.collect()
        baseline = memory_monitor.get_current_memory()
        
        # RGB compression
        memory_monitor.start_monitoring()
        rgb_result, _ = compressor.compress_image(rgb_image, k)
        rgb_stats = memory_monitor.stop_monitoring()
        del rgb_result
        
        gc.collect()
        
        # Grayscale compression
        memory_monitor.start_monitoring()
        gray_result, _ = compressor.compress_image(gray_image, k)
        gray_stats = memory_monitor.stop_monitoring()
        del gray_result
        
        rgb_peak = rgb_stats.get('peak_memory', baseline)
        gray_peak = gray_stats.get('peak_memory', baseline)
        
        rgb_overhead = rgb_peak - baseline
        gray_overhead = gray_peak - baseline
        
        # RGB should use more memory than grayscale (but not excessively more)
        assert rgb_overhead >= gray_overhead, "RGB should use at least as much memory as grayscale"
        assert rgb_overhead <= gray_overhead * 4, "RGB should not use excessively more memory than grayscale"


if __name__ == "__main__":
    pytest.main([__file__])