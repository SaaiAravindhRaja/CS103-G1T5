"""
Performance benchmarks for SVD compression algorithms.

This module provides comprehensive performance testing for compression
algorithms including timing, memory usage, and scalability analysis.
"""

import pytest
import numpy as np
import time
import psutil
import os
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any

from src.compression.svd_compressor import SVDCompressor
from src.evaluation.performance_profiler import PerformanceProfiler
from src.evaluation.metrics_calculator import MetricsCalculator


class TestCompressionPerformance:
    """Performance benchmarks for SVD compression algorithms."""
    
    @pytest.fixture
    def compressor(self):
        """Create SVD compressor instance."""
        return SVDCompressor()
    
    @pytest.fixture
    def profiler(self):
        """Create performance profiler instance."""
        return PerformanceProfiler(enable_logging=False)
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create metrics calculator instance."""
        return MetricsCalculator()
    
    def generate_test_images(self, sizes: List[tuple], seed: int = 42) -> Dict[str, np.ndarray]:
        """Generate test images of various sizes."""
        np.random.seed(seed)
        images = {}
        
        for height, width in sizes:
            # Generate grayscale image
            grayscale = np.random.rand(height, width).astype(np.float64)
            images[f'grayscale_{height}x{width}'] = grayscale
            
            # Generate RGB image
            rgb = np.random.rand(height, width, 3).astype(np.float64)
            images[f'rgb_{height}x{width}'] = rgb
        
        return images
    
    def test_compression_timing_scalability(self, compressor, profiler):
        """Test compression timing scalability with image size."""
        # Test various image sizes
        sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
        k_values = [10, 20, 50]
        
        results = []
        
        for height, width in sizes:
            image = np.random.rand(height, width)
            
            for k in k_values:
                if k < min(height, width):  # Ensure k is valid
                    _, perf_result = profiler.profile_compression(compressor, image, k)
                    
                    results.append({
                        'image_size': f'{height}x{width}',
                        'pixels': height * width,
                        'k': k,
                        'execution_time': perf_result.execution_time,
                        'memory_delta': perf_result.memory_delta
                    })
        
        # Analyze results
        assert len(results) > 0
        
        # Check that timing generally increases with image size
        size_64 = [r for r in results if r['image_size'] == '64x64' and r['k'] == 10]
        size_512 = [r for r in results if r['image_size'] == '512x512' and r['k'] == 10]
        
        if size_64 and size_512:
            assert size_512[0]['execution_time'] > size_64[0]['execution_time']
        
        # Check that all operations complete within reasonable time
        max_time = max(r['execution_time'] for r in results)
        assert max_time < 10.0, f"Compression took too long: {max_time}s"
    
    def test_memory_usage_scalability(self, compressor, profiler):
        """Test memory usage scalability with image size."""
        sizes = [(100, 100), (200, 200), (400, 400)]
        k = 20
        
        memory_usage = []
        
        for height, width in sizes:
            image = np.random.rand(height, width)
            
            # Measure memory before and after compression
            _, perf_result = profiler.profile_compression(compressor, image, k)
            
            memory_usage.append({
                'image_size': f'{height}x{width}',
                'pixels': height * width,
                'memory_before': perf_result.memory_before,
                'memory_after': perf_result.memory_after,
                'memory_delta': perf_result.memory_delta,
                'memory_peak': perf_result.memory_peak
            })
        
        # Check that memory usage is reasonable
        for usage in memory_usage:
            # Memory delta should be reasonable (not excessive)
            assert usage['memory_delta'] < 1000, f"Excessive memory usage: {usage['memory_delta']} MB"
            
            # Peak memory should not be much higher than final memory
            peak_overhead = usage['memory_peak'] - usage['memory_after']
            assert peak_overhead < 500, f"Excessive peak memory: {peak_overhead} MB"
    
    def test_k_value_performance_impact(self, compressor, profiler):
        """Test performance impact of different k values."""
        image = np.random.rand(256, 256)
        k_values = [5, 10, 20, 50, 100]
        
        timing_results = []
        
        for k in k_values:
            _, perf_result = profiler.profile_compression(compressor, image, k)
            timing_results.append({
                'k': k,
                'execution_time': perf_result.execution_time,
                'memory_delta': perf_result.memory_delta
            })
        
        # Check that timing generally increases with k
        times = [r['execution_time'] for r in timing_results]
        
        # Should show some correlation with k (though not necessarily strict monotonic)
        # At minimum, k=100 should not be faster than k=5
        k5_time = next(r['execution_time'] for r in timing_results if r['k'] == 5)
        k100_time = next(r['execution_time'] for r in timing_results if r['k'] == 100)
        
        # Allow some tolerance for measurement noise
        assert k100_time <= k5_time * 3, "k=100 should not be much slower than k=5 for SVD"
    
    def test_batch_compression_performance(self, compressor, profiler):
        """Test performance of batch compression operations."""
        # Create multiple images
        images = []
        for i in range(10):
            img = np.random.rand(128, 128)
            images.append(img)
        
        k = 20
        
        # Time batch processing
        start_time = time.time()
        results = []
        
        for i, image in enumerate(images):
            _, perf_result = profiler.profile_compression(compressor, image, k)
            results.append(perf_result)
        
        total_time = time.time() - start_time
        
        # Check performance characteristics
        assert total_time < 30.0, f"Batch processing took too long: {total_time}s"
        assert len(results) == 10
        
        # Check consistency of timing
        times = [r.execution_time for r in results]
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Standard deviation should be reasonable (not too much variation)
        assert std_time < avg_time, "Timing should be relatively consistent"
    
    def test_memory_leak_detection(self, compressor):
        """Test for memory leaks during repeated compression."""
        image = np.random.rand(200, 200)
        k = 30
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many compression operations
        for i in range(50):
            reconstructed, metadata = compressor.compress_image(image, k)
            # Force garbage collection by deleting references
            del reconstructed, metadata
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal (< 100MB for 50 operations)
        assert memory_increase < 100, f"Potential memory leak: {memory_increase} MB increase"
    
    def test_concurrent_compression_performance(self, compressor):
        """Test performance under concurrent compression operations."""
        import threading
        import queue
        
        # Create test data
        images = [np.random.rand(100, 100) for _ in range(5)]
        k = 15
        results_queue = queue.Queue()
        
        def compress_worker(image, worker_id):
            """Worker function for concurrent compression."""
            start_time = time.time()
            try:
                reconstructed, metadata = compressor.compress_image(image, k)
                end_time = time.time()
                results_queue.put({
                    'worker_id': worker_id,
                    'success': True,
                    'execution_time': end_time - start_time,
                    'compression_ratio': metadata['compression_ratio']
                })
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
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Check that all operations succeeded
        assert len(results) == 5
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) == 5, "All concurrent operations should succeed"
        
        # Check timing consistency
        times = [r['execution_time'] for r in successful_results]
        max_time = max(times)
        assert max_time < 5.0, f"Concurrent operations took too long: {max_time}s"
    
    def test_large_image_performance(self, compressor, profiler):
        """Test performance with large images."""
        # Test with a large image (but not too large to avoid CI timeouts)
        large_image = np.random.rand(1000, 800)
        k = 50
        
        # Profile the compression
        _, perf_result = profiler.profile_compression(compressor, large_image, k)
        
        # Check that it completes in reasonable time
        assert perf_result.execution_time < 30.0, f"Large image compression took too long: {perf_result.execution_time}s"
        
        # Check memory usage is reasonable
        assert perf_result.memory_delta < 2000, f"Excessive memory usage: {perf_result.memory_delta} MB"
    
    def test_compression_quality_vs_performance_tradeoff(self, compressor, metrics_calculator):
        """Test the tradeoff between compression quality and performance."""
        image = np.random.rand(200, 200)
        k_values = [5, 10, 20, 50, 100]
        
        results = []
        
        for k in k_values:
            # Time the compression
            start_time = time.time()
            reconstructed, metadata = compressor.compress_image(image, k)
            compression_time = time.time() - start_time
            
            # Calculate quality metrics
            metrics = metrics_calculator.calculate_all_metrics(image, reconstructed, k)
            
            results.append({
                'k': k,
                'compression_time': compression_time,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'compression_ratio': metrics['compression_ratio']
            })
        
        # Analyze tradeoffs
        assert len(results) == len(k_values)
        
        # Quality should generally improve with higher k
        psnr_values = [r['psnr'] for r in results]
        assert psnr_values[-1] > psnr_values[0], "PSNR should improve with higher k"
        
        # Compression ratio should decrease with higher k
        ratios = [r['compression_ratio'] for r in results]
        assert ratios[0] > ratios[-1], "Compression ratio should decrease with higher k"
    
    def test_numerical_stability_performance(self, compressor):
        """Test performance with numerically challenging images."""
        # Test with images that might cause numerical issues
        test_cases = [
            np.ones((100, 100)) * 1e-10,  # Very small values
            np.ones((100, 100)) * 0.999999,  # Values close to 1
            np.random.rand(100, 100) * 1e-8,  # Very small random values
            np.eye(100),  # Identity matrix (sparse)
        ]
        
        k = 20
        
        for i, image in enumerate(test_cases):
            start_time = time.time()
            try:
                reconstructed, metadata = compressor.compress_image(image, k)
                compression_time = time.time() - start_time
                
                # Should complete in reasonable time even for challenging cases
                assert compression_time < 10.0, f"Numerically challenging case {i} took too long: {compression_time}s"
                
                # Results should be valid
                assert np.all(np.isfinite(reconstructed)), f"Non-finite values in result for case {i}"
                assert metadata['compression_ratio'] > 0, f"Invalid compression ratio for case {i}"
                
            except Exception as e:
                pytest.fail(f"Compression failed for numerically challenging case {i}: {e}")
    
    def test_compression_algorithm_efficiency(self, compressor, profiler):
        """Test efficiency of compression algorithms across different scenarios."""
        # Test scenarios with different characteristics
        test_scenarios = [
            ('random', np.random.rand(200, 200)),
            ('structured', np.outer(np.linspace(0, 1, 200), np.linspace(0, 1, 200))),
            ('sparse', np.zeros((200, 200))),
            ('high_frequency', np.sin(np.linspace(0, 50*np.pi, 200*200)).reshape(200, 200)),
            ('low_rank', np.outer(np.random.rand(200), np.random.rand(200)))
        ]
        
        k_values = [10, 30, 50]
        efficiency_results = []
        
        for scenario_name, image in test_scenarios:
            for k in k_values:
                _, perf_result = profiler.profile_compression(compressor, image, k)
                
                efficiency_results.append({
                    'scenario': scenario_name,
                    'k': k,
                    'execution_time': perf_result.execution_time,
                    'memory_delta': perf_result.memory_delta,
                    'efficiency_score': k / perf_result.execution_time  # Higher is better
                })
        
        # Analyze efficiency patterns
        assert len(efficiency_results) == len(test_scenarios) * len(k_values)
        
        # Low-rank images should be more efficient
        low_rank_results = [r for r in efficiency_results if r['scenario'] == 'low_rank']
        random_results = [r for r in efficiency_results if r['scenario'] == 'random']
        
        if low_rank_results and random_results:
            avg_low_rank_time = np.mean([r['execution_time'] for r in low_rank_results])
            avg_random_time = np.mean([r['execution_time'] for r in random_results])
            
            # Low-rank should generally be faster or comparable
            assert avg_low_rank_time <= avg_random_time * 2, "Low-rank compression should be efficient"
    
    def test_compression_throughput_analysis(self, compressor):
        """Test compression throughput under various conditions."""
        # Test different image sizes and measure throughput
        image_sizes = [(64, 64), (128, 128), (256, 256)]
        k = 20
        
        throughput_results = []
        
        for height, width in image_sizes:
            images = [np.random.rand(height, width) for _ in range(5)]
            
            # Measure batch throughput
            start_time = time.time()
            for image in images:
                compressor.compress_image(image, k)
            end_time = time.time()
            
            total_pixels = len(images) * height * width
            total_time = end_time - start_time
            throughput = total_pixels / total_time  # pixels per second
            
            throughput_results.append({
                'image_size': f'{height}x{width}',
                'total_pixels': total_pixels,
                'total_time': total_time,
                'throughput_pixels_per_sec': throughput,
                'throughput_megapixels_per_sec': throughput / 1e6
            })
        
        # Analyze throughput
        for result in throughput_results:
            # Should achieve reasonable throughput
            assert result['throughput_megapixels_per_sec'] > 0.1, f"Low throughput: {result['throughput_megapixels_per_sec']} MP/s"
            
            # Larger images might have different throughput characteristics
            assert result['total_time'] > 0, "Should record positive processing time"
    
    def test_compression_scalability_limits(self, compressor):
        """Test compression behavior at scalability limits."""
        # Test with very large k values (but still valid)
        large_image = np.random.rand(500, 400)
        max_k = min(large_image.shape) - 1  # Maximum valid k
        
        # Test k values approaching the limit
        test_k_values = [max_k // 4, max_k // 2, max_k - 10, max_k - 1]
        
        scalability_results = []
        
        for k in test_k_values:
            if k > 0:
                start_time = time.time()
                try:
                    reconstructed, metadata = compressor.compress_image(large_image, k)
                    compression_time = time.time() - start_time
                    
                    scalability_results.append({
                        'k': k,
                        'k_ratio': k / max_k,
                        'compression_time': compression_time,
                        'compression_ratio': metadata['compression_ratio'],
                        'success': True
                    })
                    
                except Exception as e:
                    scalability_results.append({
                        'k': k,
                        'k_ratio': k / max_k,
                        'compression_time': float('inf'),
                        'compression_ratio': 0,
                        'success': False,
                        'error': str(e)
                    })
        
        # Analyze scalability limits
        successful_results = [r for r in scalability_results if r['success']]
        assert len(successful_results) > 0, "Should handle some large k values"
        
        # Time should increase with k, but not exponentially
        if len(successful_results) > 1:
            times = [r['compression_time'] for r in successful_results]
            k_values = [r['k'] for r in successful_results]
            
            # Should complete even for large k values
            max_time = max(times)
            assert max_time < 60, f"Large k compression took too long: {max_time}s"
    
    def test_compression_memory_efficiency_benchmarks(self, compressor):
        """Benchmark memory efficiency of compression operations."""
        import psutil
        import os
        import gc
        
        process = psutil.Process(os.getpid())
        
        # Test memory efficiency with different image characteristics
        test_cases = [
            ('small_k', np.random.rand(200, 200), 5),
            ('medium_k', np.random.rand(200, 200), 25),
            ('large_k', np.random.rand(200, 200), 75),
            ('large_image_small_k', np.random.rand(400, 300), 10),
            ('large_image_large_k', np.random.rand(400, 300), 50)
        ]
        
        memory_benchmarks = []
        
        for case_name, image, k in test_cases:
            # Force garbage collection before test
            gc.collect()
            
            # Measure memory before compression
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Perform compression
            reconstructed, metadata = compressor.compress_image(image, k)
            
            # Measure memory after compression
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Clean up and measure final memory
            del reconstructed, metadata
            gc.collect()
            memory_final = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_benchmarks.append({
                'case': case_name,
                'image_size': image.shape,
                'k': k,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_final': memory_final,
                'memory_peak_delta': memory_after - memory_before,
                'memory_cleanup_efficiency': memory_after - memory_final
            })
        
        # Analyze memory efficiency
        for benchmark in memory_benchmarks:
            # Peak memory usage should be reasonable
            expected_image_memory = np.prod(benchmark['image_size']) * 8 / 1024 / 1024  # 8 bytes per float64
            assert benchmark['memory_peak_delta'] < expected_image_memory * 10, f"Excessive memory for {benchmark['case']}"
            
            # Memory cleanup should be effective
            assert benchmark['memory_cleanup_efficiency'] >= 0, f"Memory not cleaned up for {benchmark['case']}"
    
    def test_compression_performance_regression(self, compressor, profiler):
        """Test for performance regression with standard benchmarks."""
        # Standard benchmark cases
        benchmark_cases = [
            ('standard_64x64', np.random.rand(64, 64), 10),
            ('standard_128x128', np.random.rand(128, 128), 20),
            ('standard_256x256', np.random.rand(256, 256), 30)
        ]
        
        # Expected performance baselines (these would be updated based on actual measurements)
        performance_baselines = {
            'standard_64x64': {'max_time': 1.0, 'max_memory': 50},
            'standard_128x128': {'max_time': 3.0, 'max_memory': 100},
            'standard_256x256': {'max_time': 10.0, 'max_memory': 200}
        }
        
        regression_results = []
        
        for case_name, image, k in benchmark_cases:
            # Run benchmark multiple times for stability
            times = []
            memory_deltas = []
            
            for _ in range(3):  # 3 runs for averaging
                _, perf_result = profiler.profile_compression(compressor, image, k)
                times.append(perf_result.execution_time)
                memory_deltas.append(perf_result.memory_delta)
            
            avg_time = np.mean(times)
            avg_memory = np.mean(memory_deltas)
            
            regression_results.append({
                'case': case_name,
                'avg_time': avg_time,
                'avg_memory': avg_memory,
                'time_stability': np.std(times) / avg_time,  # Coefficient of variation
                'memory_stability': np.std(memory_deltas) / max(avg_memory, 1)
            })
            
            # Check against baselines
            baseline = performance_baselines.get(case_name, {})
            if 'max_time' in baseline:
                assert avg_time <= baseline['max_time'], f"Performance regression in {case_name}: {avg_time}s > {baseline['max_time']}s"
            if 'max_memory' in baseline:
                assert avg_memory <= baseline['max_memory'], f"Memory regression in {case_name}: {avg_memory}MB > {baseline['max_memory']}MB"
        
        # Check performance stability
        for result in regression_results:
            # Time measurements should be relatively stable
            assert result['time_stability'] < 0.5, f"Unstable timing for {result['case']}: CV={result['time_stability']}"
            
            # Memory measurements should be stable
            assert result['memory_stability'] < 1.0, f"Unstable memory for {result['case']}: CV={result['memory_stability']}"


if __name__ == "__main__":
    pytest.main([__file__])