"""
Unit tests for PerformanceProfiler class.

Tests performance measurement accuracy, memory tracking,
and result storage functionality.
"""

import pytest
import numpy as np
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
from src.evaluation.performance_profiler import PerformanceProfiler, PerformanceResult


class TestPerformanceProfiler:
    """Test suite for PerformanceProfiler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler(enable_logging=False)
        
    def test_init(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(enable_logging=True)
        assert profiler.enable_logging is True
        assert len(profiler.results) == 0
        
        profiler_no_log = PerformanceProfiler(enable_logging=False)
        assert profiler_no_log.enable_logging is False
    
    def test_profile_function_basic(self):
        """Test basic function profiling."""
        def test_function(x, y):
            time.sleep(0.01)  # Small delay to measure
            return x + y
        
        result, perf_result = self.profiler.profile_function(test_function, 2, 3)
        
        # Check function result
        assert result == 5
        
        # Check performance result
        assert isinstance(perf_result, PerformanceResult)
        assert perf_result.operation_name == 'test_function'
        assert perf_result.execution_time > 0.005  # Should be at least 5ms
        assert perf_result.execution_time < 0.1    # Should be less than 100ms
        assert isinstance(perf_result.memory_before, float)
        assert isinstance(perf_result.memory_after, float)
        assert perf_result.memory_delta == perf_result.memory_after - perf_result.memory_before
        
        # Check that result was stored
        assert len(self.profiler.results) == 1
        assert self.profiler.results[0] == perf_result
    
    def test_profile_function_with_kwargs(self):
        """Test function profiling with keyword arguments."""
        def test_function(x, y, multiplier=1):
            return (x + y) * multiplier
        
        result, perf_result = self.profiler.profile_function(
            test_function, 2, 3, multiplier=5
        )
        
        assert result == 25
        assert perf_result.operation_name == 'test_function'
    
    def test_profile_function_custom_name_and_parameters(self):
        """Test function profiling with custom operation name and parameters."""
        def simple_add(a, b):
            return a + b
        
        parameters = {'input_a': 10, 'input_b': 20}
        result, perf_result = self.profiler.profile_function(
            simple_add, 10, 20,
            operation_name='custom_addition',
            parameters=parameters
        )
        
        assert result == 30
        assert perf_result.operation_name == 'custom_addition'
        assert perf_result.parameters == parameters
    
    def test_profile_function_exception_handling(self):
        """Test that exceptions are properly propagated."""
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            self.profiler.profile_function(failing_function)
        
        # Should not have stored any results
        assert len(self.profiler.results) == 0
    
    def test_profile_compression(self):
        """Test compression operation profiling."""
        # Mock compressor
        mock_compressor = Mock()
        mock_compressor.compress_image.return_value = (np.random.rand(10, 10), {'ratio': 2.5})
        
        test_image = np.random.rand(10, 10)
        k = 5
        
        result, perf_result = self.profiler.profile_compression(mock_compressor, test_image, k)
        
        # Check that compressor was called correctly
        mock_compressor.compress_image.assert_called_once_with(test_image, k)
        
        # Check performance result
        assert perf_result.operation_name == 'svd_compression'
        assert perf_result.parameters['k'] == k
        assert perf_result.parameters['image_shape'] == test_image.shape
        assert perf_result.parameters['image_dtype'] == str(test_image.dtype)
        assert 'image_size_mb' in perf_result.parameters
    
    def test_profile_metrics_calculation(self):
        """Test metrics calculation profiling."""
        # Mock calculator
        mock_calculator = Mock()
        mock_calculator.calculate_all_metrics.return_value = {
            'psnr': 30.5, 'ssim': 0.95, 'mse': 0.01
        }
        
        original = np.random.rand(20, 20)
        compressed = np.random.rand(20, 20)
        k = 10
        
        result, perf_result = self.profiler.profile_metrics_calculation(
            mock_calculator, original, compressed, k
        )
        
        # Check that calculator was called correctly
        mock_calculator.calculate_all_metrics.assert_called_once_with(original, compressed, k)
        
        # Check performance result
        assert perf_result.operation_name == 'metrics_calculation'
        assert perf_result.parameters['k'] == k
        assert perf_result.parameters['image_shape'] == original.shape
        assert 'metrics_calculated' in perf_result.parameters
    
    def test_get_summary_stats_empty(self):
        """Test summary statistics with no results."""
        stats = self.profiler.get_summary_stats()
        assert stats == {}
        
        stats_filtered = self.profiler.get_summary_stats('nonexistent_operation')
        assert stats_filtered == {}
    
    def test_get_summary_stats_with_data(self):
        """Test summary statistics calculation."""
        # Add some mock results
        for i in range(5):
            def dummy_func():
                time.sleep(0.001 * (i + 1))  # Variable timing
                return i
            
            self.profiler.profile_function(dummy_func, operation_name='test_op')
        
        stats = self.profiler.get_summary_stats('test_op')
        
        assert stats['operation_name'] == 'test_op'
        assert stats['total_operations'] == 5
        assert 'timing_stats' in stats
        assert 'memory_stats' in stats
        
        timing_stats = stats['timing_stats']
        assert 'mean_time' in timing_stats
        assert 'median_time' in timing_stats
        assert 'min_time' in timing_stats
        assert 'max_time' in timing_stats
        assert 'std_time' in timing_stats
        assert 'total_time' in timing_stats
        
        # Check that timing makes sense
        assert timing_stats['min_time'] > 0
        assert timing_stats['max_time'] >= timing_stats['min_time']
        assert timing_stats['total_time'] >= timing_stats['max_time']
    
    def test_save_results_json(self):
        """Test saving results to JSON format."""
        # Add a test result
        def test_func():
            return 42
        
        self.profiler.profile_function(test_func, operation_name='test_save')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_results.json'
            self.profiler.save_results(filepath, format='json')
            
            # Check file was created
            assert filepath.exists()
            
            # Check content
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'results' in data
            assert data['metadata']['total_results'] == 1
            assert len(data['results']) == 1
            assert data['results'][0]['operation_name'] == 'test_save'
    
    def test_save_results_csv(self):
        """Test saving results to CSV format."""
        # Add a test result
        def test_func():
            return 42
        
        parameters = {'test_param': 'test_value'}
        self.profiler.profile_function(
            test_func, 
            operation_name='test_save_csv',
            parameters=parameters
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_results.csv'
            self.profiler.save_results(filepath, format='csv')
            
            # Check file was created
            assert filepath.exists()
            
            # Check content (basic check)
            content = filepath.read_text()
            assert 'operation_name' in content
            assert 'test_save_csv' in content
            assert 'param_test_param' in content
    
    def test_save_results_invalid_format(self):
        """Test error handling for invalid save format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = Path(temp_dir) / 'test_results.txt'
            
            with pytest.raises(ValueError, match="Unsupported format"):
                self.profiler.save_results(filepath, format='txt')
    
    def test_clear_results(self):
        """Test clearing stored results."""
        # Add some results
        def test_func():
            return 1
        
        self.profiler.profile_function(test_func)
        self.profiler.profile_function(test_func)
        
        assert len(self.profiler.results) == 2
        
        self.profiler.clear_results()
        assert len(self.profiler.results) == 0
    
    def test_get_operation_names(self):
        """Test getting unique operation names."""
        def func1():
            return 1
        def func2():
            return 2
        
        # Initially empty
        assert self.profiler.get_operation_names() == []
        
        # Add operations
        self.profiler.profile_function(func1, operation_name='op1')
        self.profiler.profile_function(func2, operation_name='op2')
        self.profiler.profile_function(func1, operation_name='op1')  # Duplicate
        
        operation_names = self.profiler.get_operation_names()
        assert set(operation_names) == {'op1', 'op2'}
        assert len(operation_names) == 2
    
    def test_benchmark_compression_performance(self):
        """Test comprehensive compression benchmarking."""
        # Mock compressor
        mock_compressor = Mock()
        mock_compressor.compress_image.return_value = (np.random.rand(10, 10), {'ratio': 2.0})
        
        # Test data
        images = [np.random.rand(10, 10), np.random.rand(20, 20)]
        k_values = [2, 5]
        
        benchmark_results = self.profiler.benchmark_compression_performance(
            mock_compressor, images, k_values
        )
        
        # Check structure
        assert 'total_operations' in benchmark_results
        assert 'by_k_value' in benchmark_results
        assert 'by_image_size' in benchmark_results
        assert 'summary' in benchmark_results
        
        # Check counts
        expected_operations = len(images) * len(k_values)
        assert benchmark_results['total_operations'] == expected_operations
        
        # Check that compressor was called correctly
        assert mock_compressor.compress_image.call_count == expected_operations
        
        # Check groupings
        assert set(benchmark_results['by_k_value'].keys()) == set(k_values)
        assert '10x10' in benchmark_results['by_image_size']
        assert '20x20' in benchmark_results['by_image_size']
    
    @patch('psutil.Process')
    def test_memory_usage_tracking(self, mock_process_class):
        """Test memory usage measurement."""
        # Mock memory info
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100 MB in bytes
        mock_process.memory_info.return_value = mock_memory_info
        mock_process_class.return_value = mock_process
        
        profiler = PerformanceProfiler()
        memory_mb = profiler._get_memory_usage()
        
        assert memory_mb == 100.0  # Should convert to MB
    
    @patch('psutil.Process')
    def test_memory_usage_error_handling(self, mock_process_class):
        """Test memory usage error handling."""
        # Mock process that raises exception
        mock_process = Mock()
        mock_process.memory_info.side_effect = Exception("Memory access error")
        mock_process_class.return_value = mock_process
        
        profiler = PerformanceProfiler()
        memory_mb = profiler._get_memory_usage()
        
        assert memory_mb == 0.0  # Should return 0 on error
    
    def test_performance_result_to_dict(self):
        """Test PerformanceResult serialization."""
        from datetime import datetime
        
        result = PerformanceResult(
            operation_name='test_op',
            execution_time=1.5,
            memory_before=100.0,
            memory_after=110.0,
            memory_peak=115.0,
            memory_delta=10.0,
            timestamp=datetime(2023, 1, 1, 12, 0, 0),
            parameters={'k': 5},
            additional_metrics={'custom': 'value'}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['operation_name'] == 'test_op'
        assert result_dict['execution_time'] == 1.5
        assert result_dict['memory_delta'] == 10.0
        assert result_dict['timestamp'] == '2023-01-01T12:00:00'
        assert result_dict['parameters'] == {'k': 5}
        assert result_dict['additional_metrics'] == {'custom': 'value'}
    
    def test_timing_accuracy(self):
        """Test timing measurement accuracy."""
        def timed_function():
            time.sleep(0.05)  # 50ms delay
            return True
        
        result, perf_result = self.profiler.profile_function(timed_function)
        
        # Should measure approximately 50ms (allow some tolerance)
        assert 0.04 < perf_result.execution_time < 0.1
        assert result is True
    
    def test_concurrent_profiling(self):
        """Test profiling multiple operations concurrently."""
        def quick_function(delay):
            time.sleep(delay)
            return delay
        
        # Profile multiple operations
        results = []
        for i in range(3):
            delay = 0.01 * (i + 1)
            result, perf_result = self.profiler.profile_function(
                quick_function, delay, operation_name=f'concurrent_op_{i}'
            )
            results.append((result, perf_result))
        
        # Check that all operations were recorded
        assert len(self.profiler.results) == 3
        
        # Check that timing increases with delay
        times = [perf.execution_time for _, perf in results]
        assert times[0] < times[1] < times[2]
    
    def test_large_dataset_profiling(self):
        """Test profiling with large datasets."""
        def process_large_array():
            # Create and process a moderately large array
            arr = np.random.rand(1000, 1000)
            return np.sum(arr)
        
        result, perf_result = self.profiler.profile_function(process_large_array)
        
        # Should handle large operations
        assert isinstance(result, float)
        assert perf_result.execution_time > 0
        assert perf_result.memory_delta >= 0  # May use additional memory
    
    def test_profiler_state_management(self):
        """Test profiler state management and cleanup."""
        # Add some results
        def dummy_func():
            return 1
        
        for i in range(5):
            self.profiler.profile_function(dummy_func, operation_name=f'test_{i}')
        
        assert len(self.profiler.results) == 5
        
        # Test filtering by operation name
        stats = self.profiler.get_summary_stats('test_0')
        assert stats['total_operations'] == 1
        
        # Test clearing results
        self.profiler.clear_results()
        assert len(self.profiler.results) == 0
        
        # Test getting operation names after clear
        assert self.profiler.get_operation_names() == []