"""
Performance profiler for image compression operations.

This module provides comprehensive performance monitoring including
timing, memory usage tracking, and structured result storage.
"""

import time
import psutil
import logging
from typing import Callable, Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceResult:
    """Container for performance measurement results."""
    operation_name: str
    execution_time: float  # seconds
    memory_before: float  # MB
    memory_after: float  # MB
    memory_peak: float  # MB
    memory_delta: float  # MB
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_name': self.operation_name,
            'execution_time': self.execution_time,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_peak': self.memory_peak,
            'memory_delta': self.memory_delta,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'additional_metrics': self.additional_metrics
        }


class PerformanceProfiler:
    """
    Profiler for measuring performance of image compression operations.
    
    Provides timing, memory usage tracking, and structured result storage
    for analyzing compression algorithm performance.
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Initialize the performance profiler.
        
        Args:
            enable_logging: Whether to enable automatic logging of results
        """
        self.enable_logging = enable_logging
        self.results: List[PerformanceResult] = []
        self._process = psutil.Process()
        
    def profile_function(self, func: Callable, *args, operation_name: str = None,
                        parameters: Dict[str, Any] = None, **kwargs) -> Tuple[Any, PerformanceResult]:
        """
        Profile a function call with timing and memory tracking.
        
        Args:
            func: Function to profile
            *args: Positional arguments for the function
            operation_name: Name for the operation (defaults to function name)
            parameters: Dictionary of parameters to store with results
            **kwargs: Keyword arguments for the function
            
        Returns:
            Tuple of (function_result, performance_result)
        """
        if operation_name is None:
            operation_name = func.__name__
        
        if parameters is None:
            parameters = {}
        
        # Record initial memory usage
        memory_before = self._get_memory_usage()
        
        # Start timing
        start_time = time.perf_counter()
        timestamp = datetime.now()
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Record end time and memory
            end_time = time.perf_counter()
            memory_after = self._get_memory_usage()
            
            # Calculate metrics
            execution_time = end_time - start_time
            memory_delta = memory_after - memory_before
            memory_peak = max(memory_before, memory_after)  # Simplified peak detection
            
            # Create performance result
            perf_result = PerformanceResult(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
                memory_delta=memory_delta,
                timestamp=timestamp,
                parameters=parameters.copy()
            )
            
            # Store result
            self.results.append(perf_result)
            
            # Log if enabled
            if self.enable_logging:
                logger.info(f"Profiled {operation_name}: {execution_time:.4f}s, "
                           f"Memory: {memory_before:.1f}MB -> {memory_after:.1f}MB "
                           f"(Δ{memory_delta:+.1f}MB)")
            
            return result, perf_result
            
        except Exception as e:
            logger.error(f"Error profiling {operation_name}: {e}")
            raise
    
    def profile_compression(self, compressor, image: np.ndarray, k: int) -> Tuple[Any, PerformanceResult]:
        """
        Profile image compression operation with detailed metrics.
        
        Args:
            compressor: Compression object with compress_image method
            image: Input image array
            k: Compression parameter
            
        Returns:
            Tuple of (compression_result, performance_result)
        """
        parameters = {
            'k': k,
            'image_shape': image.shape,
            'image_dtype': str(image.dtype),
            'image_size_mb': image.nbytes / (1024 * 1024)
        }
        
        return self.profile_function(
            compressor.compress_image,
            image, k,
            operation_name='svd_compression',
            parameters=parameters
        )
    
    def profile_metrics_calculation(self, calculator, original: np.ndarray, 
                                  compressed: np.ndarray, k: int) -> Tuple[Any, PerformanceResult]:
        """
        Profile metrics calculation operation.
        
        Args:
            calculator: Metrics calculator object
            original: Original image array
            compressed: Compressed image array
            k: Compression parameter
            
        Returns:
            Tuple of (metrics_result, performance_result)
        """
        parameters = {
            'k': k,
            'image_shape': original.shape,
            'metrics_calculated': ['psnr', 'ssim', 'mse', 'compression_ratio']
        }
        
        return self.profile_function(
            calculator.calculate_all_metrics,
            original, compressed, k,
            operation_name='metrics_calculation',
            parameters=parameters
        )
    
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        try:
            # Get RSS (Resident Set Size) memory usage
            memory_info = self._process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 0.0
    
    def get_summary_stats(self, operation_name: str = None) -> Dict[str, Any]:
        """
        Get summary statistics for profiled operations.
        
        Args:
            operation_name: Filter by operation name (optional)
            
        Returns:
            Dictionary with summary statistics
        """
        # Filter results if operation name specified
        if operation_name:
            filtered_results = [r for r in self.results if r.operation_name == operation_name]
        else:
            filtered_results = self.results
        
        if not filtered_results:
            return {}
        
        # Extract timing data
        execution_times = [r.execution_time for r in filtered_results]
        memory_deltas = [r.memory_delta for r in filtered_results]
        
        summary = {
            'operation_name': operation_name or 'all_operations',
            'total_operations': len(filtered_results),
            'timing_stats': {
                'mean_time': np.mean(execution_times),
                'median_time': np.median(execution_times),
                'min_time': np.min(execution_times),
                'max_time': np.max(execution_times),
                'std_time': np.std(execution_times),
                'total_time': np.sum(execution_times)
            },
            'memory_stats': {
                'mean_delta': np.mean(memory_deltas),
                'median_delta': np.median(memory_deltas),
                'min_delta': np.min(memory_deltas),
                'max_delta': np.max(memory_deltas),
                'std_delta': np.std(memory_deltas),
                'total_delta': np.sum(memory_deltas)
            }
        }
        
        return summary
    
    def save_results(self, filepath: Path, format: str = 'json') -> None:
        """
        Save profiling results to file.
        
        Args:
            filepath: Path to save results
            format: Output format ('json' or 'csv')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            self._save_json(filepath)
        elif format.lower() == 'csv':
            self._save_csv(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(self.results)} profiling results to {filepath}")
    
    def _save_json(self, filepath: Path) -> None:
        """Save results as JSON."""
        data = {
            'metadata': {
                'total_results': len(self.results),
                'generated_at': datetime.now().isoformat(),
                'profiler_version': '1.0'
            },
            'results': [result.to_dict() for result in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_csv(self, filepath: Path) -> None:
        """Save results as CSV."""
        import pandas as pd
        
        # Convert results to flat dictionary format
        rows = []
        for result in self.results:
            row = {
                'operation_name': result.operation_name,
                'execution_time': result.execution_time,
                'memory_before': result.memory_before,
                'memory_after': result.memory_after,
                'memory_peak': result.memory_peak,
                'memory_delta': result.memory_delta,
                'timestamp': result.timestamp.isoformat()
            }
            
            # Add parameters as separate columns
            for key, value in result.parameters.items():
                row[f'param_{key}'] = value
            
            # Add additional metrics
            for key, value in result.additional_metrics.items():
                row[f'metric_{key}'] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def clear_results(self) -> None:
        """Clear all stored profiling results."""
        self.results.clear()
        logger.info("Cleared all profiling results")
    
    def get_operation_names(self) -> List[str]:
        """Get list of unique operation names in results."""
        return list(set(result.operation_name for result in self.results))
    
    def benchmark_compression_performance(self, compressor, images: List[np.ndarray], 
                                        k_values: List[int]) -> Dict[str, Any]:
        """
        Run comprehensive benchmark of compression performance.
        
        Args:
            compressor: Compression object
            images: List of test images
            k_values: List of k values to test
            
        Returns:
            Dictionary with benchmark results
        """
        benchmark_results = {
            'total_operations': 0,
            'by_k_value': {},
            'by_image_size': {},
            'summary': {}
        }
        
        for i, image in enumerate(images):
            for k in k_values:
                try:
                    # Profile compression
                    _, perf_result = self.profile_compression(compressor, image, k)
                    
                    # Update counters
                    benchmark_results['total_operations'] += 1
                    
                    # Group by k value
                    if k not in benchmark_results['by_k_value']:
                        benchmark_results['by_k_value'][k] = []
                    benchmark_results['by_k_value'][k].append(perf_result.execution_time)
                    
                    # Group by image size
                    size_key = f"{image.shape[0]}x{image.shape[1]}"
                    if size_key not in benchmark_results['by_image_size']:
                        benchmark_results['by_image_size'][size_key] = []
                    benchmark_results['by_image_size'][size_key].append(perf_result.execution_time)
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for image {i}, k={k}: {e}")
        
        # Calculate summary statistics
        benchmark_results['summary'] = self.get_summary_stats('svd_compression')
        
        return benchmark_results