"""
Performance optimization utilities for the SVD image compression webapp.
Includes client-side caching, image processing optimization, and memory management.
"""

import streamlit as st
import numpy as np
from PIL import Image
import hashlib
import pickle
import time
import gc
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import threading
import queue
import logging
from functools import wraps, lru_cache
import psutil
import os


class ImageCache:
    """Client-side image caching system for better performance."""
    
    def __init__(self, max_cache_size_mb: int = 100, max_entries: int = 50):
        """
        Initialize the image cache.
        
        Args:
            max_cache_size_mb: Maximum cache size in megabytes
            max_entries: Maximum number of cached entries
        """
        self.max_cache_size = max_cache_size_mb * 1024 * 1024  # Convert to bytes
        self.max_entries = max_entries
        self.cache_dir = Path("temp/image_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize session state for cache metadata
        if 'image_cache_metadata' not in st.session_state:
            st.session_state.image_cache_metadata = {
                'entries': {},
                'access_times': {},
                'total_size': 0
            }
    
    def _generate_cache_key(self, image_data: np.ndarray, k_value: int, mode: str) -> str:
        """Generate a unique cache key for the given parameters."""
        # Create hash from image data and parameters
        image_hash = hashlib.md5(image_data.tobytes()).hexdigest()[:16]
        param_hash = hashlib.md5(f"{k_value}_{mode}".encode()).hexdigest()[:8]
        return f"{image_hash}_{param_hash}"
    
    def get(self, image_data: np.ndarray, k_value: int, mode: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached compression result.
        
        Args:
            image_data: Original image array
            k_value: Compression k value
            mode: Processing mode (RGB/Grayscale)
            
        Returns:
            Cached result dictionary or None if not found
        """
        cache_key = self._generate_cache_key(image_data, k_value, mode)
        cache_metadata = st.session_state.image_cache_metadata
        
        if cache_key in cache_metadata['entries']:
            # Update access time
            cache_metadata['access_times'][cache_key] = time.time()
            
            # Load from disk
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    return result
                except Exception as e:
                    logging.warning(f"Failed to load cache entry {cache_key}: {e}")
                    self._remove_entry(cache_key)
        
        return None
    
    def put(self, image_data: np.ndarray, k_value: int, mode: str, result: Dict[str, Any]):
        """
        Store compression result in cache.
        
        Args:
            image_data: Original image array
            k_value: Compression k value
            mode: Processing mode
            result: Compression result to cache
        """
        cache_key = self._generate_cache_key(image_data, k_value, mode)
        cache_metadata = st.session_state.image_cache_metadata
        
        try:
            # Serialize result
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            
            # Calculate file size
            file_size = cache_file.stat().st_size
            
            # Update metadata
            cache_metadata['entries'][cache_key] = {
                'size': file_size,
                'created': time.time()
            }
            cache_metadata['access_times'][cache_key] = time.time()
            cache_metadata['total_size'] += file_size
            
            # Cleanup if necessary
            self._cleanup_cache()
            
        except Exception as e:
            logging.warning(f"Failed to cache result for {cache_key}: {e}")
    
    def _remove_entry(self, cache_key: str):
        """Remove a cache entry."""
        cache_metadata = st.session_state.image_cache_metadata
        
        if cache_key in cache_metadata['entries']:
            # Remove file
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    file_size = cache_file.stat().st_size
                    cache_file.unlink()
                    cache_metadata['total_size'] -= file_size
                except Exception as e:
                    logging.warning(f"Failed to remove cache file {cache_key}: {e}")
            
            # Remove metadata
            cache_metadata['entries'].pop(cache_key, None)
            cache_metadata['access_times'].pop(cache_key, None)
    
    def _cleanup_cache(self):
        """Clean up cache based on size and entry limits."""
        cache_metadata = st.session_state.image_cache_metadata
        
        # Remove entries if over limits
        while (len(cache_metadata['entries']) > self.max_entries or 
               cache_metadata['total_size'] > self.max_cache_size):
            
            if not cache_metadata['access_times']:
                break
            
            # Remove least recently used entry
            oldest_key = min(cache_metadata['access_times'], 
                           key=cache_metadata['access_times'].get)
            self._remove_entry(oldest_key)
    
    def clear(self):
        """Clear all cache entries."""
        cache_metadata = st.session_state.image_cache_metadata
        
        # Remove all files
        for cache_key in list(cache_metadata['entries'].keys()):
            self._remove_entry(cache_key)
        
        # Reset metadata
        st.session_state.image_cache_metadata = {
            'entries': {},
            'access_times': {},
            'total_size': 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_metadata = st.session_state.image_cache_metadata
        
        return {
            'entries': len(cache_metadata['entries']),
            'total_size_mb': cache_metadata['total_size'] / (1024 * 1024),
            'max_size_mb': self.max_cache_size / (1024 * 1024),
            'max_entries': self.max_entries,
            'hit_rate': getattr(self, '_hit_rate', 0.0)
        }


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.metrics = {
            'processing_times': [],
            'memory_usage': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def start_timing(self) -> float:
        """Start timing an operation."""
        return time.time()
    
    def end_timing(self, start_time: float, operation: str = "operation") -> float:
        """End timing and record the duration."""
        duration = time.time() - start_time
        self.metrics['processing_times'].append({
            'operation': operation,
            'duration': duration,
            'timestamp': time.time()
        })
        return duration
    
    def record_memory_usage(self):
        """Record current memory usage."""
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.metrics['memory_usage'].append({
                'memory_mb': memory_mb,
                'timestamp': time.time()
            })
            return memory_mb
        except Exception:
            return None
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.metrics['cache_misses'] += 1
    
    def get_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        if total == 0:
            return 0.0
        return self.metrics['cache_hits'] / total
    
    def get_average_processing_time(self, operation: str = None) -> float:
        """Get average processing time for operations."""
        times = self.metrics['processing_times']
        if operation:
            times = [t for t in times if t['operation'] == operation]
        
        if not times:
            return 0.0
        
        return sum(t['duration'] for t in times) / len(times)
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0


class ImageProcessor:
    """Optimized image processing with memory management and error handling."""
    
    def __init__(self, cache: ImageCache, monitor: PerformanceMonitor):
        self.cache = cache
        self.monitor = monitor
        self.max_image_size = 2048  # Maximum dimension for processing
        self.memory_threshold_mb = 500  # Memory usage threshold
    
    def optimize_image_for_processing(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize image for processing by resizing if necessary.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (optimized_image, optimization_info)
        """
        original_shape = image.shape
        optimization_info = {
            'was_resized': False,
            'original_shape': original_shape,
            'resize_factor': 1.0,
            'memory_saved_mb': 0
        }
        
        # Check if image is too large
        max_dim = max(image.shape[:2])
        if max_dim > self.max_image_size:
            # Calculate resize factor
            resize_factor = self.max_image_size / max_dim
            new_height = int(image.shape[0] * resize_factor)
            new_width = int(image.shape[1] * resize_factor)
            
            # Resize using PIL for better quality
            if image.ndim == 3:
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                pil_image = Image.fromarray((image * 255).astype(np.uint8), mode='L')
            
            resized_pil = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            optimized_image = np.array(resized_pil) / 255.0
            
            # Calculate memory savings
            original_size = np.prod(original_shape) * 8 / (1024 * 1024)  # 8 bytes per float64
            new_size = np.prod(optimized_image.shape) * 8 / (1024 * 1024)
            
            optimization_info.update({
                'was_resized': True,
                'resize_factor': resize_factor,
                'memory_saved_mb': original_size - new_size,
                'new_shape': optimized_image.shape
            })
            
            return optimized_image, optimization_info
        
        return image, optimization_info
    
    def process_with_fallback(self, image: np.ndarray, k_value: int, mode: str) -> Dict[str, Any]:
        """
        Process image with fallback strategies for large images or memory issues.
        
        Args:
            image: Input image array
            k_value: Compression k value
            mode: Processing mode
            
        Returns:
            Processing result with error handling
        """
        start_time = self.monitor.start_timing()
        
        try:
            # Check cache first
            cached_result = self.cache.get(image, k_value, mode)
            if cached_result is not None:
                self.monitor.record_cache_hit()
                self.monitor.end_timing(start_time, "cache_hit")
                return cached_result
            
            self.monitor.record_cache_miss()
            
            # Optimize image for processing
            optimized_image, optimization_info = self.optimize_image_for_processing(image)
            
            # Import compression module
            import sys
            from pathlib import Path
            src_path = Path(__file__).parent.parent.parent / "src"
            sys.path.insert(0, str(src_path))
            from compression.svd_compressor import SVDCompressor
            
            # Process with SVD compression
            compressor = SVDCompressor()
            
            # Monitor memory before processing
            memory_before = self.monitor.record_memory_usage()
            
            # Perform compression
            compressed_image, metadata = compressor.compress_image(optimized_image, k_value)
            
            # Monitor memory after processing
            memory_after = self.monitor.record_memory_usage()
            
            # Scale back to original size if image was resized
            if optimization_info['was_resized']:
                original_shape = optimization_info['original_shape']
                if len(original_shape) == 3:
                    pil_compressed = Image.fromarray((compressed_image * 255).astype(np.uint8))
                    pil_scaled = pil_compressed.resize(
                        (original_shape[1], original_shape[0]), 
                        Image.Resampling.LANCZOS
                    )
                    compressed_image = np.array(pil_scaled) / 255.0
                else:
                    pil_compressed = Image.fromarray((compressed_image * 255).astype(np.uint8), mode='L')
                    pil_scaled = pil_compressed.resize(
                        (original_shape[1], original_shape[0]), 
                        Image.Resampling.LANCZOS
                    )
                    compressed_image = np.array(pil_scaled) / 255.0
            
            # Create result
            result = {
                'compressed_image': compressed_image,
                'metadata': metadata,
                'optimization_info': optimization_info,
                'processing_time': self.monitor.end_timing(start_time, "compression"),
                'memory_usage': {
                    'before_mb': memory_before,
                    'after_mb': memory_after,
                    'peak_usage_mb': max(memory_before or 0, memory_after or 0)
                },
                'success': True,
                'error': None
            }
            
            # Cache the result
            self.cache.put(image, k_value, mode, result)
            
            # Force garbage collection to free memory
            gc.collect()
            
            return result
            
        except MemoryError as e:
            # Handle memory errors with fallback strategies
            return self._handle_memory_error(image, k_value, mode, str(e))
        
        except Exception as e:
            # Handle other processing errors
            return self._handle_processing_error(image, k_value, mode, str(e))
    
    def _handle_memory_error(self, image: np.ndarray, k_value: int, mode: str, error_msg: str) -> Dict[str, Any]:
        """Handle memory errors with fallback strategies."""
        
        # Try with smaller k value
        fallback_k = min(k_value // 2, 10)
        if fallback_k > 0 and fallback_k != k_value:
            try:
                st.warning(f"Memory error occurred. Trying with reduced k={fallback_k}")
                return self.process_with_fallback(image, fallback_k, mode)
            except Exception:
                pass
        
        # Try with grayscale mode
        if mode != "Grayscale":
            try:
                st.warning("Memory error occurred. Trying with grayscale mode")
                return self.process_with_fallback(image, k_value, "Grayscale")
            except Exception:
                pass
        
        # Return error result
        return {
            'compressed_image': None,
            'metadata': None,
            'optimization_info': None,
            'processing_time': 0,
            'memory_usage': None,
            'success': False,
            'error': f"Memory error: {error_msg}. Try reducing image size or k-value.",
            'fallback_suggestions': [
                "Reduce the k-value",
                "Use grayscale mode",
                "Resize the image to a smaller size",
                "Close other applications to free memory"
            ]
        }
    
    def _handle_processing_error(self, image: np.ndarray, k_value: int, mode: str, error_msg: str) -> Dict[str, Any]:
        """Handle general processing errors."""
        
        return {
            'compressed_image': None,
            'metadata': None,
            'optimization_info': None,
            'processing_time': 0,
            'memory_usage': None,
            'success': False,
            'error': f"Processing error: {error_msg}",
            'fallback_suggestions': [
                "Check if the image is valid",
                "Try a different k-value",
                "Ensure the image is not corrupted",
                "Contact support if the problem persists"
            ]
        }


# Global instances
_image_cache = None
_performance_monitor = None
_image_processor = None


def get_image_cache() -> ImageCache:
    """Get global image cache instance."""
    global _image_cache
    if _image_cache is None:
        _image_cache = ImageCache()
    return _image_cache


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_image_processor() -> ImageProcessor:
    """Get global image processor instance."""
    global _image_processor
    if _image_processor is None:
        cache = get_image_cache()
        monitor = get_performance_monitor()
        _image_processor = ImageProcessor(cache, monitor)
    return _image_processor


def performance_decorator(operation_name: str):
    """Decorator to monitor performance of functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            start_time = monitor.start_timing()
            try:
                result = func(*args, **kwargs)
                monitor.end_timing(start_time, operation_name)
                return result
            except Exception as e:
                monitor.end_timing(start_time, f"{operation_name}_error")
                raise
        return wrapper
    return decorator


@lru_cache(maxsize=32)
def get_optimal_k_for_size(height: int, width: int, target_compression: float = 0.1) -> int:
    """
    Calculate optimal k value for given image size and target compression ratio.
    
    Args:
        height: Image height
        width: Image width
        target_compression: Target compression ratio (0.1 = 10:1 compression)
        
    Returns:
        Optimal k value
    """
    total_elements = height * width
    target_compressed_elements = int(total_elements * target_compression)
    
    # For SVD: compressed_size = k * (height + width + 1)
    # Solve for k: k = target_compressed_elements / (height + width + 1)
    optimal_k = max(1, target_compressed_elements // (height + width + 1))
    
    # Ensure k doesn't exceed matrix rank
    max_k = min(height, width)
    return min(optimal_k, max_k)


def clear_all_caches():
    """Clear all performance caches."""
    global _image_cache, _performance_monitor, _image_processor
    
    if _image_cache:
        _image_cache.clear()
    
    if _performance_monitor:
        _performance_monitor.metrics = {
            'processing_times': [],
            'memory_usage': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    # Clear LRU cache
    get_optimal_k_for_size.cache_clear()
    
    # Force garbage collection
    gc.collect()