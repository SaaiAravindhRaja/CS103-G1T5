"""
Evaluation module for image compression quality assessment.

This module provides comprehensive evaluation tools including:
- Quality metrics calculation (PSNR, SSIM, MSE, compression ratios)
- Performance profiling and benchmarking
- Structured result storage and analysis
"""

from .metrics_calculator import MetricsCalculator
from .performance_profiler import PerformanceProfiler, PerformanceResult

__all__ = [
    'MetricsCalculator',
    'PerformanceProfiler', 
    'PerformanceResult'
]