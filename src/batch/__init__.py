"""
Batch processing and experimentation framework for SVD image compression.

This module provides tools for running systematic experiments across
multiple datasets, images, and compression parameters with parallel processing
support and comprehensive result management.
"""

from .experiment_runner import ExperimentRunner, ExperimentConfig
from .result_manager import ResultManager

__all__ = ['ExperimentRunner', 'ExperimentConfig', 'ResultManager']