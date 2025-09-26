"""
Quality metrics calculator for image compression evaluation.

This module provides comprehensive quality metrics for evaluating
image compression performance including PSNR, SSIM, MSE, and compression ratios.
"""

import numpy as np
from typing import Tuple, Union
from skimage.metrics import structural_similarity as ssim
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculator for image quality metrics and compression analysis.
    
    Provides methods to calculate PSNR, SSIM, MSE, and compression ratios
    for evaluating image compression performance.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_psnr(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
        
        Args:
            original: Original image array with values in [0, 1]
            compressed: Compressed image array with values in [0, 1]
            
        Returns:
            PSNR value in decibels (dB)
            
        Raises:
            ValueError: If images have different shapes or invalid pixel ranges
        """
        if original.shape != compressed.shape:
            raise ValueError(f"Image shapes must match: {original.shape} vs {compressed.shape}")
        
        # Validate pixel value ranges
        if not (0 <= original.min() and original.max() <= 1):
            raise ValueError("Original image values must be in range [0, 1]")
        if not (0 <= compressed.min() and compressed.max() <= 1):
            raise ValueError("Compressed image values must be in range [0, 1]")
        
        # Calculate MSE
        mse = self.calculate_mse(original, compressed)
        
        # Handle perfect reconstruction case
        if mse == 0:
            return float('inf')
        
        # PSNR formula: 20 * log10(MAX_I / sqrt(MSE))
        # For normalized images, MAX_I = 1
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        
        return float(psnr)
    
    def calculate_ssim(self, original: np.ndarray, compressed: np.ndarray, 
                      multichannel: bool = None) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two images.
        
        Args:
            original: Original image array with values in [0, 1]
            compressed: Compressed image array with values in [0, 1]
            multichannel: Whether to treat as multichannel image. 
                         If None, auto-detect based on shape.
            
        Returns:
            SSIM value between -1 and 1 (higher is better)
            
        Raises:
            ValueError: If images have different shapes or invalid pixel ranges
        """
        if original.shape != compressed.shape:
            raise ValueError(f"Image shapes must match: {original.shape} vs {compressed.shape}")
        
        # Validate pixel value ranges
        if not (0 <= original.min() and original.max() <= 1):
            raise ValueError("Original image values must be in range [0, 1]")
        if not (0 <= compressed.min() and compressed.max() <= 1):
            raise ValueError("Compressed image values must be in range [0, 1]")
        
        # Auto-detect multichannel if not specified
        if multichannel is None:
            multichannel = len(original.shape) == 3 and original.shape[2] > 1
        
        # Determine appropriate window size based on image dimensions
        min_dim = min(original.shape[:2])
        if min_dim < 7:
            # For very small images, use smaller window size
            win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
            if win_size < 3:
                win_size = 3  # Minimum window size
        else:
            win_size = 7
        
        # Calculate SSIM with appropriate parameters
        if multichannel:
            # For RGB images
            ssim_value = ssim(
                original, compressed,
                multichannel=True,
                channel_axis=2,
                data_range=1.0,
                win_size=win_size
            )
        else:
            # For grayscale images
            ssim_value = ssim(
                original, compressed,
                data_range=1.0,
                win_size=win_size
            )
        
        return float(ssim_value)
    
    def calculate_mse(self, original: np.ndarray, compressed: np.ndarray) -> float:
        """
        Calculate Mean Squared Error (MSE) between two images.
        
        Args:
            original: Original image array
            compressed: Compressed image array
            
        Returns:
            MSE value (lower is better)
            
        Raises:
            ValueError: If images have different shapes
        """
        if original.shape != compressed.shape:
            raise ValueError(f"Image shapes must match: {original.shape} vs {compressed.shape}")
        
        # Calculate MSE
        mse = np.mean((original - compressed) ** 2)
        
        return float(mse)
    
    def calculate_compression_ratio(self, original_shape: Tuple[int, ...], k: int) -> float:
        """
        Calculate theoretical compression ratio for SVD compression.
        
        Args:
            original_shape: Shape of the original image (height, width) or (height, width, channels)
            k: Number of singular values used in compression
            
        Returns:
            Compression ratio (original_size / compressed_size)
            
        Raises:
            ValueError: If k is invalid for the given image dimensions
        """
        if len(original_shape) == 2:
            # Grayscale image
            height, width = original_shape
            channels = 1
        elif len(original_shape) == 3:
            # RGB image
            height, width, channels = original_shape
        else:
            raise ValueError(f"Unsupported image shape: {original_shape}")
        
        # Validate k value
        max_k = min(height, width)
        if k <= 0 or k > max_k:
            raise ValueError(f"k must be between 1 and {max_k}, got {k}")
        
        # Calculate storage requirements
        original_size = height * width * channels
        
        # For each channel: U[:, :k] + S[:k] + Vt[:k, :]
        # Storage per channel: height*k + k + k*width = k*(height + 1 + width)
        compressed_size = channels * k * (height + 1 + width)
        
        # Calculate compression ratio
        compression_ratio = original_size / compressed_size
        
        return float(compression_ratio)
    
    def calculate_storage_estimate(self, height: int, width: int, k: int, 
                                 channels: int = 1) -> Tuple[int, int]:
        """
        Calculate storage estimates for original and compressed images.
        
        Args:
            height: Image height
            width: Image width
            k: Number of singular values
            channels: Number of color channels
            
        Returns:
            Tuple of (original_storage, compressed_storage) in number of elements
        """
        original_storage = height * width * channels
        compressed_storage = channels * k * (height + 1 + width)
        
        return original_storage, compressed_storage
    
    def calculate_all_metrics(self, original: np.ndarray, compressed: np.ndarray, 
                            k: int) -> dict:
        """
        Calculate all quality metrics for a compressed image.
        
        Args:
            original: Original image array with values in [0, 1]
            compressed: Compressed image array with values in [0, 1]
            k: Number of singular values used in compression
            
        Returns:
            Dictionary containing all calculated metrics
        """
        metrics = {}
        
        try:
            metrics['psnr'] = self.calculate_psnr(original, compressed)
            metrics['ssim'] = self.calculate_ssim(original, compressed)
            metrics['mse'] = self.calculate_mse(original, compressed)
            metrics['compression_ratio'] = self.calculate_compression_ratio(original.shape, k)
            
            # Storage estimates
            original_storage, compressed_storage = self.calculate_storage_estimate(
                *original.shape[:2], k, 
                channels=original.shape[2] if len(original.shape) == 3 else 1
            )
            metrics['original_storage'] = original_storage
            metrics['compressed_storage'] = compressed_storage
            metrics['k_value'] = k
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise
        
        return metrics