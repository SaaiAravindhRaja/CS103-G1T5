"""
SVD-based image compression module.

This module provides the core functionality for compressing images using
Singular Value Decomposition (SVD). It supports both single-channel and
multi-channel image compression with configurable compression levels.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import warnings


class SVDCompressor:
    """
    A class for performing SVD-based image compression.
    
    This class implements Singular Value Decomposition compression for images,
    supporting both grayscale (single-channel) and RGB (multi-channel) images.
    The compression is achieved by keeping only the top k singular values and
    their corresponding singular vectors.
    """
    
    def __init__(self) -> None:
        """Initialize the SVD compressor."""
        pass
    
    def compress_channel_svd(self, channel: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compress a single channel using SVD decomposition.
        
        This method performs Singular Value Decomposition on a 2D array (image channel)
        and returns the compressed representation using only the top k singular values.
        
        Args:
            channel (np.ndarray): 2D array representing a single image channel.
                                Shape should be (height, width).
            k (int): Number of singular values to keep for compression.
                    Must be positive and <= min(channel.shape).
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - U (np.ndarray): Left singular vectors, shape (height, k)
                - S (np.ndarray): Singular values, shape (k,)
                - Vt (np.ndarray): Right singular vectors (transposed), shape (k, width)
        
        Raises:
            ValueError: If k is invalid (non-positive or exceeds matrix rank).
            TypeError: If channel is not a 2D numpy array.
            
        Example:
            >>> compressor = SVDCompressor()
            >>> channel = np.random.rand(100, 100)
            >>> U, S, Vt = compressor.compress_channel_svd(channel, k=50)
            >>> reconstructed = U @ np.diag(S) @ Vt
        """
        # Input validation
        if not isinstance(channel, np.ndarray):
            raise TypeError("Channel must be a numpy array")
        
        if channel.ndim != 2:
            raise ValueError("Channel must be a 2D array")
        
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        
        # Get matrix dimensions
        m, n = channel.shape
        max_rank = min(m, n)
        
        if k > max_rank:
            warnings.warn(
                f"k={k} exceeds matrix rank {max_rank}. Using k={max_rank} instead.",
                UserWarning
            )
            k = max_rank
        
        # Handle edge cases
        if m == 0 or n == 0:
            raise ValueError("Channel cannot have zero dimensions")
        
        try:
            # Perform SVD decomposition
            # Use full_matrices=False for efficiency when k < min(m,n)
            U_full, S_full, Vt_full = np.linalg.svd(channel, full_matrices=False)
            
            # Extract top k components
            U = U_full[:, :k].copy()
            S = S_full[:k].copy()
            Vt = Vt_full[:k, :].copy()
            
            return U, S, Vt
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD decomposition failed: {str(e)}")
    
    def reconstruct_channel(self, U: np.ndarray, S: np.ndarray, Vt: np.ndarray, 
                           clip_to_unit_range: bool = False) -> np.ndarray:
        """
        Reconstruct a channel from its SVD components.
        
        Args:
            U (np.ndarray): Left singular vectors
            S (np.ndarray): Singular values
            Vt (np.ndarray): Right singular vectors (transposed)
            clip_to_unit_range (bool): Whether to clip values to [0,1] range
        
        Returns:
            np.ndarray: Reconstructed channel
        """
        reconstructed = U @ np.diag(S) @ Vt
        
        if clip_to_unit_range:
            # Clip values to [0,1] range to ensure valid image data
            reconstructed = np.clip(reconstructed, 0.0, 1.0)
        
        return reconstructed
    
    def compress_image(self, image: np.ndarray, k: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compress a full image (grayscale or RGB) using SVD.
        
        This method handles both grayscale and RGB images by applying SVD compression
        to each channel independently. For RGB images, each color channel is processed
        separately and then recombined.
        
        Args:
            image (np.ndarray): Input image array. Can be:
                              - 2D array (height, width) for grayscale
                              - 3D array (height, width, channels) for RGB
                              Values should be in range [0, 1] for best results.
            k (int): Number of singular values to keep for compression.
                    Must be positive and <= min(height, width).
        
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: A tuple containing:
                - reconstructed_image (np.ndarray): Compressed and reconstructed image
                - metadata (Dict): Dictionary containing:
                    - 'compression_ratio': Theoretical compression ratio
                    - 'storage_original': Original storage requirement (elements)
                    - 'storage_compressed': Compressed storage requirement (elements)
                    - 'k_value': The k value used
                    - 'image_shape': Original image shape
                    - 'channels': Number of channels processed
        
        Raises:
            ValueError: If image dimensions are invalid or k is inappropriate.
            TypeError: If image is not a numpy array.
            
        Example:
            >>> compressor = SVDCompressor()
            >>> # Grayscale image
            >>> gray_img = np.random.rand(100, 100)
            >>> compressed, metadata = compressor.compress_image(gray_img, k=50)
            >>> 
            >>> # RGB image
            >>> rgb_img = np.random.rand(100, 100, 3)
            >>> compressed, metadata = compressor.compress_image(rgb_img, k=50)
        """
        # Input validation
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        if image.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")
        
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        
        # Handle different image formats
        # Check if input is in [0,1] range (typical for normalized images)
        is_normalized = (image.min() >= 0.0 and image.max() <= 1.0)
        
        if image.ndim == 2:
            # Grayscale image
            height, width = image.shape
            channels = 1
            
            # Compress single channel
            U, S, Vt = self.compress_channel_svd(image, k)
            reconstructed_image = self.reconstruct_channel(U, S, Vt, clip_to_unit_range=is_normalized)
            
        elif image.ndim == 3:
            # Multi-channel (RGB) image
            height, width, channels = image.shape
            
            if channels > 4:  # Reasonable limit for image channels
                raise ValueError("Too many channels. Expected <= 4 channels (RGBA)")
            
            # Process each channel independently
            reconstructed_channels = []
            
            for c in range(channels):
                channel = image[:, :, c]
                U, S, Vt = self.compress_channel_svd(channel, k)
                reconstructed_channel = self.reconstruct_channel(U, S, Vt, clip_to_unit_range=is_normalized)
                reconstructed_channels.append(reconstructed_channel)
            
            # Recombine channels
            reconstructed_image = np.stack(reconstructed_channels, axis=2)
        
        # Calculate storage requirements and compression ratio
        storage_original = height * width * channels
        storage_compressed = self.storage_estimate(height, width, k) * channels
        compression_ratio = storage_original / storage_compressed if storage_compressed > 0 else float('inf')
        
        # Create metadata dictionary
        metadata = {
            'compression_ratio': compression_ratio,
            'storage_original': storage_original,
            'storage_compressed': storage_compressed,
            'k_value': k,
            'image_shape': image.shape,
            'channels': channels
        }
        
        return reconstructed_image, metadata
    
    def singular_value_spectrum(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the singular value spectrum for analysis.
        
        This method computes the singular values for each channel of an image
        and returns them for analysis purposes. For multi-channel images,
        it returns the singular values from the first channel.
        
        Args:
            image (np.ndarray): Input image array (2D or 3D).
        
        Returns:
            np.ndarray: Array of singular values in descending order.
                       For multi-channel images, returns singular values
                       from the first channel.
        
        Raises:
            ValueError: If image dimensions are invalid.
            TypeError: If image is not a numpy array.
            
        Example:
            >>> compressor = SVDCompressor()
            >>> image = np.random.rand(100, 100)
            >>> singular_values = compressor.singular_value_spectrum(image)
            >>> print(f"Number of singular values: {len(singular_values)}")
        """
        # Input validation
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy array")
        
        if image.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (RGB)")
        
        # Extract first channel for analysis
        if image.ndim == 2:
            channel = image
        else:
            channel = image[:, :, 0]  # Use first channel for multi-channel images
        
        # Handle edge cases
        if channel.size == 0:
            return np.array([])
        
        try:
            # Compute full SVD to get all singular values
            _, S, _ = np.linalg.svd(channel, full_matrices=False)
            return S
            
        except np.linalg.LinAlgError as e:
            raise ValueError(f"SVD computation failed: {str(e)}")
    
    def storage_estimate(self, m: int, n: int, k: int) -> int:
        """
        Calculate theoretical storage requirements for SVD compression.
        
        For an m×n matrix compressed with k singular values, the storage
        requirement is: m*k + k + k*n = k*(m + n + 1)
        
        This represents:
        - m*k elements for U matrix (m×k)
        - k elements for singular values vector
        - k*n elements for Vt matrix (k×n)
        
        Args:
            m (int): Number of rows (height)
            n (int): Number of columns (width)
            k (int): Number of singular values kept
        
        Returns:
            int: Total number of elements needed to store compressed representation
        
        Raises:
            ValueError: If any parameter is non-positive.
            
        Example:
            >>> compressor = SVDCompressor()
            >>> storage = compressor.storage_estimate(100, 100, 50)
            >>> print(f"Storage for 100x100 image with k=50: {storage} elements")
        """
        # Input validation
        if not all(isinstance(x, int) and x > 0 for x in [m, n, k]):
            raise ValueError("All parameters (m, n, k) must be positive integers")
        
        # Calculate storage: U (m×k) + S (k) + Vt (k×n)
        storage = m * k + k + k * n
        return storage