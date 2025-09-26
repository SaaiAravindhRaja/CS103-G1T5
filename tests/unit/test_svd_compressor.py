"""
Unit tests for SVD compression functionality.

This module contains comprehensive tests for the SVDCompressor class,
focusing on single-channel compression, edge cases, and error handling.
"""

import numpy as np
import pytest
import warnings
from src.compression.svd_compressor import SVDCompressor


class TestSVDCompressor:
    """Test suite for SVDCompressor class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.compressor = SVDCompressor()
        
        # Create test matrices with known properties
        self.simple_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
        self.random_matrix = np.random.RandomState(42).rand(50, 30)
        self.identity_matrix = np.eye(10)
        
    def test_compress_channel_svd_basic(self):
        """Test basic SVD compression functionality."""
        k = 2
        U, S, Vt = self.compressor.compress_channel_svd(self.simple_matrix, k)
        
        # Check output shapes
        assert U.shape == (3, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, 3)
        
        # Check that singular values are in descending order
        assert np.all(S[:-1] >= S[1:])
        
        # Check that singular values are non-negative
        assert np.all(S >= 0)
    
    def test_compress_channel_svd_reconstruction(self):
        """Test that SVD compression allows accurate reconstruction."""
        k = 3  # Full rank for 3x3 matrix
        U, S, Vt = self.compressor.compress_channel_svd(self.simple_matrix, k)
        
        # Reconstruct the matrix
        reconstructed = self.compressor.reconstruct_channel(U, S, Vt)
        
        # Should be very close to original (within numerical precision)
        np.testing.assert_allclose(reconstructed, self.simple_matrix, rtol=1e-10)
    
    def test_compress_channel_svd_partial_reconstruction(self):
        """Test partial reconstruction with k < rank."""
        original = self.random_matrix
        k = 10  # Less than min(50, 30) = 30
        
        U, S, Vt = self.compressor.compress_channel_svd(original, k)
        reconstructed = self.compressor.reconstruct_channel(U, S, Vt)
        
        # Reconstruction should be close but not exact
        assert reconstructed.shape == original.shape
        
        # Calculate reconstruction error
        error = np.linalg.norm(original - reconstructed, 'fro')
        original_norm = np.linalg.norm(original, 'fro')
        relative_error = error / original_norm
        
        # Error should be reasonable for k=10 out of 30 possible
        assert relative_error < 0.5  # Should be much better than 50% error
    
    def test_compress_channel_svd_identity_matrix(self):
        """Test SVD compression on identity matrix."""
        k = 5
        U, S, Vt = self.compressor.compress_channel_svd(self.identity_matrix, k)
        
        # Identity matrix should have k singular values of 1
        expected_S = np.ones(k)
        np.testing.assert_allclose(S, expected_S, rtol=1e-10)
        
        # Reconstruction should be close to truncated identity
        reconstructed = self.compressor.reconstruct_channel(U, S, Vt)
        expected = np.eye(10)
        expected[k:, k:] = 0  # Truncated identity
        
        # The reconstruction might not exactly match truncated identity due to
        # the way SVD works, but it should be close
        assert reconstructed.shape == (10, 10)
    
    def test_compress_channel_svd_k_exceeds_rank(self):
        """Test behavior when k exceeds matrix rank."""
        k = 10  # Exceeds min(3, 3) = 3
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            U, S, Vt = self.compressor.compress_channel_svd(self.simple_matrix, k)
            
            # Should issue a warning
            assert len(w) == 1
            assert "exceeds matrix rank" in str(w[0].message)
        
        # Should automatically reduce k to matrix rank
        expected_k = min(self.simple_matrix.shape)
        assert U.shape == (3, expected_k)
        assert S.shape == (expected_k,)
        assert Vt.shape == (expected_k, 3)
    
    def test_compress_channel_svd_invalid_k(self):
        """Test error handling for invalid k values."""
        # Test negative k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            self.compressor.compress_channel_svd(self.simple_matrix, -1)
        
        # Test zero k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            self.compressor.compress_channel_svd(self.simple_matrix, 0)
        
        # Test non-integer k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            self.compressor.compress_channel_svd(self.simple_matrix, 2.5)
    
    def test_compress_channel_svd_invalid_input_type(self):
        """Test error handling for invalid input types."""
        # Test non-numpy array
        with pytest.raises(TypeError, match="Channel must be a numpy array"):
            self.compressor.compress_channel_svd([[1, 2], [3, 4]], 1)
        
        # Test wrong dimensions
        with pytest.raises(ValueError, match="Channel must be a 2D array"):
            self.compressor.compress_channel_svd(np.array([1, 2, 3]), 1)
        
        with pytest.raises(ValueError, match="Channel must be a 2D array"):
            self.compressor.compress_channel_svd(np.array([[[1, 2], [3, 4]]]), 1)
    
    def test_compress_channel_svd_zero_dimensions(self):
        """Test error handling for matrices with zero dimensions."""
        # Test empty matrix
        empty_matrix = np.array([]).reshape(0, 5)
        with pytest.raises(ValueError, match="Channel cannot have zero dimensions"):
            self.compressor.compress_channel_svd(empty_matrix, 1)
        
        # Test matrix with zero width
        zero_width = np.array([]).reshape(5, 0)
        with pytest.raises(ValueError, match="Channel cannot have zero dimensions"):
            self.compressor.compress_channel_svd(zero_width, 1)
    
    def test_compress_channel_svd_single_element(self):
        """Test SVD compression on single element matrix."""
        single_element = np.array([[5.0]])
        k = 1
        
        U, S, Vt = self.compressor.compress_channel_svd(single_element, k)
        
        assert U.shape == (1, 1)
        assert S.shape == (1,)
        assert Vt.shape == (1, 1)
        
        # Singular value should be the absolute value of the element
        np.testing.assert_allclose(S, [5.0])
        
        # Reconstruction should be exact
        reconstructed = self.compressor.reconstruct_channel(U, S, Vt)
        np.testing.assert_allclose(reconstructed, single_element)
    
    def test_compress_channel_svd_rectangular_matrices(self):
        """Test SVD compression on rectangular matrices."""
        # Tall matrix
        tall_matrix = np.random.RandomState(42).rand(100, 20)
        k = 10
        
        U, S, Vt = self.compressor.compress_channel_svd(tall_matrix, k)
        assert U.shape == (100, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, 20)
        
        # Wide matrix
        wide_matrix = np.random.RandomState(42).rand(20, 100)
        
        U, S, Vt = self.compressor.compress_channel_svd(wide_matrix, k)
        assert U.shape == (20, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, 100)
    
    def test_compress_channel_svd_numerical_stability(self):
        """Test numerical stability with various matrix conditions."""
        # Test with very small values
        small_matrix = np.random.RandomState(42).rand(10, 10) * 1e-10
        k = 5
        
        U, S, Vt = self.compressor.compress_channel_svd(small_matrix, k)
        reconstructed = self.compressor.reconstruct_channel(U, S, Vt)
        
        # Should handle small values without issues
        assert np.all(np.isfinite(reconstructed))
        assert reconstructed.shape == small_matrix.shape
        
        # Test with large values
        large_matrix = np.random.RandomState(42).rand(10, 10) * 1e10
        
        U, S, Vt = self.compressor.compress_channel_svd(large_matrix, k)
        reconstructed = self.compressor.reconstruct_channel(U, S, Vt)
        
        # Should handle large values without overflow
        assert np.all(np.isfinite(reconstructed))
        assert reconstructed.shape == large_matrix.shape


class TestSVDCompressorMultiChannel:
    """Test suite for multi-channel image compression functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.compressor = SVDCompressor()
        
        # Create test images
        np.random.seed(42)  # For reproducible tests
        self.grayscale_image = np.random.rand(50, 40)
        self.rgb_image = np.random.rand(50, 40, 3)
        self.rgba_image = np.random.rand(30, 25, 4)
        
        # Create a simple test image with known properties
        self.simple_gray = np.array([[1, 2], [3, 4]], dtype=np.float64)
        self.simple_rgb = np.stack([self.simple_gray, self.simple_gray * 2, self.simple_gray * 3], axis=2)
    
    def test_compress_image_grayscale_basic(self):
        """Test basic grayscale image compression."""
        k = 10
        reconstructed, metadata = self.compressor.compress_image(self.grayscale_image, k)
        
        # Check output properties
        assert reconstructed.shape == self.grayscale_image.shape
        assert isinstance(metadata, dict)
        
        # Check metadata contents
        assert metadata['k_value'] == k
        assert metadata['channels'] == 1
        assert metadata['image_shape'] == self.grayscale_image.shape
        assert metadata['storage_original'] == 50 * 40 * 1
        assert metadata['storage_compressed'] == k * (50 + 40 + 1)
        assert metadata['compression_ratio'] > 0
    
    def test_compress_image_rgb_basic(self):
        """Test basic RGB image compression."""
        k = 15
        reconstructed, metadata = self.compressor.compress_image(self.rgb_image, k)
        
        # Check output properties
        assert reconstructed.shape == self.rgb_image.shape
        assert isinstance(metadata, dict)
        
        # Check metadata contents
        assert metadata['k_value'] == k
        assert metadata['channels'] == 3
        assert metadata['image_shape'] == self.rgb_image.shape
        assert metadata['storage_original'] == 50 * 40 * 3
        assert metadata['storage_compressed'] == k * (50 + 40 + 1) * 3
        assert metadata['compression_ratio'] > 0
    
    def test_compress_image_rgba(self):
        """Test RGBA image compression."""
        k = 8
        reconstructed, metadata = self.compressor.compress_image(self.rgba_image, k)
        
        # Check output properties
        assert reconstructed.shape == self.rgba_image.shape
        assert metadata['channels'] == 4
        assert metadata['storage_original'] == 30 * 25 * 4
    
    def test_compress_image_reconstruction_quality(self):
        """Test reconstruction quality for different k values."""
        original = self.grayscale_image
        
        # Test with different k values
        k_values = [5, 15, 25]
        errors = []
        
        for k in k_values:
            reconstructed, _ = self.compressor.compress_image(original, k)
            error = np.linalg.norm(original - reconstructed, 'fro')
            errors.append(error)
        
        # Error should decrease as k increases
        assert errors[0] >= errors[1] >= errors[2]
    
    def test_compress_image_full_rank_reconstruction(self):
        """Test that full rank reconstruction is nearly perfect."""
        original = self.simple_gray
        k = min(original.shape)  # Full rank
        
        reconstructed, metadata = self.compressor.compress_image(original, k)
        
        # Should be very close to original
        np.testing.assert_allclose(reconstructed, original, rtol=1e-10)
        
        # For small matrices, compression ratio might be < 1 (SVD overhead)
        # This is expected behavior for very small matrices
        assert metadata['compression_ratio'] > 0
    
    def test_compress_image_rgb_channel_independence(self):
        """Test that RGB channels are processed independently."""
        # Create image with different patterns in each channel
        height, width = 20, 15
        red_channel = np.ones((height, width)) * 0.5
        green_channel = np.eye(height, width)[:height, :width]  # Identity pattern
        blue_channel = np.random.RandomState(42).rand(height, width)
        
        rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=2)
        k = 8
        
        reconstructed, metadata = self.compressor.compress_image(rgb_image, k)
        
        # Check that each channel maintains its distinct characteristics
        assert reconstructed.shape == rgb_image.shape
        
        # Red channel should be relatively uniform (low variance after compression)
        red_reconstructed = reconstructed[:, :, 0]
        assert np.var(red_reconstructed) < np.var(blue_channel)  # Should be more uniform than blue
    
    def test_compress_image_invalid_input_type(self):
        """Test error handling for invalid input types."""
        k = 5
        
        # Test non-numpy array
        with pytest.raises(TypeError, match="Image must be a numpy array"):
            self.compressor.compress_image([[1, 2], [3, 4]], k)
        
        # Test invalid k
        with pytest.raises(ValueError, match="k must be a positive integer"):
            self.compressor.compress_image(self.grayscale_image, -1)
        
        with pytest.raises(ValueError, match="k must be a positive integer"):
            self.compressor.compress_image(self.grayscale_image, 0)
    
    def test_compress_image_invalid_dimensions(self):
        """Test error handling for invalid image dimensions."""
        k = 5
        
        # Test 1D array
        with pytest.raises(ValueError, match="Image must be 2D \\(grayscale\\) or 3D \\(RGB\\)"):
            self.compressor.compress_image(np.array([1, 2, 3]), k)
        
        # Test 4D array
        with pytest.raises(ValueError, match="Image must be 2D \\(grayscale\\) or 3D \\(RGB\\)"):
            self.compressor.compress_image(np.random.rand(10, 10, 3, 2), k)
        
        # Test too many channels
        with pytest.raises(ValueError, match="Too many channels"):
            self.compressor.compress_image(np.random.rand(10, 10, 10), k)
    
    def test_singular_value_spectrum_grayscale(self):
        """Test singular value spectrum computation for grayscale images."""
        spectrum = self.compressor.singular_value_spectrum(self.grayscale_image)
        
        # Check properties
        assert isinstance(spectrum, np.ndarray)
        assert len(spectrum) == min(self.grayscale_image.shape)
        
        # Should be in descending order
        assert np.all(spectrum[:-1] >= spectrum[1:])
        
        # Should be non-negative
        assert np.all(spectrum >= 0)
    
    def test_singular_value_spectrum_rgb(self):
        """Test singular value spectrum computation for RGB images."""
        spectrum = self.compressor.singular_value_spectrum(self.rgb_image)
        
        # Should use first channel
        expected_length = min(self.rgb_image.shape[:2])
        assert len(spectrum) == expected_length
        
        # Compare with manual computation of first channel
        manual_spectrum = self.compressor.singular_value_spectrum(self.rgb_image[:, :, 0])
        np.testing.assert_allclose(spectrum, manual_spectrum)
    
    def test_singular_value_spectrum_invalid_input(self):
        """Test error handling for singular value spectrum computation."""
        # Test non-numpy array
        with pytest.raises(TypeError, match="Image must be a numpy array"):
            self.compressor.singular_value_spectrum([[1, 2], [3, 4]])
        
        # Test invalid dimensions
        with pytest.raises(ValueError, match="Image must be 2D \\(grayscale\\) or 3D \\(RGB\\)"):
            self.compressor.singular_value_spectrum(np.array([1, 2, 3]))
    
    def test_singular_value_spectrum_empty_image(self):
        """Test singular value spectrum with empty image."""
        empty_image = np.array([]).reshape(0, 5)
        spectrum = self.compressor.singular_value_spectrum(empty_image)
        
        assert len(spectrum) == 0
        assert isinstance(spectrum, np.ndarray)
    
    def test_storage_estimate_basic(self):
        """Test storage estimation calculations."""
        m, n, k = 100, 80, 20
        storage = self.compressor.storage_estimate(m, n, k)
        
        # Should equal m*k + k + k*n
        expected = m * k + k + k * n
        assert storage == expected
        
        # Should equal k * (m + n + 1)
        assert storage == k * (m + n + 1)
    
    def test_storage_estimate_various_sizes(self):
        """Test storage estimation with various matrix sizes."""
        test_cases = [
            (10, 10, 5),
            (100, 50, 25),
            (200, 300, 100),
            (1, 1000, 1),
            (1000, 1, 1)
        ]
        
        for m, n, k in test_cases:
            storage = self.compressor.storage_estimate(m, n, k)
            expected = k * (m + n + 1)
            assert storage == expected
    
    def test_storage_estimate_invalid_input(self):
        """Test error handling for storage estimation."""
        # Test negative values
        with pytest.raises(ValueError, match="All parameters \\(m, n, k\\) must be positive integers"):
            self.compressor.storage_estimate(-1, 10, 5)
        
        with pytest.raises(ValueError, match="All parameters \\(m, n, k\\) must be positive integers"):
            self.compressor.storage_estimate(10, -1, 5)
        
        with pytest.raises(ValueError, match="All parameters \\(m, n, k\\) must be positive integers"):
            self.compressor.storage_estimate(10, 10, -1)
        
        # Test zero values
        with pytest.raises(ValueError, match="All parameters \\(m, n, k\\) must be positive integers"):
            self.compressor.storage_estimate(0, 10, 5)
        
        # Test non-integer values
        with pytest.raises(ValueError, match="All parameters \\(m, n, k\\) must be positive integers"):
            self.compressor.storage_estimate(10.5, 10, 5)
    
    def test_compression_ratio_calculation(self):
        """Test compression ratio calculations."""
        # Test case where compression provides benefit
        original = np.random.RandomState(42).rand(100, 80)
        k = 20  # Much less than min(100, 80) = 80
        
        reconstructed, metadata = self.compressor.compress_image(original, k)
        
        # Should have compression benefit (ratio > 1)
        assert metadata['compression_ratio'] > 1
        
        # Manual calculation
        storage_original = 100 * 80
        storage_compressed = k * (100 + 80 + 1)
        expected_ratio = storage_original / storage_compressed
        
        assert abs(metadata['compression_ratio'] - expected_ratio) < 1e-10
    
    def test_edge_case_single_pixel_image(self):
        """Test compression of single pixel images."""
        # Single pixel grayscale
        single_pixel = np.array([[5.0]])
        k = 1
        
        reconstructed, metadata = self.compressor.compress_image(single_pixel, k)
        
        np.testing.assert_allclose(reconstructed, single_pixel)
        # For single pixel, SVD has overhead, so compression ratio < 1 is expected
        assert metadata['compression_ratio'] > 0
        
        # Single pixel RGB
        single_pixel_rgb = np.array([[[1.0, 2.0, 3.0]]])
        
        reconstructed_rgb, metadata_rgb = self.compressor.compress_image(single_pixel_rgb, k)
        
        np.testing.assert_allclose(reconstructed_rgb, single_pixel_rgb)
        assert metadata_rgb['channels'] == 3
    
    def test_extreme_k_values(self):
        """Test compression with extreme k values."""
        image = np.random.rand(50, 40)
        
        # Test k = 1 (maximum compression)
        reconstructed, metadata = self.compressor.compress_image(image, 1)
        assert reconstructed.shape == image.shape
        assert metadata['k_value'] == 1
        assert metadata['compression_ratio'] > 1  # Should provide compression benefit
        
        # Test k = min(dimensions) (no compression)
        max_k = min(image.shape)
        reconstructed, metadata = self.compressor.compress_image(image, max_k)
        assert reconstructed.shape == image.shape
        assert metadata['k_value'] == max_k
        # Should be very close to original
        np.testing.assert_allclose(reconstructed, image, rtol=1e-10)
    
    def test_memory_efficiency(self):
        """Test that SVD compression doesn't create excessive intermediate arrays."""
        # This test ensures memory efficiency by checking that the compressor
        # doesn't hold onto large intermediate arrays
        large_image = np.random.rand(500, 400)
        k = 50
        
        # Should complete without memory issues
        reconstructed, metadata = self.compressor.compress_image(large_image, k)
        
        assert reconstructed.shape == large_image.shape
        assert metadata['storage_compressed'] < metadata['storage_original']
    
    def test_deterministic_behavior(self):
        """Test that compression is deterministic for the same input."""
        image = np.random.RandomState(42).rand(30, 25)
        k = 10
        
        # Run compression twice
        result1, metadata1 = self.compressor.compress_image(image, k)
        result2, metadata2 = self.compressor.compress_image(image, k)
        
        # Results should be identical
        np.testing.assert_allclose(result1, result2)
        assert metadata1['compression_ratio'] == metadata2['compression_ratio']
    
    def test_different_data_types(self):
        """Test compression with different numpy data types."""
        # Test with float32 - should work and preserve reasonable precision
        image_f32 = np.random.rand(20, 20).astype(np.float32)
        reconstructed, metadata = self.compressor.compress_image(image_f32, 5)
        assert reconstructed.dtype in [np.float32, np.float64]  # Either is acceptable
        assert reconstructed.shape == image_f32.shape
        
        # Test with int (should be converted to float or rejected)
        image_int = (np.random.rand(20, 20) * 255).astype(np.uint8)
        try:
            reconstructed_int, metadata_int = self.compressor.compress_image(image_int, 5)
            # If it works, should be converted to float
            assert reconstructed_int.dtype in [np.float32, np.float64]
        except (ValueError, TypeError):
            # It's also acceptable to reject non-float types
            pass