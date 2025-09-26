"""
Unit tests for MetricsCalculator class.

Tests all quality metrics calculations with known reference values
and edge cases to ensure accuracy and robustness.
"""

import pytest
import numpy as np
from src.evaluation.metrics_calculator import MetricsCalculator


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MetricsCalculator()
        
        # Create test images with known properties
        self.test_image_gray = np.array([
            [0.0, 0.5, 1.0],
            [0.25, 0.75, 0.5],
            [1.0, 0.0, 0.25]
        ])
        
        self.test_image_rgb = np.stack([
            self.test_image_gray,
            self.test_image_gray * 0.8,
            self.test_image_gray * 0.6
        ], axis=2)
        
        # Perfect reconstruction (identical images)
        self.identical_image = self.test_image_gray.copy()
        
        # Slightly modified image for known MSE
        self.modified_image = self.test_image_gray + 0.1
        np.clip(self.modified_image, 0, 1, out=self.modified_image)
    
    def test_calculate_mse_identical_images(self):
        """Test MSE calculation with identical images."""
        mse = self.calculator.calculate_mse(self.test_image_gray, self.identical_image)
        assert mse == 0.0, "MSE should be 0 for identical images"
    
    def test_calculate_mse_known_difference(self):
        """Test MSE calculation with known difference."""
        # Create images with known MSE
        original = np.array([[0.0, 0.5], [1.0, 0.25]])
        modified = np.array([[0.1, 0.6], [0.9, 0.35]])
        
        expected_mse = np.mean((original - modified) ** 2)
        calculated_mse = self.calculator.calculate_mse(original, modified)
        
        assert abs(calculated_mse - expected_mse) < 1e-10, f"Expected MSE {expected_mse}, got {calculated_mse}"
    
    def test_calculate_mse_shape_mismatch(self):
        """Test MSE calculation with mismatched shapes."""
        image1 = np.array([[0.0, 0.5]])
        image2 = np.array([[0.0], [0.5]])
        
        with pytest.raises(ValueError, match="Image shapes must match"):
            self.calculator.calculate_mse(image1, image2)
    
    def test_calculate_psnr_identical_images(self):
        """Test PSNR calculation with identical images."""
        psnr = self.calculator.calculate_psnr(self.test_image_gray, self.identical_image)
        assert psnr == float('inf'), "PSNR should be infinite for identical images"
    
    def test_calculate_psnr_known_mse(self):
        """Test PSNR calculation with known MSE."""
        # Create images with MSE = 0.01
        original = np.array([[0.0, 1.0]])
        modified = np.array([[0.1, 0.9]])  # MSE = (0.1^2 + 0.1^2) / 2 = 0.01
        
        expected_psnr = 20 * np.log10(1.0 / np.sqrt(0.01))  # Should be 20 dB
        calculated_psnr = self.calculator.calculate_psnr(original, modified)
        
        assert abs(calculated_psnr - expected_psnr) < 1e-10, f"Expected PSNR {expected_psnr}, got {calculated_psnr}"
    
    def test_calculate_psnr_invalid_range(self):
        """Test PSNR calculation with invalid pixel ranges."""
        invalid_image = np.array([[0.0, 1.5]])  # Values > 1
        valid_image = np.array([[0.0, 1.0]])
        
        with pytest.raises(ValueError, match="values must be in range"):
            self.calculator.calculate_psnr(invalid_image, valid_image)
        
        with pytest.raises(ValueError, match="values must be in range"):
            self.calculator.calculate_psnr(valid_image, invalid_image)
    
    def test_calculate_ssim_identical_images(self):
        """Test SSIM calculation with identical images."""
        # Use larger image for SSIM (minimum 7x7 for default window)
        large_image = np.random.rand(10, 10)
        ssim_value = self.calculator.calculate_ssim(large_image, large_image)
        assert abs(ssim_value - 1.0) < 1e-10, "SSIM should be 1.0 for identical images"
    
    def test_calculate_ssim_grayscale(self):
        """Test SSIM calculation for grayscale images."""
        # Create larger test images
        original = np.random.rand(20, 20)
        # Add small amount of noise
        noisy = original + np.random.normal(0, 0.01, original.shape)
        np.clip(noisy, 0, 1, out=noisy)
        
        ssim_value = self.calculator.calculate_ssim(original, noisy)
        assert 0.8 < ssim_value < 1.0, f"SSIM should be high for low noise, got {ssim_value}"
    
    def test_calculate_ssim_rgb(self):
        """Test SSIM calculation for RGB images."""
        # Create larger RGB test images
        original = np.random.rand(20, 20, 3)
        # Add small amount of noise
        noisy = original + np.random.normal(0, 0.01, original.shape)
        np.clip(noisy, 0, 1, out=noisy)
        
        ssim_value = self.calculator.calculate_ssim(original, noisy)
        assert 0.8 < ssim_value < 1.0, f"SSIM should be high for low noise, got {ssim_value}"
    
    def test_calculate_ssim_invalid_range(self):
        """Test SSIM calculation with invalid pixel ranges."""
        large_image = np.random.rand(10, 10)
        invalid_image = large_image + 1.5  # Values > 1
        
        with pytest.raises(ValueError, match="values must be in range"):
            self.calculator.calculate_ssim(invalid_image, large_image)
    
    def test_calculate_compression_ratio_grayscale(self):
        """Test compression ratio calculation for grayscale images."""
        # 100x100 grayscale image with k=10
        shape = (100, 100)
        k = 10
        
        expected_ratio = (100 * 100) / (10 * (100 + 1 + 100))  # 10000 / 2010 ≈ 4.975
        calculated_ratio = self.calculator.calculate_compression_ratio(shape, k)
        
        assert abs(calculated_ratio - expected_ratio) < 1e-10, f"Expected ratio {expected_ratio}, got {calculated_ratio}"
    
    def test_calculate_compression_ratio_rgb(self):
        """Test compression ratio calculation for RGB images."""
        # 100x100 RGB image with k=10
        shape = (100, 100, 3)
        k = 10
        
        expected_ratio = (100 * 100 * 3) / (3 * 10 * (100 + 1 + 100))  # 30000 / 6030 ≈ 4.975
        calculated_ratio = self.calculator.calculate_compression_ratio(shape, k)
        
        assert abs(calculated_ratio - expected_ratio) < 1e-10, f"Expected ratio {expected_ratio}, got {calculated_ratio}"
    
    def test_calculate_compression_ratio_invalid_k(self):
        """Test compression ratio calculation with invalid k values."""
        shape = (10, 20)  # min dimension is 10
        
        # k too large
        with pytest.raises(ValueError, match="k must be between 1 and 10"):
            self.calculator.calculate_compression_ratio(shape, 15)
        
        # k too small
        with pytest.raises(ValueError, match="k must be between 1 and 10"):
            self.calculator.calculate_compression_ratio(shape, 0)
    
    def test_calculate_compression_ratio_invalid_shape(self):
        """Test compression ratio calculation with invalid image shapes."""
        invalid_shape = (10,)  # 1D array
        
        with pytest.raises(ValueError, match="Unsupported image shape"):
            self.calculator.calculate_compression_ratio(invalid_shape, 5)
    
    def test_calculate_storage_estimate(self):
        """Test storage estimate calculations."""
        height, width, k = 100, 80, 10
        
        # Grayscale
        original, compressed = self.calculator.calculate_storage_estimate(height, width, k, channels=1)
        assert original == 100 * 80 * 1
        assert compressed == 1 * 10 * (100 + 1 + 80)
        
        # RGB
        original_rgb, compressed_rgb = self.calculator.calculate_storage_estimate(height, width, k, channels=3)
        assert original_rgb == 100 * 80 * 3
        assert compressed_rgb == 3 * 10 * (100 + 1 + 80)
    
    def test_calculate_all_metrics(self):
        """Test calculation of all metrics together."""
        # Use larger images for SSIM
        original = np.random.rand(50, 50)
        # Create compressed version with known properties
        compressed = original + np.random.normal(0, 0.05, original.shape)
        np.clip(compressed, 0, 1, out=compressed)
        
        k = 20
        metrics = self.calculator.calculate_all_metrics(original, compressed, k)
        
        # Check that all expected metrics are present
        expected_keys = ['psnr', 'ssim', 'mse', 'compression_ratio', 
                        'original_storage', 'compressed_storage', 'k_value']
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
        
        # Check value ranges
        assert metrics['psnr'] > 0, "PSNR should be positive"
        assert 0 <= metrics['ssim'] <= 1, "SSIM should be between 0 and 1"
        assert metrics['mse'] >= 0, "MSE should be non-negative"
        assert metrics['compression_ratio'] > 0, "Compression ratio should be positive"
        assert metrics['k_value'] == k, "k_value should match input"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Small images (3x3 minimum for SSIM)
        small_image = np.random.rand(3, 3)
        metrics = self.calculator.calculate_all_metrics(small_image, small_image, 1)
        assert metrics['psnr'] == float('inf')
        assert metrics['mse'] == 0.0
        assert metrics['ssim'] == 1.0  # Identical images should have SSIM = 1
        
        # Maximum k value
        image_10x5 = np.random.rand(10, 5)
        max_k = min(10, 5)  # Should be 5
        metrics = self.calculator.calculate_all_metrics(image_10x5, image_10x5, max_k)
        assert metrics['k_value'] == max_k
    
    def test_numerical_precision(self):
        """Test numerical precision with very small differences."""
        original = np.ones((20, 20)) * 0.5
        # Very small difference
        compressed = original + 1e-10
        
        metrics = self.calculator.calculate_all_metrics(original, compressed, 10)
        
        # Should handle very small differences gracefully
        assert metrics['psnr'] > 100, "PSNR should be very high for tiny differences"
        assert metrics['ssim'] > 0.99, "SSIM should be very close to 1 for tiny differences"
    
    def test_extreme_image_values(self):
        """Test metrics with extreme image values."""
        # Test with all zeros
        zeros = np.zeros((10, 10))
        ones = np.ones((10, 10))
        
        mse = self.calculator.calculate_mse(zeros, ones)
        assert mse == 1.0  # Maximum possible MSE for [0,1] range
        
        psnr = self.calculator.calculate_psnr(zeros, ones)
        assert psnr == 0.0  # Minimum PSNR for maximum MSE
        
        # Test with very small differences
        almost_identical = zeros + 1e-10
        mse_small = self.calculator.calculate_mse(zeros, almost_identical)
        assert mse_small < 1e-15
        
        psnr_small = self.calculator.calculate_psnr(zeros, almost_identical)
        assert psnr_small > 100  # Very high PSNR for tiny differences
    
    def test_ssim_edge_cases(self):
        """Test SSIM with edge cases."""
        # Test with constant images
        constant_image = np.ones((20, 20)) * 0.5
        ssim_constant = self.calculator.calculate_ssim(constant_image, constant_image)
        assert ssim_constant == 1.0
        
        # Test with very different images
        black_image = np.zeros((20, 20))
        white_image = np.ones((20, 20))
        ssim_different = self.calculator.calculate_ssim(black_image, white_image)
        assert ssim_different < 0.5  # Should be low for very different images
    
    def test_compression_ratio_edge_cases(self):
        """Test compression ratio with edge cases."""
        # Test with k=1 (maximum compression)
        shape = (100, 100)
        ratio_max = self.calculator.calculate_compression_ratio(shape, 1)
        expected_max = (100 * 100) / (1 * (100 + 1 + 100))
        assert abs(ratio_max - expected_max) < 1e-10
        
        # Test with k=min(dimensions) (no compression benefit)
        ratio_min = self.calculator.calculate_compression_ratio(shape, 100)
        expected_min = (100 * 100) / (100 * (100 + 1 + 100))
        assert abs(ratio_min - expected_min) < 1e-10
        assert ratio_min < 1.0  # Should be less than 1 (SVD overhead)
    
    def test_metrics_with_different_image_sizes(self):
        """Test metrics calculation with various image sizes."""
        sizes = [(10, 10), (50, 30), (100, 100), (200, 150)]
        
        for height, width in sizes:
            original = np.random.rand(height, width)
            # Add small amount of noise
            compressed = original + np.random.normal(0, 0.01, original.shape)
            np.clip(compressed, 0, 1, out=compressed)
            
            # All metrics should work regardless of size
            mse = self.calculator.calculate_mse(original, compressed)
            psnr = self.calculator.calculate_psnr(original, compressed)
            ssim_val = self.calculator.calculate_ssim(original, compressed)
            
            assert mse >= 0
            assert psnr > 0
            assert 0 <= ssim_val <= 1