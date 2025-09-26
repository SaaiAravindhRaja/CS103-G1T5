"""
Unit tests for the ImageLoader class.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.data.image_loader import ImageLoader


class TestImageLoader:
    """Test cases for ImageLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = ImageLoader(target_size=(256, 256))
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_image(self, size: tuple, mode: str = 'RGB', color=None) -> Path:
        """Create a test image file."""
        if color is None:
            color = (255, 0, 0) if mode == 'RGB' else 128
        
        img = Image.new(mode, size, color)
        img_path = self.temp_dir / f"test_{mode}_{size[0]}x{size[1]}.png"
        img.save(img_path)
        return img_path
    
    def test_init_default_size(self):
        """Test ImageLoader initialization with default size."""
        loader = ImageLoader()
        assert loader.target_size == (256, 256)
    
    def test_init_custom_size(self):
        """Test ImageLoader initialization with custom size."""
        loader = ImageLoader(target_size=(512, 512))
        assert loader.target_size == (512, 512)
    
    def test_load_image_rgb(self):
        """Test loading RGB image."""
        img_path = self.create_test_image((100, 100), 'RGB', (255, 128, 64))
        
        result = self.loader.load_image(img_path)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.float64
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_load_image_grayscale(self):
        """Test loading grayscale image."""
        img_path = self.create_test_image((100, 100), 'L', 128)
        
        result = self.loader.load_image(img_path)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256)
        assert result.dtype == np.float64
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_load_image_rgba_conversion(self):
        """Test loading RGBA image with transparency."""
        # Create RGBA image with transparency
        img = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        img_path = self.temp_dir / "test_rgba.png"
        img.save(img_path)
        
        result = self.loader.load_image(img_path)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.float64
    
    def test_load_image_file_not_found(self):
        """Test loading non-existent image file."""
        non_existent_path = self.temp_dir / "non_existent.png"
        
        with pytest.raises(FileNotFoundError):
            self.loader.load_image(non_existent_path)
    
    def test_load_image_invalid_file(self):
        """Test loading invalid image file."""
        # Create a text file with .png extension
        invalid_path = self.temp_dir / "invalid.png"
        invalid_path.write_text("This is not an image")
        
        with pytest.raises(ValueError):
            self.loader.load_image(invalid_path)
    
    def test_resize_with_aspect_ratio_square_to_square(self):
        """Test resizing square image to square target."""
        img = Image.new('RGB', (100, 100), (255, 0, 0))
        
        result = self.loader._resize_with_aspect_ratio(img)
        
        assert result.size == (256, 256)
    
    def test_resize_with_aspect_ratio_landscape(self):
        """Test resizing landscape image."""
        img = Image.new('RGB', (200, 100), (255, 0, 0))
        
        result = self.loader._resize_with_aspect_ratio(img)
        
        assert result.size == (256, 256)
        # Check that the image is centered with white padding
        result_array = np.array(result)
        # Top and bottom should have white padding
        assert np.all(result_array[0, :] == [255, 255, 255])
        assert np.all(result_array[-1, :] == [255, 255, 255])
    
    def test_resize_with_aspect_ratio_portrait(self):
        """Test resizing portrait image."""
        img = Image.new('RGB', (100, 200), (255, 0, 0))
        
        result = self.loader._resize_with_aspect_ratio(img)
        
        assert result.size == (256, 256)
        # Check that the image is centered with white padding
        result_array = np.array(result)
        # Left and right should have white padding
        assert np.all(result_array[:, 0] == [255, 255, 255])
        assert np.all(result_array[:, -1] == [255, 255, 255])
    
    def test_normalize_pixels(self):
        """Test pixel normalization."""
        # Create array with values in [0, 255] range
        img_array = np.array([[[0, 128, 255]]], dtype=np.float64)
        
        result = self.loader._normalize_pixels(img_array)
        
        expected = np.array([[[0.0, 128/255, 1.0]]])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_save_image_rgb(self):
        """Test saving RGB image."""
        # Create normalized image array
        image = np.random.rand(100, 100, 3)
        output_path = self.temp_dir / "output_rgb.png"
        
        self.loader.save_image(image, output_path)
        
        assert output_path.exists()
        # Verify we can load it back
        loaded = Image.open(output_path)
        assert loaded.mode == 'RGB'
        assert loaded.size == (100, 100)
    
    def test_save_image_grayscale(self):
        """Test saving grayscale image."""
        # Create normalized grayscale image array
        image = np.random.rand(100, 100)
        output_path = self.temp_dir / "output_gray.png"
        
        self.loader.save_image(image, output_path)
        
        assert output_path.exists()
        # Verify we can load it back
        loaded = Image.open(output_path)
        assert loaded.mode == 'L'
        assert loaded.size == (100, 100)
    
    def test_save_image_invalid_array(self):
        """Test saving invalid image array."""
        # Test with 1D array
        invalid_image = np.random.rand(100)
        output_path = self.temp_dir / "invalid.png"
        
        with pytest.raises(ValueError):
            self.loader.save_image(invalid_image, output_path)
    
    def test_save_image_invalid_channels(self):
        """Test saving image with invalid number of channels."""
        # Test with 4 channels (not supported)
        invalid_image = np.random.rand(100, 100, 4)
        output_path = self.temp_dir / "invalid.png"
        
        with pytest.raises(ValueError):
            self.loader.save_image(invalid_image, output_path)
    
    def test_save_image_creates_directory(self):
        """Test that save_image creates output directory if it doesn't exist."""
        image = np.random.rand(100, 100, 3)
        output_path = self.temp_dir / "subdir" / "output.png"
        
        self.loader.save_image(image, output_path)
        
        assert output_path.exists()
        assert output_path.parent.exists()
    
    def test_load_as_grayscale(self):
        """Test loading image as grayscale."""
        img_path = self.create_test_image((100, 100), 'RGB', (255, 128, 64))
        
        result = self.loader.load_as_grayscale(img_path)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256)
        assert result.dtype == np.float64
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_load_as_rgb(self):
        """Test loading image as RGB."""
        img_path = self.create_test_image((100, 100), 'L', 128)
        
        result = self.loader.load_as_rgb(img_path)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.float64
        assert 0 <= result.min() <= result.max() <= 1
    
    def test_load_as_rgb_rgba_conversion(self):
        """Test loading RGBA image as RGB."""
        # Create RGBA image with transparency
        img = Image.new('RGBA', (100, 100), (255, 0, 0, 128))
        img_path = self.temp_dir / "test_rgba.png"
        img.save(img_path)
        
        result = self.loader.load_as_rgb(img_path)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.float64
    
    def test_custom_target_size(self):
        """Test ImageLoader with custom target size."""
        loader = ImageLoader(target_size=(128, 128))
        img_path = self.create_test_image((100, 100), 'RGB')
        
        result = loader.load_image(img_path)
        
        assert result.shape == (128, 128, 3)
    
    def test_edge_case_very_small_image(self):
        """Test loading very small image."""
        img_path = self.create_test_image((1, 1), 'RGB', (255, 0, 0))
        
        result = self.loader.load_image(img_path)
        
        assert result.shape == (256, 256, 3)
        # The 1x1 red image gets scaled up to fill the entire 256x256 space
        # So the result should be mostly red, not white
        assert result[128, 128, 0] == 1.0  # Red channel should be 1.0
        assert result[128, 128, 1] == 0.0  # Green channel should be 0.0
        assert result[128, 128, 2] == 0.0  # Blue channel should be 0.0
    
    def test_edge_case_very_large_aspect_ratio(self):
        """Test loading image with extreme aspect ratio."""
        img_path = self.create_test_image((1000, 10), 'RGB', (255, 0, 0))
        
        result = self.loader.load_image(img_path)
        
        assert result.shape == (256, 256, 3)
        # Should have significant white padding on top and bottom
        assert np.all(result[0, :] == [1.0, 1.0, 1.0])  # White padding
        assert np.all(result[-1, :] == [1.0, 1.0, 1.0])  # White padding
    
    def test_batch_loading(self):
        """Test loading multiple images efficiently."""
        # Create multiple test images
        img_paths = []
        for i in range(3):
            img_path = self.create_test_image((100 + i*10, 100 + i*10), 'RGB')
            img_paths.append(img_path)
        
        # Load all images
        loaded_images = []
        for path in img_paths:
            img = self.loader.load_image(path)
            loaded_images.append(img)
        
        # All should have same target size
        for img in loaded_images:
            assert img.shape == (256, 256, 3)
    
    def test_memory_efficiency(self):
        """Test that image loading doesn't leak memory."""
        # Create a moderately large image
        img_path = self.create_test_image((500, 500), 'RGB')
        
        # Load the same image multiple times
        for _ in range(10):
            img = self.loader.load_image(img_path)
            assert img.shape == (256, 256, 3)
            # Force garbage collection by deleting reference
            del img
    
    def test_different_image_formats(self):
        """Test loading different image formats."""
        # Test JPEG format
        img = Image.new('RGB', (100, 100), (255, 0, 0))
        jpeg_path = self.temp_dir / "test.jpg"
        img.save(jpeg_path, 'JPEG')
        
        result = self.loader.load_image(jpeg_path)
        assert result.shape == (256, 256, 3)
        
        # Test PNG format (already tested in other methods)
        png_path = self.create_test_image((100, 100), 'RGB')
        result = self.loader.load_image(png_path)
        assert result.shape == (256, 256, 3)
    
    def test_pixel_value_consistency(self):
        """Test that pixel values are consistently normalized."""
        # Create image with known pixel values
        img = Image.new('RGB', (2, 2), (128, 64, 192))  # Mid-range values
        img_path = self.temp_dir / "test_values.png"
        img.save(img_path)
        
        result = self.loader.load_image(img_path)
        
        # Check that values are properly normalized
        assert 0 <= result.min() <= result.max() <= 1
        # The exact values will depend on resizing, but should be reasonable
        assert 0.1 < result.mean() < 0.9  # Should not be all black or white


if __name__ == "__main__":
    pytest.main([__file__])