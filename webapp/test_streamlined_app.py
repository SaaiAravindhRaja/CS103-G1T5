#!/usr/bin/env python3
"""
Test script for the streamlined SVD Image Compression webapp.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import io

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from compression.svd_compressor import SVDCompressor
        print("✅ SVDCompressor import successful")
    except ImportError as e:
        print(f"❌ SVDCompressor import failed: {e}")
        return False
    
    try:
        from evaluation.metrics_calculator import MetricsCalculator
        print("✅ MetricsCalculator import successful")
    except ImportError as e:
        print(f"❌ MetricsCalculator import failed: {e}")
        return False
    
    try:
        from utils.styling import load_core_styles, create_metric_card
        print("✅ Styling utilities import successful")
    except ImportError as e:
        print(f"❌ Styling utilities import failed: {e}")
        return False
    
    try:
        from utils.simple_upload import create_simple_upload
        print("✅ Simple upload component import successful")
    except ImportError as e:
        print(f"❌ Simple upload component import failed: {e}")
        return False
    
    return True


def test_core_functionality():
    """Test core SVD compression functionality."""
    print("\nTesting core functionality...")
    
    try:
        from compression.svd_compressor import SVDCompressor
        from evaluation.metrics_calculator import MetricsCalculator
        
        # Create a simple test image
        test_image = np.random.rand(50, 50, 3)
        print("✅ Test image created")
        
        # Test SVD compression
        compressor = SVDCompressor()
        k_value = 10
        
        # Compress the image
        compressed_image, metadata = compressor.compress_image(test_image, k_value)
        
        # Convert grayscale back to RGB if needed
        if len(compressed_image.shape) == 2:
            compressed_image = np.stack([compressed_image] * 3, axis=-1)
        
        compressed_image = np.clip(compressed_image, 0, 1)
        print("✅ SVD compression successful")
        
        # Test metrics calculation
        metrics_calc = MetricsCalculator()
        psnr = metrics_calc.calculate_psnr(test_image, compressed_image)
        ssim = metrics_calc.calculate_ssim(test_image, compressed_image)
        mse = metrics_calc.calculate_mse(test_image, compressed_image)
        
        print(f"✅ Metrics calculation successful:")
        print(f"   PSNR: {psnr:.2f} dB")
        print(f"   SSIM: {ssim:.3f}")
        print(f"   MSE: {mse:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False


def test_styling():
    """Test styling utilities."""
    print("\nTesting styling utilities...")
    
    try:
        from utils.styling import create_metric_card, show_loading
        
        # Test metric card creation
        card_html = create_metric_card("Test Metric", "42.5", "Test description", "success")
        assert "Test Metric" in card_html
        assert "42.5" in card_html
        print("✅ Metric card creation successful")
        
        # Test loading indicator
        loading_html = show_loading("Test loading...")
        assert "Test loading..." in loading_html
        assert "loading-spinner" in loading_html
        print("✅ Loading indicator creation successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Styling utilities test failed: {e}")
        return False


def test_image_processing():
    """Test image processing utilities."""
    print("\nTesting image processing...")
    
    try:
        # Create a test PIL image
        test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_pil = Image.fromarray(test_array)
        
        # Convert to bytes (simulating file upload)
        img_buffer = io.BytesIO()
        test_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Test loading from bytes
        loaded_image = Image.open(img_buffer)
        loaded_array = np.array(loaded_image) / 255.0
        
        assert loaded_array.shape == (100, 100, 3)
        assert loaded_array.dtype == np.float64
        assert 0 <= loaded_array.min() <= loaded_array.max() <= 1
        
        print("✅ Image processing successful")
        return True
        
    except Exception as e:
        print(f"❌ Image processing test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing Streamlined SVD Image Compression Webapp")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_core_functionality,
        test_styling,
        test_image_processing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The streamlined webapp is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)