#!/usr/bin/env python3
"""
Simple test script to verify the web application functionality.
"""

import sys
from pathlib import Path
import importlib.util

def test_imports():
    """Test that all required modules can be imported."""
    
    print("Testing imports...")
    
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))
    
    try:
        # Test core modules
        from compression.svd_compressor import SVDCompressor
        from evaluation.metrics_calculator import MetricsCalculator
        from data.image_loader import ImageLoader
        print("✅ Core modules imported successfully")
        
        # Test web app modules
        from utils.styling import create_metric_card, create_status_indicator
        from utils.navigation import setup_navigation
        print("✅ Web app utilities imported successfully")
        
        # Test page modules
        from pages import home, single_compression, batch_processing, comparison, tutorial
        print("✅ Page modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality of core components."""
    
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from compression.svd_compressor import SVDCompressor
        from evaluation.metrics_calculator import MetricsCalculator
        
        # Create test image
        test_image = np.random.rand(64, 64, 3)
        
        # Test compression
        compressor = SVDCompressor()
        compressed_image, metadata = compressor.compress_image(test_image, k=10)
        print("✅ Image compression works")
        
        # Test metrics
        metrics_calc = MetricsCalculator()
        psnr = metrics_calc.calculate_psnr(test_image, compressed_image)
        ssim = metrics_calc.calculate_ssim(test_image, compressed_image)
        print(f"✅ Metrics calculation works (PSNR: {psnr:.2f}, SSIM: {ssim:.3f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False


def test_web_app_structure():
    """Test web application structure."""
    
    print("\nTesting web app structure...")
    
    try:
        # Check if main app file exists
        app_file = Path(__file__).parent / "app.py"
        if not app_file.exists():
            print("❌ app.py not found")
            return False
        
        # Check page files
        pages_dir = Path(__file__).parent / "pages"
        required_pages = ["home.py", "single_compression.py", "batch_processing.py", 
                         "comparison.py", "tutorial.py"]
        
        for page in required_pages:
            page_file = pages_dir / page
            if not page_file.exists():
                print(f"❌ {page} not found")
                return False
        
        print("✅ All required page files exist")
        
        # Check utils
        utils_dir = Path(__file__).parent / "utils"
        required_utils = ["styling.py", "navigation.py"]
        
        for util in required_utils:
            util_file = utils_dir / util
            if not util_file.exists():
                print(f"❌ {util} not found")
                return False
        
        print("✅ All utility files exist")
        return True
        
    except Exception as e:
        print(f"❌ Structure error: {e}")
        return False


def main():
    """Run all tests."""
    
    print("🧪 SVD Image Compression Web App Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_web_app_structure
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Web application is ready to run.")
        print("\nTo start the application, run:")
        print("streamlit run webapp/app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())