"""
Integration test for the new compression controls in the single compression page.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_integration():
    """Test integration of compression controls with the main application."""
    
    print("Testing compression controls integration...")
    
    # Test imports
    try:
        from utils.compression_controls import (
            create_compression_controls_panel,
            create_compression_tooltip_guide,
            create_compression_metrics_display,
            _generate_quick_preview
        )
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test with sample data
    test_image = np.random.rand(128, 128, 3)
    
    # Test quick preview generation
    try:
        preview = _generate_quick_preview(test_image, 20, "RGB (Color)")
        assert preview.shape == test_image.shape or preview.shape[0] <= test_image.shape[0]
        print("✅ Quick preview generation works")
    except Exception as e:
        print(f"❌ Quick preview failed: {e}")
        return False
    
    # Test grayscale preview
    try:
        preview_gray = _generate_quick_preview(test_image, 10, "Grayscale")
        assert preview_gray.shape == test_image.shape
        print("✅ Grayscale preview generation works")
    except Exception as e:
        print(f"❌ Grayscale preview failed: {e}")
        return False
    
    # Test metrics display data structure
    test_metrics = {
        'psnr': 28.5,
        'ssim': 0.82,
        'compression_ratio': 3.2,
        'k_value': 25,
        'mse': 0.001
    }
    
    try:
        # This would normally create Streamlit components, but we can test the data processing
        from utils.compression_controls import _calculate_composite_quality_score
        score = _calculate_composite_quality_score(test_metrics['psnr'], test_metrics['ssim'])
        assert 0 <= score <= 100
        print("✅ Metrics processing works")
    except Exception as e:
        print(f"❌ Metrics processing failed: {e}")
        return False
    
    print("🎉 All integration tests passed!")
    return True


if __name__ == "__main__":
    success = test_integration()
    if not success:
        sys.exit(1)