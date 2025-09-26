"""
Test script for the new compression controls component.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def test_compression_controls():
    """Test the compression controls component functionality."""
    
    # Create a test image
    test_image = np.random.rand(64, 64, 3)
    
    # Test helper functions
    from utils.compression_controls import (
        _calculate_quality_indicator,
        _get_quality_color,
        _estimate_compression_ratio,
        _calculate_energy_based_k,
        _calculate_composite_quality_score
    )
    
    # Test quality indicator
    quality = _calculate_quality_indicator(20, 64)
    print(f"Quality indicator for k=20, max_k=64: {quality}")
    
    # Test quality color
    color = _get_quality_color(quality)
    print(f"Quality color: {color}")
    
    # Test compression ratio estimation
    ratio = _estimate_compression_ratio(20, test_image.shape)
    print(f"Estimated compression ratio for k=20: {ratio:.2f}:1")
    
    # Test energy-based k calculation
    optimal_k = _calculate_energy_based_k(test_image, energy_threshold=0.9)
    print(f"Optimal k for 90% energy retention: {optimal_k}")
    
    # Test composite quality score
    score = _calculate_composite_quality_score(30.0, 0.85)
    print(f"Composite quality score (PSNR=30, SSIM=0.85): {score:.1f}")
    
    print("All tests passed successfully!")


if __name__ == "__main__":
    test_compression_controls()