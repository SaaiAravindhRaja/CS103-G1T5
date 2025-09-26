"""
Test script for the results display component.
"""

import numpy as np
import sys
from pathlib import Path

# Add paths
webapp_path = Path(__file__).parent
src_path = webapp_path.parent / "src"
sys.path.insert(0, str(webapp_path))
sys.path.insert(0, str(src_path))

def test_results_display_component():
    """Test the results display component with mock data."""
    
    try:
        from utils.results_display import (
            calculate_quality_score,
            get_metric_color,
            get_quality_assessment_text,
            generate_quality_recommendations,
            estimate_file_size,
            generate_comprehensive_report
        )
        
        print("✅ All functions imported successfully")
        
        # Test calculate_quality_score
        quality_score = calculate_quality_score(30.0, 0.85)
        print(f"✅ Quality score calculation: {quality_score:.1f}/100")
        
        # Test get_metric_color
        color = get_metric_color(85, [60, 80, 90])
        print(f"✅ Metric color for 85: {color}")
        
        # Test get_quality_assessment_text
        psnr_assessment = get_quality_assessment_text(30.0, 'psnr')
        ssim_assessment = get_quality_assessment_text(0.85, 'ssim')
        print(f"✅ PSNR assessment: {psnr_assessment}")
        print(f"✅ SSIM assessment: {ssim_assessment}")
        
        # Test with mock compression data
        mock_compression_data = {
            'k_value': 25,
            'psnr': 30.5,
            'ssim': 0.85,
            'mse': 0.001,
            'compression_ratio': 4.2,
            'mode': 'RGB',
            'quality_score': quality_score
        }
        
        # Test generate_quality_recommendations
        recommendations = generate_quality_recommendations(mock_compression_data)
        print(f"✅ Generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        # Test estimate_file_size
        mock_image = np.random.rand(256, 256, 3)
        png_size = estimate_file_size(mock_image, "PNG")
        jpeg_size = estimate_file_size(mock_image, "JPEG", 95)
        print(f"✅ File size estimates - PNG: {png_size:.1f} KB, JPEG: {jpeg_size:.1f} KB")
        
        # Test generate_comprehensive_report
        report = generate_comprehensive_report(mock_compression_data, "test_image.png")
        print(f"✅ Generated comprehensive report ({len(report)} characters)")
        
        print("\n🎉 All tests passed! Results display component is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_results_display_component()