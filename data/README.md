# Sample Dataset for SVD Image Compression

This directory contains synthetic sample images designed to demonstrate the characteristics of SVD-based image compression across different image types.

## Dataset Structure

- **portraits/**: Images with smooth gradients and low-frequency content
- **landscapes/**: Images with mixed frequency content and natural patterns  
- **textures/**: Images with high-frequency content and complex patterns
- **samples/**: Additional sample images for general use

## Image Characteristics

### Portraits
- Smooth gradients and transitions
- Low-frequency content dominates
- Excellent SVD compression ratios (10-50x typical)
- High PSNR/SSIM values even at low k-values

### Landscapes  
- Mixed frequency content
- Natural patterns and structures
- Moderate SVD compression ratios (5-20x typical)
- Balanced quality vs compression trade-offs

### Textures
- High-frequency details and patterns
- Complex spatial relationships
- Lower SVD compression ratios (2-10x typical)
- Requires higher k-values for acceptable quality

## Usage

These images are specifically designed for:
- Educational demonstrations of SVD compression
- Benchmarking compression algorithms
- Understanding the relationship between image content and compression performance
- Testing and validation of the SVD compression system

## Technical Specifications

- **Size**: 256×256 pixels (standard for this system)
- **Format**: PNG (lossless)
- **Color**: RGB, 8-bit per channel
- **Generated**: Synthetically created for consistent, reproducible results

## Attribution

These sample images are synthetically generated and are in the public domain. They are created specifically for educational and research purposes in the context of SVD image compression.
