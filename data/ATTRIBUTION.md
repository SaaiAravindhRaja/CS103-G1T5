# Dataset Attribution and Licensing

## Sample Dataset Information

This directory contains synthetic sample images created specifically for the SVD Image Compression System. All images are generated programmatically and are designed to demonstrate different compression characteristics.

## Image Categories and Sources

### Portraits (`portraits/`)
- **Source**: Synthetically generated using Python PIL/Pillow
- **Description**: Artificial portrait-like images with smooth gradients and low-frequency content
- **Purpose**: Demonstrate optimal SVD compression scenarios
- **Generation Method**: Programmatic creation using geometric shapes, gradients, and simple features

### Landscapes (`landscapes/`)
- **Source**: Synthetically generated using Python PIL/Pillow and NumPy
- **Description**: Artificial landscape-like images with mixed frequency content
- **Purpose**: Demonstrate moderate SVD compression performance
- **Generation Method**: Algorithmic creation using gradients, noise patterns, and geometric shapes

### Textures (`textures/`)
- **Source**: Synthetically generated using Python PIL/Pillow and NumPy
- **Description**: Artificial texture patterns with high-frequency content
- **Purpose**: Demonstrate challenging scenarios for SVD compression
- **Generation Method**: Mathematical patterns, noise generation, and geometric repetition

## Licensing

### Synthetic Images
- **License**: Public Domain (CC0 1.0 Universal)
- **Rights**: No rights reserved - free for any use
- **Attribution**: Not required, but appreciated
- **Commercial Use**: Permitted
- **Modification**: Permitted
- **Distribution**: Permitted

### Generation Scripts
- **License**: MIT License (same as project)
- **Location**: `scripts/generate_sample_datasets.py`
- **Rights**: See project LICENSE file

## Technical Specifications

### Image Properties
- **Dimensions**: 256×256 pixels
- **Color Mode**: RGB (24-bit)
- **File Format**: PNG (lossless compression)
- **Bit Depth**: 8 bits per channel
- **Color Space**: sRGB

### Generation Parameters
- **Reproducibility**: Fixed random seeds for consistent results
- **Quality**: Lossless PNG format preserves original synthetic data
- **Standardization**: All images normalized to same dimensions and format

## Usage Rights and Responsibilities

### Permitted Uses
- ✅ Educational and research purposes
- ✅ Academic presentations and publications
- ✅ Software testing and benchmarking
- ✅ Commercial applications and products
- ✅ Modification and derivative works
- ✅ Redistribution with or without modification

### Recommended Attribution
While not required, if you use these images in academic work, please consider citing:

```
SVD Image Compression System Sample Dataset
Generated synthetically for educational and research purposes
Available at: [Your Repository URL]
```

### No Warranties
These synthetic images are provided "as is" without any warranties. They are created for educational and demonstration purposes and may not represent real-world image characteristics perfectly.

## Dataset Validation

### Quality Assurance
- All images generated using documented, reproducible methods
- Consistent technical specifications across all samples
- Verified to work correctly with the SVD compression system
- Tested across different image categories for demonstration purposes

### Reproducibility
To regenerate this dataset:
```bash
python scripts/generate_sample_datasets.py
```

This will create identical images using the same algorithms and parameters.

## Contact and Support

For questions about the dataset or licensing:
- Check the main project README
- Review the project documentation
- Contact the project maintainers through GitHub

## Version History

- **v1.0** (Initial Release): 10 synthetic images across 3 categories
  - 3 portrait-style images
  - 3 landscape-style images  
  - 4 texture-style images

## Related Files

- `dataset_manifest.json`: Technical metadata and specifications
- `README.md`: Dataset overview and usage instructions
- `scripts/generate_sample_datasets.py`: Generation script and methodology

---

*This dataset is created to support education and research in image compression techniques. All synthetic images are generated specifically for this purpose and do not contain any copyrighted or proprietary content.*