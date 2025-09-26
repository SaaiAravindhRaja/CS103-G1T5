# SVD Image Compression Web Application - Usage Guide

## Quick Start

1. **Start the application:**
   ```bash
   cd webapp
   python run.py
   ```
   Or directly with Streamlit:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Navigate** using the sidebar menu to access different tools

## Features Overview

### 🏠 Home Page
- **Purpose**: Introduction to SVD compression and project overview
- **Features**:
  - Mathematical background and theory
  - Interactive SVD visualization
  - Getting started guide
  - Best practices and tips

### 🔍 Single Image Compression
- **Purpose**: Analyze individual images with real-time controls
- **Features**:
  - Drag-and-drop image upload (PNG, JPG, JPEG)
  - Real-time k-value slider (1-256)
  - RGB and Grayscale processing modes
  - Quality metrics dashboard (PSNR, SSIM, MSE)
  - Side-by-side image comparison
  - Difference visualization
  - Singular values analysis with interactive plots
  - Download compressed images and reports

**How to use:**
1. Upload an image using the file uploader
2. Adjust compression level with the k-value slider
3. Choose processing mode (RGB or Grayscale)
4. Click "Compress Image" to process
5. View results, metrics, and comparisons
6. Download compressed image or metrics report

### 📊 Batch Processing
- **Purpose**: Process multiple images simultaneously
- **Features**:
  - Multiple image upload
  - Configurable k-value ranges or custom values
  - RGB, Grayscale, or Both processing modes
  - Progress tracking with real-time updates
  - Comprehensive results table with filtering
  - Interactive visualizations (quality trends, compression analysis)
  - Batch download of compressed images as ZIP
  - Export results as CSV

**How to use:**
1. Upload multiple images (up to 10 recommended)
2. Configure k-values (predefined range, custom values, or range analysis)
3. Select processing options (mode, resize, save options)
4. Click "Start Batch Processing"
5. View comprehensive results and visualizations
6. Download results CSV or compressed images ZIP

### ⚖️ Comparison Analysis
- **Purpose**: Compare different compression levels side-by-side
- **Features**:
  - Upload single image for multi-level analysis
  - Quick comparison (4 levels) or custom k-values
  - Range analysis with configurable parameters
  - Visual comparison grid showing all compression levels
  - Interactive quality trend analysis
  - Trade-off analysis (quality vs compression)
  - Statistical analysis with correlation matrix
  - Detailed metrics and insights
  - Export comparison reports and data

**How to use:**
1. Upload an image for comparison
2. Choose comparison mode (Quick, Custom, or Range)
3. Configure analysis options (grid view, metrics, statistics)
4. Click "Run Comparison Analysis"
5. Explore visual comparisons and interactive plots
6. Review statistical analysis and insights
7. Download comparison report or results CSV

## Tips for Best Results

### Image Upload
- **Supported formats**: PNG, JPG, JPEG
- **Recommended size**: Images are automatically resized to 256×256
- **File size limit**: 10MB maximum
- **Quality**: Higher resolution originals generally produce better results

### K-Value Selection
- **Low k (1-10)**: High compression, lower quality, visible artifacts
- **Medium k (10-30)**: Balanced compression and quality
- **High k (30-100)**: Lower compression, higher quality
- **Very high k (100+)**: Minimal compression, near-original quality

### Processing Modes
- **RGB (Color)**: Preserves color information, larger file sizes
- **Grayscale**: Faster processing, smaller files, good for analysis
- **Both**: Compare color vs grayscale results (batch processing only)

### Quality Metrics Interpretation
- **PSNR (Peak Signal-to-Noise Ratio)**:
  - > 40 dB: Excellent quality
  - 30-40 dB: Good quality
  - 20-30 dB: Fair quality
  - < 20 dB: Poor quality

- **SSIM (Structural Similarity Index)**:
  - > 0.9: Excellent structural similarity
  - 0.7-0.9: Good similarity
  - 0.5-0.7: Fair similarity
  - < 0.5: Poor similarity

- **MSE (Mean Squared Error)**:
  - Lower values indicate better quality
  - Highly sensitive to pixel-level differences

- **Compression Ratio**:
  - Higher ratios mean more space saved
  - Calculated as original_size / compressed_size

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure you're running from the webapp directory
   - Check that all dependencies are installed: `pip install -r ../requirements.txt`

2. **Image Upload Fails**:
   - Check file format (PNG, JPG, JPEG only)
   - Ensure file size is under 10MB
   - Try converting image to RGB format

3. **Processing Errors**:
   - Very small k-values may cause numerical issues
   - Large images may take longer to process
   - Try reducing image size or k-value range

4. **Performance Issues**:
   - Batch processing with many images/k-values can be slow
   - Reduce the number of images or k-values for faster processing
   - Close other browser tabs to free up memory

### Browser Compatibility
- **Recommended**: Chrome 90+, Firefox 88+, Safari 14+, Edge 90+
- **Features**: JavaScript must be enabled
- **Performance**: Better performance on desktop browsers

## Advanced Usage

### Custom Analysis
- Use the comparison page for detailed k-value analysis
- Export results for further analysis in other tools
- Combine batch processing results with custom visualization

### Academic Use
- All pages include professional styling suitable for presentations
- Export comprehensive reports with metrics and analysis
- Use interactive plots for demonstrations and teaching

### Development
- Modify `config.py` for custom settings
- Extend pages in the `pages/` directory
- Add custom styling in `utils/styling.py`

## Support

For issues or questions:
1. Check this usage guide
2. Review the main project README
3. Check the GitHub repository for documentation
4. Report bugs through the GitHub issues page