# Streamlined SVD Image Compression Webapp

A clean, focused web interface for exploring image compression using Singular Value Decomposition (SVD).

## ✨ Features

- **Simple Interface**: Single-page design focused on core functionality
- **Easy Upload**: Drag-and-drop image upload with format validation
- **Real-time Compression**: Adjust k-value and see results instantly
- **Quality Metrics**: PSNR, SSIM, MSE, and compression ratio analysis
- **Mobile Responsive**: Works well on desktop, tablet, and mobile devices
- **Download Results**: Export compressed images and analysis reports

## 🚀 Quick Start

1. **Run the webapp:**
   ```bash
   cd webapp
   streamlit run app.py
   ```

2. **Upload an image:**
   - Drag and drop or click to browse
   - Supported formats: PNG, JPG, JPEG, BMP, TIFF

3. **Adjust compression:**
   - Use the k-value slider (lower = more compression)
   - Choose RGB or Grayscale mode

4. **View results:**
   - Compare original vs compressed images
   - Check quality metrics
   - Download compressed image and report

## 📊 Understanding the Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better (>25 dB = good quality)
- **SSIM (Structural Similarity Index)**: Closer to 1 is better (>0.8 = high similarity)
- **Compression Ratio**: Higher means more space saved
- **Quality Score**: Overall assessment (0-100 scale)

## 🎯 Tips for Best Results

- **Start with k=20** for most images
- **Lower k-values** for higher compression (but lower quality)
- **Higher k-values** for better quality (but less compression)
- **Smooth images** (gradients, simple patterns) compress better
- **Detailed photos** may need higher k-values to maintain quality

## 🧮 How SVD Compression Works

SVD decomposes an image matrix **A** into three components:

**A = U × Σ × V^T**

Where:
- **U**: Left singular vectors (spatial patterns)
- **Σ**: Singular values (importance weights)
- **V^T**: Right singular vectors (frequency patterns)

By keeping only the **k** largest singular values, we reconstruct an approximation with reduced storage requirements.

## 🔧 Technical Details

### File Structure
```
webapp/
├── app.py                    # Main application
├── pages/
│   └── single_compression.py # Core compression interface
├── utils/
│   ├── styling.py           # Clean CSS styles
│   └── simple_upload.py     # Upload component
└── test_streamlined_app.py  # Test suite
```

### Dependencies
- **Streamlit**: Web interface framework
- **NumPy**: Numerical computations
- **Pillow**: Image processing
- **SVDCompressor**: Core compression algorithm
- **MetricsCalculator**: Quality assessment

### Performance
- **Memory efficient**: Optimized for typical image sizes
- **Fast processing**: Real-time compression for most images
- **Error handling**: Graceful degradation for edge cases
- **Mobile optimized**: Responsive design for all devices

## 🧪 Testing

Run the test suite to verify functionality:

```bash
python test_streamlined_app.py
```

Tests cover:
- Module imports
- Core SVD compression
- Quality metrics calculation
- Styling utilities
- Image processing pipeline

## 🎨 Design Principles

This streamlined version follows these principles:

1. **Simplicity**: Focus on core functionality without clutter
2. **Usability**: Intuitive interface with clear feedback
3. **Performance**: Fast, responsive user experience
4. **Accessibility**: Works well across devices and screen sizes
5. **Reliability**: Robust error handling and validation

## 📱 Mobile Support

The interface is fully responsive and includes:
- Touch-friendly controls
- Optimized layouts for small screens
- Readable text and buttons
- Efficient image handling

## 🔍 Troubleshooting

**Image won't upload:**
- Check file format (PNG, JPG, JPEG, BMP, TIFF)
- Ensure file size is reasonable (<10MB)
- Try refreshing the page

**Compression fails:**
- Try a lower k-value
- Check if image is valid
- Ensure sufficient memory available

**Poor quality results:**
- Increase k-value for better quality
- Some image types compress better than others
- Check PSNR and SSIM metrics for guidance

## 📈 Future Enhancements

Potential improvements for future versions:
- Batch processing support
- Additional compression algorithms
- Advanced quality metrics
- Export to different formats
- Comparison tools

## 🤝 Contributing

This is a streamlined educational tool. For contributions:
1. Keep the interface simple and focused
2. Maintain mobile responsiveness
3. Add comprehensive tests
4. Follow the existing code style
5. Document any new features

## 📄 License

Built for educational purposes. See main project LICENSE for details.