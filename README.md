# SVD Image Compression System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

A comprehensive, professional-grade implementation of image compression using **Singular Value Decomposition (SVD)**. This system provides multiple interfaces for experimenting with SVD-based compression, including interactive web applications, batch processing tools, and Jupyter notebooks for research and education.

![SVD Compression Demo](slides/plots/compression_analysis.png)

## 🌟 Features

### 🎯 Core Compression Engine
- **Mathematical Precision**: Implements SVD compression with numerical stability and error handling
- **Multi-Channel Support**: Handles both RGB and grayscale images with channel-wise processing
- **Configurable Compression**: Adjustable k-values for fine-tuned compression control
- **Quality Metrics**: Comprehensive evaluation using PSNR, SSIM, MSE, and compression ratios

### 🖥️ Interactive Web Application
- **Real-time Compression**: Instant preview with interactive k-value sliders
- **Professional Interface**: Academic-grade styling suitable for presentations and demonstrations
- **Multi-Page Layout**: Dedicated pages for single compression, batch processing, and comparison analysis
- **Export Capabilities**: Download compressed images, analysis reports, and comprehensive datasets

### 📊 Advanced Analysis Tools
- **Batch Processing**: Systematic experiments across multiple images and compression levels
- **Visualization Suite**: Professional plots showing quality trends, singular value analysis, and compression trade-offs
- **Statistical Analysis**: Correlation matrices, trend analysis, and performance profiling
- **Jupyter Integration**: Reproducible research notebooks with complete experimental workflows

### 🎓 Educational Resources
- **Interactive Tutorials**: Step-by-step guides for understanding SVD compression
- **Mathematical Background**: Clear explanations of the theory behind SVD compression
- **Demo Materials**: Comprehensive demo scripts and presentation materials
- **Academic Documentation**: Complete research report with methodology and results

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/svd-image-compression.git
   cd svd-image-compression
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Launch the Web Application

```bash
cd webapp
streamlit run app.py
```

Open your browser to `http://localhost:8501` to access the interactive interface.

### Run Jupyter Notebooks

```bash
jupyter notebook notebooks/experiments.ipynb
```

## 📖 Usage Examples

### Web Application

#### Single Image Compression
1. Navigate to "Single Image Compression" in the sidebar
2. Upload an image (PNG, JPG, JPEG supported)
3. Adjust the k-value slider to control compression level
4. View real-time quality metrics and side-by-side comparison
5. Download compressed images and analysis reports

#### Batch Processing
1. Go to "Batch Processing" page
2. Upload multiple images (up to 10 recommended)
3. Configure k-value ranges and processing options
4. Monitor progress with real-time updates
5. Export results as CSV or download compressed images as ZIP

#### Comparison Analysis
1. Visit "Comparison Analysis" page
2. Upload an image for multi-level analysis
3. Choose quick comparison or custom k-values
4. Explore interactive visualizations and statistical analysis
5. Export comprehensive comparison reports

### Command Line Interface

```python
from src.compression.svd_compressor import SVDCompressor
from src.data.image_loader import ImageLoader

# Load and compress an image
loader = ImageLoader()
compressor = SVDCompressor()

image = loader.load_image("path/to/image.jpg")
compressed_image, metadata = compressor.compress_image(image, k=50)

print(f"Compression ratio: {metadata['compression_ratio']:.2f}x")
print(f"PSNR: {metadata['psnr']:.2f} dB")
```

### Batch Experiments

```python
from src.batch.experiment_runner import ExperimentRunner, ExperimentConfig

config = ExperimentConfig(
    datasets=["portraits", "landscapes", "textures"],
    k_values=list(range(10, 101, 10)),
    output_dir="results/experiments"
)

runner = ExperimentRunner()
results = runner.run_batch_experiments(config)
```

## 📁 Project Structure

```
svd-image-compression/
├── src/                          # Core source code
│   ├── compression/              # SVD compression algorithms
│   ├── data/                     # Data management and preprocessing
│   ├── evaluation/               # Quality metrics and performance profiling
│   ├── visualization/            # Plotting and analysis tools
│   └── batch/                    # Batch processing and experiments
├── webapp/                       # Streamlit web application
│   ├── pages/                    # Multi-page application structure
│   ├── utils/                    # Web-specific utilities
│   └── app.py                    # Main application entry point
├── notebooks/                    # Jupyter notebooks for research
├── tests/                        # Comprehensive test suite
│   ├── unit/                     # Unit tests for core modules
│   ├── integration/              # Integration and workflow tests
│   └── performance/              # Performance benchmarks
├── data/                         # Sample datasets
│   ├── portraits/                # Portrait images
│   ├── landscapes/               # Landscape images
│   └── textures/                 # Texture and diagram images
├── results/                      # Generated results and plots
├── report/                       # Academic documentation
├── slides/                       # Presentation materials
└── demo/                         # Demo scripts and materials
```

## 🔬 Technical Details

### SVD Compression Algorithm

The system implements SVD compression using the mathematical decomposition:

```
A = UΣV^T
```

Where for compression with k singular values:
- **Storage Original**: m × n pixels
- **Storage Compressed**: mk + k + kn values
- **Compression Ratio**: mn / (mk + k + kn)

### Quality Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality in decibels
- **SSIM (Structural Similarity Index)**: Evaluates structural similarity (0-1 scale)
- **MSE (Mean Squared Error)**: Pixel-level difference measurement
- **Compression Ratio**: Storage reduction factor

### Performance Characteristics

- **Processing Time**: 15-35ms for 256×256 images on standard hardware
- **Memory Usage**: Optimized for batch processing with memory profiling
- **Scalability**: Parallel processing support for large datasets

## 📊 Sample Results

![Singular Values Analysis](slides/plots/singular_values.png)
*Singular value decay analysis showing compression characteristics*

![PSNR vs K Analysis](slides/plots/psnr_vs_k.png)
*Quality vs compression trade-off analysis across different image categories*

## 🧪 Testing

Run the complete test suite:

```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Performance benchmarks
python -m pytest tests/performance/ -v

# All tests
python -m pytest tests/ -v
```

## 📚 Documentation

- **[Web App Usage Guide](webapp/USAGE.md)**: Comprehensive guide for the web interface
- **[Demo Script](demo/demo_script.md)**: Step-by-step presentation guide
- **[Academic Report](report/academic_report.md)**: Complete research documentation
- **[Source Code Documentation](src/)**: Well-documented source code with comprehensive docstrings

## 🎓 Educational Use

This system is designed for educational and research purposes:

- **Linear Algebra Education**: Visualize SVD concepts with real images
- **Research Platform**: Systematic evaluation of compression algorithms
- **Academic Presentations**: Professional interface suitable for demonstrations
- **Reproducible Research**: Complete experimental workflows with documentation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Run tests: `python -m pytest`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mathematical Foundation**: Based on classical SVD theory and linear algebra principles
- **Scientific Libraries**: Built with NumPy, SciPy, scikit-image, and matplotlib
- **Web Framework**: Powered by Streamlit for interactive applications
- **Academic Inspiration**: Designed for educational excellence and research reproducibility

## 📞 Support

- **Documentation**: Check the comprehensive guides in the `docs/` directory
- **Issues**: Report bugs and request features through GitHub Issues
- **Discussions**: Join community discussions for questions and collaboration
- **Academic Use**: Contact us for educational licensing and collaboration opportunities

## Contributors

<table>
	<tr>
			<td align="center">
				<a href="https://github.com/SaaiAravindhRaja">
					<img src="https://github.com/SaaiAravindhRaja.png" width="80" alt="SaaiAravindhRaja"/><br/>
					<sub><b>SaaiAravindhRaja</b></sub><br/>
					<sub>Saai</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/halowenfright">
					<img src="https://github.com/halowenfright.png" width="80" alt="halowenfright"/><br/>
					<sub><b>halowenfright</b></sub><br/>
					<sub>Sherman</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/ravenglider">
					<img src="https://github.com/ravenglider.png" width="80" alt="ravenglider"/><br/>
					<sub><b>ravenglider</b></sub><br/>
					<sub>Sonia</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/cohiee">
					<img src="https://github.com/cohiee.png" width="80" alt="cohiee"/><br/>
					<sub><b>cohiee</b></sub><br/>
					<sub>Vincent</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/seraphiii">
					<img src="https://github.com/seraphiii.png" width="80" alt="seraphiii"/><br/>
					<sub><b>seraphiii</b></sub><br/>
					<sub>Zaccheus</sub>
				</a>
			</td>
			<td align="center">
				<a href="https://github.com/Ridheema776">
					<img src="https://github.com/Ridheema776.png" width="80" alt="Ridheema776"/><br/>
					<sub><b>Ridheema776</b></sub><br/>
					<sub>Ridheema</sub>
				</a>
			</td>
	</tr>
</table>

---

*Built with ❤️ for education, research, and the advancement of computational mathematics.*

