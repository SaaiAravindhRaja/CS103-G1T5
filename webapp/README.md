# SVD Image Compression Web Application

A professional, academic-grade web interface for exploring image compression using Singular Value Decomposition (SVD).

## Features

- **Multi-page Layout**: Clean navigation between different analysis tools
- **Professional Styling**: Academic-grade CSS with responsive design
- **Interactive Analysis**: Real-time compression with quality metrics
- **Batch Processing**: Handle multiple images simultaneously
- **Comparison Tools**: Side-by-side analysis of compression levels
- **Export Options**: Download compressed images and analysis reports

## Quick Start

### Running the Application

```bash
# From the webapp directory
python run.py

# Or directly with streamlit
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Application Structure

```
webapp/
├── app.py                 # Main application entry point
├── run.py                 # Simple run script
├── config.py              # Configuration settings
├── styles/                # Tailwind CSS files
│   ├── input.css          # Source CSS with custom styles
│   ├── output.css         # Generated CSS (build mode)
│   └── README.md          # Styling documentation
├── utils/                 # Utility modules
│   ├── styling.py         # Tailwind CSS integration
│   └── navigation.py      # Navigation and UI utilities
├── pages/                 # Individual page modules
│   ├── home.py            # Home page with project overview
│   ├── single_compression.py  # Single image analysis
│   ├── batch_processing.py    # Batch processing interface
│   └── comparison.py          # Comparison analysis tools
├── tailwind.config.js     # Tailwind configuration
├── package.json           # Node.js dependencies
├── build_css.py           # CSS build script
├── setup_tailwind.py      # Tailwind setup script
└── TAILWIND_SETUP.md      # Detailed Tailwind guide
```

## Pages Overview

### 🏠 Home
- Project overview and introduction to SVD compression
- Mathematical background and theory
- Getting started guide and best practices
- Interactive visualizations and examples

### 🔍 Single Image Compression
- Upload individual images for analysis
- Real-time k-value adjustment with slider
- Side-by-side image comparison
- Quality metrics dashboard (PSNR, SSIM, MSE)
- Download compressed results

### 📊 Batch Processing
- Upload and process multiple images
- Configurable compression parameters
- Progress tracking and error handling
- Comprehensive results summary
- Batch export functionality

### ⚖️ Comparison Analysis
- Compare different compression levels
- Grid view of multiple k-values
- Interactive quality metric charts
- Statistical analysis and reporting
- Export comparison reports

## Styling and Design

The application now uses **Tailwind CSS** for modern, utility-first styling:

- **Design System**: Consistent color palette, typography, and spacing
- **Typography**: Inter font family with JetBrains Mono for code
- **Color Scheme**: Professional slate, blue, and green palette
- **Responsive Layout**: Mobile-first design with breakpoint utilities
- **Interactive Elements**: Smooth animations and hover effects
- **Component Library**: Reusable cards, buttons, and form elements
- **Performance**: Optimized CSS delivery via CDN or build process

### Tailwind CSS Setup

1. **Quick Setup (CDN Mode)**:
   ```bash
   cd webapp
   python setup_tailwind.py cdn
   ```

2. **Advanced Setup (Build Mode)**:
   ```bash
   cd webapp
   python setup_tailwind.py build
   ```

See `TAILWIND_SETUP.md` for detailed configuration and usage guide.

## Configuration

Key settings can be modified in `config.py`:

- File size limits and supported formats
- Default compression parameters
- UI dimensions and styling
- Quality metric thresholds
- Directory paths

## Dependencies

The web application requires:

- `streamlit>=1.25.0` - Web framework
- `plotly>=5.15.0` - Interactive plotting
- Core SVD compression modules from `../src/`

## Development Notes

- Uses Streamlit's multi-page architecture
- Custom CSS for professional appearance
- Modular page structure for maintainability
- Responsive design principles
- Academic presentation standards

## Browser Compatibility

Tested and optimized for:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Performance Considerations

- Image processing is done server-side
- Large images are automatically resized
- Progress indicators for long operations
- Efficient memory management for batch processing