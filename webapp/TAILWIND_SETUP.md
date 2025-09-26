# Tailwind CSS Integration Guide

This document explains the Tailwind CSS integration for the SVD Image Compression webapp redesign.

## Overview

The webapp has been updated to use Tailwind CSS for modern, utility-first styling. This replaces the previous custom CSS approach with a more maintainable and scalable design system.

## Setup Methods

### Method 1: CDN Mode (Recommended)
- ✅ No build process required
- ✅ Automatic updates
- ✅ Quick setup
- ⚠️ Larger initial load (CDN)

### Method 2: Build Mode (Advanced)
- ✅ Optimized bundle size
- ✅ Custom configuration
- ✅ Offline development
- ⚠️ Requires Node.js and build process

## Quick Start

1. **Initialize Tailwind CSS:**
   ```bash
   cd webapp
   python setup_tailwind.py
   ```

2. **Run the webapp:**
   ```bash
   streamlit run app.py
   ```

## Design System

### Color Palette
```css
/* Primary Colors (Slate) */
--primary-50: #f8fafc
--primary-500: #64748b
--primary-700: #334155
--primary-900: #0f172a

/* Secondary Colors (Blue) */
--secondary-50: #eff6ff
--secondary-500: #3b82f6
--secondary-600: #2563eb
--secondary-700: #1d4ed8

/* Accent Colors (Green) */
--accent-50: #ecfdf5
--accent-500: #10b981
--accent-600: #059669
```

### Typography
- **Font Family**: Inter (Google Fonts)
- **Monospace**: JetBrains Mono
- **Sizes**: xs, sm, base, lg, xl, 2xl, 3xl, 4xl, 5xl

### Spacing
- **Base unit**: 0.25rem (4px)
- **Scale**: 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 56, 64

### Shadows
- **Light**: `0 1px 3px rgba(0,0,0,0.1)`
- **Medium**: `0 4px 6px rgba(0,0,0,0.1)`
- **Large**: `0 10px 15px rgba(0,0,0,0.1)`

## Component Classes

### Buttons
```html
<!-- Primary Button -->
<button class="bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg px-6 py-3 font-medium hover:shadow-md transition-all">
    Primary Action
</button>

<!-- Secondary Button -->
<button class="bg-white text-blue-600 border border-blue-300 rounded-lg px-6 py-3 font-medium hover:bg-blue-50">
    Secondary Action
</button>
```

### Cards
```html
<!-- Basic Card -->
<div class="bg-white border border-gray-200 rounded-xl shadow-sm hover:shadow-md transition-all p-6">
    Card Content
</div>

<!-- Metric Card -->
<div class="metric-card">
    <div class="metric-value">42</div>
    <div class="metric-label">PSNR (dB)</div>
</div>
```

### Upload Zone
```html
<div class="upload-zone">
    <p>Drag and drop files here or click to browse</p>
</div>
```

### Status Indicators
```html
<span class="status-processing">Processing...</span>
<span class="status-complete">Complete</span>
<span class="status-error">Error</span>
```

## Animations

### Available Animations
- `animate-fade-in`: Fade in with slide up
- `animate-slide-in`: Slide in from left
- `animate-slide-up`: Slide up from bottom
- `animate-spin`: Rotating spinner
- `animate-pulse`: Pulsing effect

### Usage
```html
<div class="animate-fade-in">Fades in smoothly</div>
<div class="loading-spinner animate-spin">Loading...</div>
```

## Streamlit Integration

### Loading Styles
The styles are automatically loaded via `load_tailwind_css()` in `utils/styling.py`:

```python
from utils.styling import load_tailwind_css

# In your app
load_tailwind_css()
```

### Using Custom Classes
```python
import streamlit as st

# Custom styled component
st.markdown("""
<div class="bg-white rounded-xl shadow-md p-6 mb-4">
    <h3 class="text-xl font-semibold text-gray-900 mb-2">Custom Component</h3>
    <p class="text-gray-600">This uses Tailwind CSS classes.</p>
</div>
""", unsafe_allow_html=True)
```

### Helper Functions
The styling module provides helper functions:

```python
from utils.styling import (
    create_metric_card,
    create_status_indicator,
    show_loading_animation
)

# Create a metric card
create_metric_card("PSNR", "42.5 dB", "Peak Signal-to-Noise Ratio")

# Show status
create_status_indicator("processing", "Compressing image...")

# Loading animation
show_loading_animation("Processing image...")
```

## Development Workflow

### CDN Mode Development
1. Edit Python files with Tailwind classes
2. Refresh Streamlit app to see changes
3. No build step required

### Build Mode Development
1. Edit `styles/input.css` for custom styles
2. Run `python build_css.py watch` for auto-rebuild
3. Edit Python files with Tailwind classes
4. Refresh Streamlit app to see changes

## File Structure

```
webapp/
├── styles/
│   ├── input.css          # Source CSS (build mode)
│   ├── output.css         # Generated CSS (build mode)
│   ├── .cdn-mode          # CDN mode indicator
│   └── README.md          # Styles documentation
├── utils/
│   └── styling.py         # Tailwind integration
├── tailwind.config.js     # Tailwind configuration
├── package.json           # Node.js dependencies
├── build_css.py           # CSS build script
├── setup_tailwind.py      # Setup script
└── TAILWIND_SETUP.md      # This file
```

## Troubleshooting

### Common Issues

1. **Styles not loading**
   - Check that `load_tailwind_css()` is called in app.py
   - Verify internet connection (CDN mode)
   - Check browser console for errors

2. **Custom classes not working**
   - Ensure classes are valid Tailwind utilities
   - Check spelling and syntax
   - Verify CDN is loading properly

3. **Build mode issues**
   - Install Node.js from https://nodejs.org/
   - Run `npm install` in webapp directory
   - Check `build_css.py` output for errors

4. **Performance issues**
   - Consider switching to build mode for production
   - Optimize custom CSS in `input.css`
   - Use Tailwind's purge feature

### Getting Help

1. Check the [Tailwind CSS documentation](https://tailwindcss.com/docs)
2. Review the design system in this file
3. Look at existing component examples in the codebase
4. Test with simple HTML first, then integrate with Streamlit

## Migration from Custom CSS

The old custom CSS has been replaced with Tailwind utilities:

| Old CSS Class | New Tailwind Classes |
|---------------|---------------------|
| `.metric-card` | `bg-white border border-gray-200 rounded-xl p-6 shadow-sm` |
| `.info-card` | `bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-xl p-6` |
| `.loading-spinner` | `border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin` |
| `.nav-container` | `bg-gradient-to-br from-slate-700 to-blue-600 rounded-lg shadow-md` |

## Best Practices

1. **Use semantic class names** for complex components
2. **Leverage Tailwind's responsive prefixes** (sm:, md:, lg:, xl:)
3. **Combine utilities** for complex designs
4. **Use CSS variables** for consistent theming
5. **Test across different screen sizes**
6. **Keep custom CSS minimal** - prefer Tailwind utilities

## Performance Optimization

### CDN Mode
- Tailwind CSS loads from CDN (~3MB uncompressed)
- Browser caching improves subsequent loads
- Consider build mode for production

### Build Mode
- Only includes used utilities (~10-50KB typical)
- Faster loading and better performance
- Requires build process setup

## Future Enhancements

1. **Dark mode support** using Tailwind's dark: prefix
2. **Custom component library** with reusable classes
3. **Animation library** with more complex transitions
4. **Theme customization** for different use cases
5. **Performance monitoring** and optimization