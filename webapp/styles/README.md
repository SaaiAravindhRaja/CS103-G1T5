# Tailwind CSS Setup for SVD Image Compression Webapp

This directory contains the Tailwind CSS configuration and build files for the webapp.

## Quick Start

1. **Install dependencies:**
   ```bash
   python build_css.py install
   ```

2. **Build CSS for development:**
   ```bash
   python build_css.py build
   ```

3. **Build CSS for production:**
   ```bash
   python build_css.py build-prod
   ```

4. **Watch for changes during development:**
   ```bash
   python build_css.py watch
   ```

## File Structure

- `input.css` - Source Tailwind CSS file with custom styles
- `output.css` - Generated CSS file (created after build)
- `../tailwind.config.js` - Tailwind configuration
- `../package.json` - Node.js dependencies

## Design System

### Colors
- **Primary**: Slate colors (50, 500, 700, 900)
- **Secondary**: Blue colors (50, 500, 600, 700)  
- **Accent**: Green colors (50, 500, 600)

### Typography
- **Font Family**: Inter (Google Fonts)
- **Monospace**: JetBrains Mono

### Components
- **Buttons**: Gradient backgrounds with hover effects
- **Cards**: White background with subtle shadows
- **Upload Zones**: Dashed borders with hover states
- **Metrics**: Centered cards with large values

### Animations
- **Fade In**: Smooth opacity and translate animations
- **Slide In**: Horizontal slide animations
- **Slide Up**: Vertical slide animations

## Usage in Streamlit

The CSS is automatically loaded via the `load_tailwind_css()` function in `utils/styling.py`. Custom classes can be applied using Streamlit's `st.markdown()` with `unsafe_allow_html=True`.

## Development Workflow

1. Make changes to `input.css`
2. Run `python build_css.py watch` to auto-rebuild
3. Refresh your Streamlit app to see changes
4. For production, run `python build_css.py build-prod`

## Troubleshooting

- **Node.js not found**: Install Node.js from https://nodejs.org/
- **Build fails**: Check that all dependencies are installed with `npm install`
- **Styles not updating**: Make sure to rebuild CSS after changes
- **Streamlit not loading styles**: Check that `load_tailwind_css()` is called in app.py