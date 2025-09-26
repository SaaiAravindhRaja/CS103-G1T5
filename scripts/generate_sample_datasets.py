#!/usr/bin/env python3
"""
Generate sample datasets for SVD image compression demonstration.

This script creates synthetic images representing different categories:
- Portraits: Smooth gradients and low-frequency content
- Landscapes: Mixed frequency content with natural patterns
- Textures: High-frequency content and complex patterns

The generated images are designed to demonstrate different compression
characteristics when using SVD compression.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
from pathlib import Path
import json
from datetime import datetime

def create_portrait_samples():
    """Create synthetic portrait-like images with smooth gradients."""
    portraits = []
    
    # Portrait 1: Smooth circular gradient (face-like)
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    
    # Create face-like ellipse with gradient
    for i in range(50):
        color_val = 220 - i * 2
        color = (color_val, color_val - 20, color_val - 10)
        draw.ellipse([78 + i, 64 + i, 178 - i, 192 - i], fill=color)
    
    # Add simple features
    draw.ellipse([100, 100, 110, 110], fill=(50, 50, 50))  # Left eye
    draw.ellipse([146, 100, 156, 110], fill=(50, 50, 50))  # Right eye
    draw.ellipse([120, 130, 136, 140], fill=(100, 50, 50))  # Nose
    draw.arc([110, 150, 146, 170], 0, 180, fill=(80, 40, 40), width=3)  # Mouth
    
    portraits.append(('portrait_smooth_gradient.png', img))
    
    # Portrait 2: Soft geometric portrait
    img = Image.new('RGB', (256, 256), (240, 235, 230))
    draw = ImageDraw.Draw(img)
    
    # Background gradient
    for y in range(256):
        color_val = int(240 - y * 0.2)
        color = (color_val, color_val - 5, color_val - 10)
        draw.line([(0, y), (256, y)], fill=color)
    
    # Simple geometric face
    draw.ellipse([64, 48, 192, 208], fill=(220, 200, 180))
    draw.ellipse([90, 90, 110, 110], fill=(100, 80, 60))  # Left eye
    draw.ellipse([146, 90, 166, 110], fill=(100, 80, 60))  # Right eye
    draw.rectangle([120, 120, 136, 150], fill=(180, 160, 140))  # Nose
    draw.ellipse([110, 160, 146, 180], fill=(160, 120, 120))  # Mouth
    
    portraits.append(('portrait_geometric.png', img))
    
    # Portrait 3: Minimalist portrait with smooth transitions
    img = Image.new('RGB', (256, 256), (250, 248, 245))
    draw = ImageDraw.Draw(img)
    
    # Create smooth background
    center_x, center_y = 128, 128
    for x in range(256):
        for y in range(256):
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            intensity = max(0, 255 - dist * 0.8)
            color = (int(intensity), int(intensity * 0.95), int(intensity * 0.9))
            img.putpixel((x, y), color)
    
    # Add subtle features
    draw.ellipse([100, 100, 120, 115], fill=(200, 180, 160))  # Left eye area
    draw.ellipse([136, 100, 156, 115], fill=(200, 180, 160))  # Right eye area
    
    portraits.append(('portrait_minimalist.png', img))
    
    return portraits

def create_landscape_samples():
    """Create synthetic landscape-like images with mixed frequencies."""
    landscapes = []
    
    # Landscape 1: Simple horizon with gradient sky
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    
    # Sky gradient
    for y in range(150):
        blue_val = int(135 + y * 0.8)
        color = (100 + y // 3, 150 + y // 4, min(255, blue_val))
        draw.line([(0, y), (256, y)], fill=color)
    
    # Ground
    for y in range(150, 256):
        green_val = int(100 - (y - 150) * 0.3)
        color = (60, max(50, green_val), 40)
        draw.line([(0, y), (256, y)], fill=color)
    
    # Add some hills
    points = [(0, 150), (64, 120), (128, 140), (192, 110), (256, 130), (256, 256), (0, 256)]
    draw.polygon(points, fill=(80, 120, 60))
    
    landscapes.append(('landscape_horizon.png', img))
    
    # Landscape 2: Mountain silhouette
    img = Image.new('RGB', (256, 256), (135, 180, 220))
    draw = ImageDraw.Draw(img)
    
    # Sky with clouds effect
    for y in range(180):
        noise = np.sin(y * 0.1) * 10
        blue_val = int(135 + y * 0.4 + noise)
        color = (100 + y // 4, 150 + y // 3, min(255, blue_val))
        draw.line([(0, y), (256, y)], fill=color)
    
    # Mountain ranges
    mountain1 = [(0, 180), (50, 100), (100, 120), (150, 80), (200, 110), (256, 90), (256, 256), (0, 256)]
    draw.polygon(mountain1, fill=(60, 80, 100))
    
    mountain2 = [(0, 200), (80, 140), (160, 160), (240, 130), (256, 140), (256, 256), (0, 256)]
    draw.polygon(mountain2, fill=(80, 100, 120))
    
    landscapes.append(('landscape_mountains.png', img))
    
    # Landscape 3: Abstract natural pattern
    img = Image.new('RGB', (256, 256), 'white')
    
    # Create noise-based natural pattern
    np.random.seed(42)  # For reproducibility
    noise = np.random.rand(256, 256)
    
    # Apply smoothing to create natural-looking patterns
    for x in range(256):
        for y in range(256):
            # Create layered noise for natural effect
            val1 = noise[x, y]
            val2 = np.sin(x * 0.02) * np.cos(y * 0.02) * 0.5 + 0.5
            val3 = np.sin(x * 0.05 + y * 0.03) * 0.3 + 0.7
            
            combined = (val1 * 0.4 + val2 * 0.3 + val3 * 0.3)
            
            # Map to earth tones
            r = int(80 + combined * 100)
            g = int(60 + combined * 120)
            b = int(40 + combined * 80)
            
            img.putpixel((x, y), (r, g, b))
    
    # Apply slight blur for more natural look
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    
    landscapes.append(('landscape_natural_pattern.png', img))
    
    return landscapes

def create_texture_samples():
    """Create synthetic texture images with high-frequency content."""
    textures = []
    
    # Texture 1: Checkerboard pattern
    img = Image.new('RGB', (256, 256), 'white')
    draw = ImageDraw.Draw(img)
    
    square_size = 16
    for x in range(0, 256, square_size):
        for y in range(0, 256, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                color = (40, 40, 40)
            else:
                color = (220, 220, 220)
            draw.rectangle([x, y, x + square_size, y + square_size], fill=color)
    
    textures.append(('texture_checkerboard.png', img))
    
    # Texture 2: Random noise pattern
    img = Image.new('RGB', (256, 256), 'white')
    
    np.random.seed(123)  # For reproducibility
    noise = np.random.randint(0, 256, (256, 256, 3))
    
    for x in range(256):
        for y in range(256):
            r, g, b = noise[x, y]
            img.putpixel((x, y), (int(r), int(g), int(b)))
    
    textures.append(('texture_noise.png', img))
    
    # Texture 3: Geometric pattern
    img = Image.new('RGB', (256, 256), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    
    # Create repeating diamond pattern
    for x in range(0, 256, 32):
        for y in range(0, 256, 32):
            # Diamond shape
            points = [(x + 16, y), (x + 32, y + 16), (x + 16, y + 32), (x, y + 16)]
            color_intensity = ((x + y) // 32) % 2
            if color_intensity:
                color = (100, 150, 200)
            else:
                color = (200, 150, 100)
            draw.polygon(points, fill=color)
    
    textures.append(('texture_geometric.png', img))
    
    # Texture 4: High-frequency sine wave pattern
    img = Image.new('RGB', (256, 256), 'white')
    
    for x in range(256):
        for y in range(256):
            # Create high-frequency interference pattern
            val1 = np.sin(x * 0.3) * np.sin(y * 0.3)
            val2 = np.sin(x * 0.7 + y * 0.5)
            val3 = np.sin(x * 0.2 - y * 0.4)
            
            combined = (val1 + val2 + val3) / 3
            intensity = int(128 + combined * 127)
            
            img.putpixel((x, y), (intensity, intensity, intensity))
    
    textures.append(('texture_interference.png', img))
    
    return textures

def create_sample_manifest(portraits, landscapes, textures):
    """Create a manifest file describing all sample images."""
    manifest = {
        "dataset_info": {
            "name": "SVD Compression Sample Dataset",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "description": "Synthetic sample images for demonstrating SVD compression characteristics",
            "total_images": len(portraits) + len(landscapes) + len(textures)
        },
        "categories": {
            "portraits": {
                "description": "Images with smooth gradients and low-frequency content, ideal for SVD compression",
                "characteristics": ["smooth gradients", "low-frequency content", "high compression ratios"],
                "images": [name for name, _ in portraits]
            },
            "landscapes": {
                "description": "Images with mixed frequency content representing natural scenes",
                "characteristics": ["mixed frequencies", "natural patterns", "moderate compression"],
                "images": [name for name, _ in landscapes]
            },
            "textures": {
                "description": "Images with high-frequency content and complex patterns",
                "characteristics": ["high-frequency content", "complex patterns", "lower compression ratios"],
                "images": [name for name, _ in textures]
            }
        },
        "technical_specs": {
            "image_size": "256x256 pixels",
            "color_mode": "RGB",
            "file_format": "PNG",
            "bit_depth": "8-bit per channel"
        },
        "usage_notes": [
            "These images are designed to demonstrate different SVD compression characteristics",
            "Portraits should show the best compression ratios with high quality",
            "Textures should show the challenges of compressing high-frequency content",
            "Landscapes provide intermediate examples between the extremes"
        ]
    }
    
    return manifest

def main():
    """Generate all sample datasets and documentation."""
    print("Generating sample datasets for SVD image compression...")
    
    # Create directories if they don't exist
    data_dir = Path("data")
    portraits_dir = data_dir / "portraits"
    landscapes_dir = data_dir / "landscapes"
    textures_dir = data_dir / "textures"
    samples_dir = data_dir / "samples"
    
    for directory in [portraits_dir, landscapes_dir, textures_dir, samples_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Generate sample images
    print("Creating portrait samples...")
    portraits = create_portrait_samples()
    
    print("Creating landscape samples...")
    landscapes = create_landscape_samples()
    
    print("Creating texture samples...")
    textures = create_texture_samples()
    
    # Save images
    print("Saving images...")
    
    for name, img in portraits:
        img.save(portraits_dir / name)
        print(f"  Saved {name} to portraits/")
    
    for name, img in landscapes:
        img.save(landscapes_dir / name)
        print(f"  Saved {name} to landscapes/")
    
    for name, img in textures:
        img.save(textures_dir / name)
        print(f"  Saved {name} to textures/")
    
    # Create manifest
    print("Creating dataset manifest...")
    manifest = create_sample_manifest(portraits, landscapes, textures)
    
    with open(data_dir / "dataset_manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create README for data directory
    readme_content = """# Sample Dataset for SVD Image Compression

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
"""
    
    with open(data_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\nDataset generation complete!")
    print(f"Generated {len(portraits)} portraits, {len(landscapes)} landscapes, {len(textures)} textures")
    print(f"Total: {len(portraits) + len(landscapes) + len(textures)} sample images")
    print(f"Manifest saved to: {data_dir / 'dataset_manifest.json'}")
    print(f"Documentation saved to: {data_dir / 'README.md'}")

if __name__ == "__main__":
    main()