#!/usr/bin/env python3
"""
Script to create an animated GIF demonstration of SVD image compression.
Shows the compression process with different k-values.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available. Install with: pip install Pillow")

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("⚠️ imageio not available. Install with: pip install imageio")

# Add src to path for imports
sys.path.append('../src')

try:
    from compression.svd_compressor import SVDCompressor
    from data.image_loader import ImageLoader
    from evaluation.metrics_calculator import MetricsCalculator
    SVD_MODULES_AVAILABLE = True
except ImportError:
    SVD_MODULES_AVAILABLE = False
    print("⚠️ SVD modules not available. Make sure src/ directory is accessible.")

class DemoGifGenerator:
    """Generate animated GIF demonstration of SVD compression."""
    
    def __init__(self):
        """Initialize the GIF generator."""
        self.demo_dir = Path(__file__).parent
        self.output_dir = self.demo_dir / 'output'
        self.output_dir.mkdir(exist_ok=True)
        
        if SVD_MODULES_AVAILABLE:
            self.compressor = SVDCompressor()
            self.image_loader = ImageLoader()
            self.metrics_calc = MetricsCalculator()
    
    def create_sample_image(self):
        """Create a sample image for demonstration."""
        # Create a simple synthetic image with clear structure
        size = 128  # Smaller for faster processing
        image = np.zeros((size, size, 3))
        
        # Add some geometric patterns
        center = size // 2
        
        # Background gradient
        for i in range(size):
            for j in range(size):
                image[i, j, 0] = (i / size) * 0.3 + 0.2  # Red channel
                image[i, j, 1] = (j / size) * 0.3 + 0.2  # Green channel
                image[i, j, 2] = 0.4  # Blue channel
        
        # Add circles
        y, x = np.ogrid[:size, :size]
        
        # Large circle
        mask1 = (x - center)**2 + (y - center)**2 <= (size//4)**2
        image[mask1] = [0.8, 0.6, 0.4]
        
        # Small circle
        mask2 = (x - center//2)**2 + (y - center//2)**2 <= (size//8)**2
        image[mask2] = [0.2, 0.8, 0.6]
        
        # Rectangle
        image[center-10:center+10, center+20:center+40] = [0.9, 0.3, 0.7]
        
        return image
    
    def create_compression_frames(self, image, k_values):
        """Create frames showing compression at different k-values."""
        frames = []
        
        # Original image frame
        original_frame = self.create_frame_with_text(
            image, "Original Image", "No compression", ""
        )
        frames.append(original_frame)
        
        # Add pause frames for original
        for _ in range(10):  # Hold original for 1 second at 10fps
            frames.append(original_frame)
        
        # Compression frames
        for k in k_values:
            if SVD_MODULES_AVAILABLE:
                # Use actual SVD compression
                compressed_img, metadata = self.compressor.compress_image(image, k)
                psnr = self.metrics_calc.calculate_psnr(image, compressed_img)
                ssim = self.metrics_calc.calculate_ssim(image, compressed_img)
                
                title = f"k = {k} singular values"
                subtitle = f"Compression: {metadata['compression_ratio']:.1f}x"
                metrics = f"PSNR: {psnr:.1f}dB, SSIM: {ssim:.3f}"
            else:
                # Create mock compression for demo
                compressed_img = self.mock_compression(image, k)
                compression_ratio = (128*128) / (k * (128 + 128 + 1))
                
                title = f"k = {k} singular values"
                subtitle = f"Compression: {compression_ratio:.1f}x"
                metrics = f"Simulated compression"
            
            frame = self.create_frame_with_text(
                compressed_img, title, subtitle, metrics
            )
            frames.append(frame)
            
            # Hold each compression frame
            for _ in range(8):  # Hold for 0.8 seconds
                frames.append(frame)
        
        return frames
    
    def mock_compression(self, image, k):
        """Create mock compression effect for demonstration."""
        # Simple blur effect to simulate compression
        from scipy import ndimage
        
        # More blur for lower k values
        sigma = max(0.5, 3.0 - (k / 20.0))
        
        compressed = np.zeros_like(image)
        for channel in range(3):
            compressed[:, :, channel] = ndimage.gaussian_filter(
                image[:, :, channel], sigma=sigma
            )
        
        # Add some quantization effect
        compressed = np.round(compressed * (k * 2)) / (k * 2)
        compressed = np.clip(compressed, 0, 1)
        
        return compressed
    
    def create_frame_with_text(self, image, title, subtitle, metrics):
        """Create a frame with image and text overlay."""
        if not PIL_AVAILABLE:
            return (image * 255).astype(np.uint8)
        
        # Convert to PIL Image
        img_pil = Image.fromarray((image * 255).astype(np.uint8))
        
        # Create a larger canvas for text
        canvas_width = 400
        canvas_height = 300
        canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
        
        # Resize and center the image
        img_resized = img_pil.resize((200, 200), Image.Resampling.LANCZOS)
        img_x = (canvas_width - 200) // 2
        img_y = 20
        canvas.paste(img_resized, (img_x, img_y))
        
        # Add text
        draw = ImageDraw.Draw(canvas)
        
        try:
            # Try to use a nice font
            title_font = ImageFont.truetype("Arial.ttf", 16)
            subtitle_font = ImageFont.truetype("Arial.ttf", 12)
            metrics_font = ImageFont.truetype("Arial.ttf", 10)
        except:
            # Fallback to default font
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            metrics_font = ImageFont.load_default()
        
        # Title
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (canvas_width - title_width) // 2
        draw.text((title_x, 230), title, fill='black', font=title_font)
        
        # Subtitle
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (canvas_width - subtitle_width) // 2
        draw.text((subtitle_x, 250), subtitle, fill='blue', font=subtitle_font)
        
        # Metrics
        if metrics:
            metrics_bbox = draw.textbbox((0, 0), metrics, font=metrics_font)
            metrics_width = metrics_bbox[2] - metrics_bbox[0]
            metrics_x = (canvas_width - metrics_width) // 2
            draw.text((metrics_x, 270), metrics, fill='gray', font=metrics_font)
        
        return np.array(canvas)
    
    def create_demo_gif(self):
        """Create the complete demo GIF."""
        if not IMAGEIO_AVAILABLE:
            print("❌ Cannot create GIF - imageio not installed")
            return None
        
        print("🎬 Creating SVD compression demo GIF...")
        
        # Create or load sample image
        print("📸 Generating sample image...")
        image = self.create_sample_image()
        
        # Define k-values for demonstration
        k_values = [5, 10, 20, 30, 50]
        
        print(f"🔄 Creating compression frames for k-values: {k_values}")
        frames = self.create_compression_frames(image, k_values)
        
        # Save as GIF
        gif_path = self.output_dir / 'svd_compression_demo.gif'
        print(f"💾 Saving GIF to {gif_path}")
        
        imageio.mimsave(
            str(gif_path),
            frames,
            duration=0.1,  # 100ms per frame = 10fps
            loop=0  # Infinite loop
        )
        
        print(f"🎉 Demo GIF created successfully: {gif_path}")
        return gif_path
    
    def create_static_comparison(self):
        """Create a static comparison image showing different k-values."""
        print("📊 Creating static comparison image...")
        
        image = self.create_sample_image()
        k_values = [5, 15, 30, 50]
        
        fig, axes = plt.subplots(1, len(k_values) + 1, figsize=(15, 3))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original', fontweight='bold')
        axes[0].axis('off')
        
        # Compressed versions
        for i, k in enumerate(k_values, 1):
            if SVD_MODULES_AVAILABLE:
                compressed_img, metadata = self.compressor.compress_image(image, k)
                psnr = self.metrics_calc.calculate_psnr(image, compressed_img)
                title = f'k={k}\nPSNR: {psnr:.1f}dB\n{metadata["compression_ratio"]:.1f}x'
            else:
                compressed_img = self.mock_compression(image, k)
                compression_ratio = (128*128) / (k * (128 + 128 + 1))
                title = f'k={k}\n{compression_ratio:.1f}x compression'
            
            axes[i].imshow(compressed_img)
            axes[i].set_title(title, fontsize=10, fontweight='bold')
            axes[i].axis('off')
        
        plt.suptitle('SVD Image Compression Demonstration', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save static comparison
        static_path = self.output_dir / 'svd_compression_comparison.png'
        plt.savefig(static_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Static comparison saved: {static_path}")
        return static_path

def main():
    """Main function to create demo materials."""
    print("🚀 Starting demo GIF generation...")
    
    generator = DemoGifGenerator()
    
    # Create demo GIF
    gif_path = generator.create_demo_gif()
    
    # Create static comparison
    static_path = generator.create_static_comparison()
    
    if gif_path or static_path:
        print("\n✅ Demo materials created successfully!")
        if gif_path:
            print(f"🎬 Animated GIF: {gif_path}")
        if static_path:
            print(f"📊 Static comparison: {static_path}")
        
        print("\n💡 Use these materials in presentations, documentation, or social media")
        print("📱 The GIF is perfect for README files and quick demonstrations")
    else:
        print("\n❌ Failed to create demo materials")
        print("💡 Make sure required packages are installed:")
        print("   pip install Pillow imageio matplotlib scipy")

if __name__ == '__main__':
    main()