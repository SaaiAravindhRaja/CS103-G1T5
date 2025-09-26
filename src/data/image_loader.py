"""
Image loading and preprocessing utilities for SVD image compression.

This module provides the ImageLoader class for standardized image loading,
preprocessing, and saving operations.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)


class ImageLoader:
    """
    Handles standardized image loading, preprocessing, and saving operations.
    
    This class provides methods for loading images from various formats,
    resizing them to standard dimensions with proper aspect ratio handling,
    normalizing pixel values, and saving processed images.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the ImageLoader with target dimensions.
        
        Args:
            target_size: Target dimensions (width, height) for image resizing.
                        Default is (256, 256).
        """
        self.target_size = target_size
        
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image from file and apply standard preprocessing.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array with shape (H, W) for grayscale
            or (H, W, 3) for RGB, with pixel values normalized to [0, 1]
            
        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image cannot be loaded or processed
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        try:
            # Load image using PIL
            with Image.open(image_path) as img:
                # Convert to RGB if needed (handles RGBA, P, etc.)
                if img.mode not in ['RGB', 'L']:
                    if img.mode == 'RGBA':
                        # Create white background for transparent images
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert('RGB')
                
                # Resize with proper aspect ratio handling
                resized_img = self._resize_with_aspect_ratio(img)
                
                # Convert to numpy array and normalize
                img_array = np.array(resized_img, dtype=np.float64)
                normalized_img = self._normalize_pixels(img_array)
                
                logger.debug(f"Loaded image {image_path.name}: shape {normalized_img.shape}")
                return normalized_img
                
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
    
    def _resize_with_aspect_ratio(self, img: Image.Image) -> Image.Image:
        """
        Resize image to target size while maintaining aspect ratio.
        
        The image is resized to fit within the target dimensions, then padded
        with white pixels to reach the exact target size.
        
        Args:
            img: PIL Image to resize
            
        Returns:
            Resized PIL Image with exact target dimensions
        """
        # Calculate scaling factor to fit within target size
        width_ratio = self.target_size[0] / img.width
        height_ratio = self.target_size[1] / img.height
        scale_factor = min(width_ratio, height_ratio)
        
        # Calculate new dimensions
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        
        # Resize image
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size and white background
        # Use appropriate color format for the image mode
        if img.mode == 'L':
            background_color = 255  # White for grayscale
        else:
            background_color = (255, 255, 255)  # White for RGB
            
        final_img = Image.new(img.mode, self.target_size, background_color)
        
        # Calculate position to center the resized image
        x_offset = (self.target_size[0] - new_width) // 2
        y_offset = (self.target_size[1] - new_height) // 2
        
        # Paste resized image onto white background
        final_img.paste(resized_img, (x_offset, y_offset))
        
        return final_img
    
    def _normalize_pixels(self, img_array: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            img_array: Image array with pixel values in [0, 255] range
            
        Returns:
            Normalized image array with pixel values in [0, 1] range
        """
        return img_array / 255.0
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path], 
                   format: str = 'PNG') -> None:
        """
        Save a normalized image array to file.
        
        Args:
            image: Image array with pixel values in [0, 1] range
            output_path: Path where to save the image
            format: Image format ('PNG', 'JPEG', etc.). Default is 'PNG'.
            
        Raises:
            ValueError: If the image array has invalid shape or values
        """
        output_path = Path(output_path)
        
        # Validate input
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
            
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D array, got {image.ndim}D")
            
        if image.ndim == 3 and image.shape[2] not in [1, 3]:
            raise ValueError(f"3D image must have 1 or 3 channels, got {image.shape[2]}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Denormalize to [0, 255] range and convert to uint8
            denormalized = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            if denormalized.ndim == 2:
                pil_img = Image.fromarray(denormalized)
            else:
                if denormalized.shape[2] == 1:
                    pil_img = Image.fromarray(denormalized.squeeze())
                else:
                    pil_img = Image.fromarray(denormalized)
            
            # Save image
            pil_img.save(output_path, format=format)
            logger.debug(f"Saved image to {output_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save image to {output_path}: {str(e)}")
    
    def load_as_grayscale(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image and convert it to grayscale.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Grayscale image as numpy array with shape (H, W) and values in [0, 1]
        """
        image_path = Path(image_path)
        
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale
                grayscale_img = img.convert('L')
                
                # Resize with proper aspect ratio handling
                resized_img = self._resize_with_aspect_ratio(grayscale_img)
                
                # Convert to numpy array and normalize
                img_array = np.array(resized_img, dtype=np.float64)
                normalized_img = self._normalize_pixels(img_array)
                
                logger.debug(f"Loaded grayscale image {image_path.name}: shape {normalized_img.shape}")
                return normalized_img
                
        except Exception as e:
            raise ValueError(f"Failed to load grayscale image {image_path}: {str(e)}")
    
    def load_as_rgb(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load an image and ensure it's in RGB format.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            RGB image as numpy array with shape (H, W, 3) and values in [0, 1]
        """
        image_path = Path(image_path)
        
        try:
            with Image.open(image_path) as img:
                # Convert to RGB
                if img.mode != 'RGB':
                    if img.mode == 'RGBA':
                        # Create white background for transparent images
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert('RGB')
                
                # Resize with proper aspect ratio handling
                resized_img = self._resize_with_aspect_ratio(img)
                
                # Convert to numpy array and normalize
                img_array = np.array(resized_img, dtype=np.float64)
                normalized_img = self._normalize_pixels(img_array)
                
                logger.debug(f"Loaded RGB image {image_path.name}: shape {normalized_img.shape}")
                return normalized_img
                
        except Exception as e:
            raise ValueError(f"Failed to load RGB image {image_path}: {str(e)}")