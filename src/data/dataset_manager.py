"""
Dataset management and organization utilities for SVD image compression.

This module provides the DatasetManager class for organizing image datasets
by categories, generating both grayscale and RGB versions, and creating
data manifests.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

from .image_loader import ImageLoader

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Metadata for an image in the dataset."""
    filename: str
    width: int
    height: int
    dataset_label: str
    channels: int
    file_size: int
    original_path: str


class DatasetManager:
    """
    Manages image datasets with categorization and preprocessing.
    
    This class organizes images into categories (portraits, landscapes, textures),
    generates both grayscale and RGB versions, and creates comprehensive
    data manifests for systematic experiments.
    """
    
    def __init__(self, data_root: Union[str, Path] = "data", 
                 target_size: Tuple[int, int] = (256, 256)):
        """
        Initialize the DatasetManager.
        
        Args:
            data_root: Root directory for dataset storage
            target_size: Target dimensions for image preprocessing
        """
        self.data_root = Path(data_root)
        self.target_size = target_size
        self.image_loader = ImageLoader(target_size=target_size)
        
        # Define dataset categories
        self.categories = {
            'portraits': self.data_root / 'portraits',
            'landscapes': self.data_root / 'landscapes', 
            'textures': self.data_root / 'textures'
        }
        
        # Supported image extensions
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
        
    def setup_directory_structure(self) -> None:
        """
        Create the directory structure for organizing datasets.
        
        Creates directories for each category and subdirectories for
        processed versions (grayscale, rgb).
        """
        logger.info("Setting up dataset directory structure")
        
        for category, category_path in self.categories.items():
            # Create main category directory
            category_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for processed versions
            (category_path / 'original').mkdir(exist_ok=True)
            (category_path / 'grayscale').mkdir(exist_ok=True)
            (category_path / 'rgb').mkdir(exist_ok=True)
            
            logger.debug(f"Created directory structure for {category}")
    
    def discover_images(self, category: Optional[str] = None) -> Dict[str, List[Path]]:
        """
        Discover images in the dataset directories.
        
        Args:
            category: Specific category to discover, or None for all categories
            
        Returns:
            Dictionary mapping category names to lists of image paths
        """
        discovered_images = {}
        
        categories_to_scan = [category] if category else self.categories.keys()
        
        for cat in categories_to_scan:
            if cat not in self.categories:
                logger.warning(f"Unknown category: {cat}")
                continue
                
            category_path = self.categories[cat]
            original_path = category_path / 'original'
            
            if not original_path.exists():
                logger.warning(f"Original images directory not found: {original_path}")
                discovered_images[cat] = []
                continue
            
            # Find all supported image files
            image_files = []
            for ext in self.supported_extensions:
                image_files.extend(original_path.glob(f"*{ext}"))
                image_files.extend(original_path.glob(f"*{ext.upper()}"))
            
            discovered_images[cat] = sorted(image_files)
            logger.info(f"Discovered {len(image_files)} images in {cat} category")
        
        return discovered_images
    
    def load_datasets(self) -> Dict[str, Dict[str, List[np.ndarray]]]:
        """
        Load all datasets with both grayscale and RGB versions.
        
        Returns:
            Nested dictionary: {category: {'grayscale': [...], 'rgb': [...]}}
        """
        logger.info("Loading datasets")
        datasets = {}
        
        discovered_images = self.discover_images()
        
        for category, image_paths in discovered_images.items():
            if not image_paths:
                logger.warning(f"No images found in {category} category")
                datasets[category] = {'grayscale': [], 'rgb': []}
                continue
            
            grayscale_images = []
            rgb_images = []
            
            for img_path in image_paths:
                try:
                    # Load grayscale version
                    grayscale_img = self.image_loader.load_as_grayscale(img_path)
                    grayscale_images.append(grayscale_img)
                    
                    # Load RGB version
                    rgb_img = self.image_loader.load_as_rgb(img_path)
                    rgb_images.append(rgb_img)
                    
                    logger.debug(f"Loaded {img_path.name} in both formats")
                    
                except Exception as e:
                    logger.error(f"Failed to load {img_path}: {str(e)}")
                    continue
            
            datasets[category] = {
                'grayscale': grayscale_images,
                'rgb': rgb_images
            }
            
            logger.info(f"Loaded {len(grayscale_images)} images from {category}")
        
        return datasets
    
    def generate_processed_versions(self, overwrite: bool = False) -> None:
        """
        Generate and save grayscale and RGB versions of all original images.
        
        Args:
            overwrite: Whether to overwrite existing processed images
        """
        logger.info("Generating processed versions of images")
        
        discovered_images = self.discover_images()
        
        for category, image_paths in discovered_images.items():
            category_path = self.categories[category]
            grayscale_dir = category_path / 'grayscale'
            rgb_dir = category_path / 'rgb'
            
            for img_path in image_paths:
                # Define output paths
                base_name = img_path.stem
                grayscale_output = grayscale_dir / f"{base_name}_gray.png"
                rgb_output = rgb_dir / f"{base_name}_rgb.png"
                
                # Skip if files exist and not overwriting
                if not overwrite and grayscale_output.exists() and rgb_output.exists():
                    logger.debug(f"Skipping {img_path.name} - processed versions exist")
                    continue
                
                try:
                    # Generate and save grayscale version
                    grayscale_img = self.image_loader.load_as_grayscale(img_path)
                    self.image_loader.save_image(grayscale_img, grayscale_output)
                    
                    # Generate and save RGB version
                    rgb_img = self.image_loader.load_as_rgb(img_path)
                    self.image_loader.save_image(rgb_img, rgb_output)
                    
                    logger.debug(f"Generated processed versions for {img_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {img_path}: {str(e)}")
                    continue
        
        logger.info("Completed generating processed versions")
    
    def generate_manifest(self, output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Generate a comprehensive data manifest CSV with image metadata.
        
        Args:
            output_path: Path to save the manifest CSV, or None to return only
            
        Returns:
            DataFrame containing image metadata
        """
        logger.info("Generating data manifest")
        
        manifest_data = []
        discovered_images = self.discover_images()
        
        for category, image_paths in discovered_images.items():
            for img_path in image_paths:
                try:
                    # Load image to get dimensions
                    img_array = self.image_loader.load_image(img_path)
                    
                    # Determine number of channels
                    if img_array.ndim == 2:
                        channels = 1
                        height, width = img_array.shape
                    else:
                        channels = img_array.shape[2]
                        height, width = img_array.shape[:2]
                    
                    # Get file size
                    file_size = img_path.stat().st_size
                    
                    # Create metadata entry
                    metadata = ImageMetadata(
                        filename=img_path.name,
                        width=width,
                        height=height,
                        dataset_label=category,
                        channels=channels,
                        file_size=file_size,
                        original_path=str(img_path.relative_to(self.data_root))
                    )
                    
                    manifest_data.append({
                        'filename': metadata.filename,
                        'width': metadata.width,
                        'height': metadata.height,
                        'dataset_label': metadata.dataset_label,
                        'channels': metadata.channels,
                        'file_size_bytes': metadata.file_size,
                        'original_path': metadata.original_path
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process metadata for {img_path}: {str(e)}")
                    continue
        
        # Create DataFrame
        manifest_df = pd.DataFrame(manifest_data)
        
        # Add summary statistics
        if not manifest_df.empty:
            logger.info(f"Generated manifest with {len(manifest_df)} images")
            logger.info(f"Categories: {manifest_df['dataset_label'].value_counts().to_dict()}")
        
        # Save to file if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_df.to_csv(output_path, index=False)
            logger.info(f"Saved manifest to {output_path}")
        
        return manifest_df
    
    def get_category_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about each dataset category.
        
        Returns:
            Dictionary with category statistics
        """
        stats = {}
        discovered_images = self.discover_images()
        
        for category, image_paths in discovered_images.items():
            stats[category] = {
                'total_images': len(image_paths),
                'total_size_bytes': sum(p.stat().st_size for p in image_paths if p.exists())
            }
        
        return stats
    
    def validate_dataset(self) -> Dict[str, List[str]]:
        """
        Validate the dataset structure and report any issues.
        
        Returns:
            Dictionary with validation results and issues found
        """
        issues = {
            'missing_directories': [],
            'empty_categories': [],
            'corrupted_images': [],
            'missing_processed': []
        }
        
        # Check directory structure
        for category, category_path in self.categories.items():
            if not category_path.exists():
                issues['missing_directories'].append(str(category_path))
                continue
            
            original_path = category_path / 'original'
            if not original_path.exists():
                issues['missing_directories'].append(str(original_path))
        
        # Check for empty categories and validate images
        discovered_images = self.discover_images()
        
        for category, image_paths in discovered_images.items():
            if not image_paths:
                issues['empty_categories'].append(category)
                continue
            
            category_path = self.categories[category]
            
            for img_path in image_paths:
                # Check if image can be loaded
                try:
                    self.image_loader.load_image(img_path)
                except Exception:
                    issues['corrupted_images'].append(str(img_path))
                
                # Check if processed versions exist
                base_name = img_path.stem
                grayscale_path = category_path / 'grayscale' / f"{base_name}_gray.png"
                rgb_path = category_path / 'rgb' / f"{base_name}_rgb.png"
                
                if not grayscale_path.exists() or not rgb_path.exists():
                    issues['missing_processed'].append(str(img_path))
        
        return issues