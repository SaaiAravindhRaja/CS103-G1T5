"""
Integration tests for the DatasetManager class.
"""

import pytest
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import tempfile
import shutil
from typing import List

from src.data.dataset_manager import DatasetManager, ImageMetadata


class TestDatasetManagerIntegration:
    """Integration test cases for DatasetManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dataset_manager = DatasetManager(data_root=self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def create_test_images(self, category: str, count: int = 3) -> List[Path]:
        """Create test images in a category."""
        category_path = self.temp_dir / category / 'original'
        category_path.mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for i in range(count):
            # Create different types of test images
            if i % 3 == 0:
                # RGB image
                img = Image.new('RGB', (100, 100), (255, 0, 0))
            elif i % 3 == 1:
                # Grayscale image
                img = Image.new('L', (150, 150), 128)
            else:
                # Different sized RGB image
                img = Image.new('RGB', (200, 50), (0, 255, 0))
            
            img_path = category_path / f"test_image_{i}.png"
            img.save(img_path)
            image_paths.append(img_path)
        
        return image_paths
    
    def test_setup_directory_structure(self):
        """Test setting up the complete directory structure."""
        self.dataset_manager.setup_directory_structure()
        
        # Check that all category directories exist
        for category in ['portraits', 'landscapes', 'textures']:
            category_path = self.temp_dir / category
            assert category_path.exists()
            assert (category_path / 'original').exists()
            assert (category_path / 'grayscale').exists()
            assert (category_path / 'rgb').exists()
    
    def test_discover_images_all_categories(self):
        """Test discovering images across all categories."""
        # Create test images in multiple categories
        self.create_test_images('portraits', 2)
        self.create_test_images('landscapes', 3)
        self.create_test_images('textures', 1)
        
        discovered = self.dataset_manager.discover_images()
        
        assert 'portraits' in discovered
        assert 'landscapes' in discovered
        assert 'textures' in discovered
        assert len(discovered['portraits']) == 2
        assert len(discovered['landscapes']) == 3
        assert len(discovered['textures']) == 1
    
    def test_discover_images_specific_category(self):
        """Test discovering images in a specific category."""
        self.create_test_images('portraits', 2)
        self.create_test_images('landscapes', 3)
        
        discovered = self.dataset_manager.discover_images(category='portraits')
        
        assert 'portraits' in discovered
        assert 'landscapes' not in discovered
        assert len(discovered['portraits']) == 2
    
    def test_discover_images_empty_category(self):
        """Test discovering images in empty categories."""
        # Create directory structure but no images
        self.dataset_manager.setup_directory_structure()
        
        discovered = self.dataset_manager.discover_images()
        
        for category in ['portraits', 'landscapes', 'textures']:
            assert category in discovered
            assert len(discovered[category]) == 0
    
    def test_load_datasets_complete_workflow(self):
        """Test loading complete datasets with both formats."""
        # Create test images
        self.create_test_images('portraits', 2)
        self.create_test_images('landscapes', 1)
        
        datasets = self.dataset_manager.load_datasets()
        
        # Check structure
        assert 'portraits' in datasets
        assert 'landscapes' in datasets
        assert 'textures' in datasets
        
        # Check portraits
        assert 'grayscale' in datasets['portraits']
        assert 'rgb' in datasets['portraits']
        assert len(datasets['portraits']['grayscale']) == 2
        assert len(datasets['portraits']['rgb']) == 2
        
        # Check image properties
        for img in datasets['portraits']['grayscale']:
            assert isinstance(img, np.ndarray)
            assert img.shape == (256, 256)  # Grayscale
            assert 0 <= img.min() <= img.max() <= 1
        
        for img in datasets['portraits']['rgb']:
            assert isinstance(img, np.ndarray)
            assert img.shape == (256, 256, 3)  # RGB
            assert 0 <= img.min() <= img.max() <= 1
    
    def test_generate_processed_versions(self):
        """Test generating processed grayscale and RGB versions."""
        # Create test images
        image_paths = self.create_test_images('portraits', 2)
        
        # Generate processed versions
        self.dataset_manager.generate_processed_versions()
        
        # Check that processed versions were created
        portraits_path = self.temp_dir / 'portraits'
        grayscale_dir = portraits_path / 'grayscale'
        rgb_dir = portraits_path / 'rgb'
        
        for img_path in image_paths:
            base_name = img_path.stem
            grayscale_file = grayscale_dir / f"{base_name}_gray.png"
            rgb_file = rgb_dir / f"{base_name}_rgb.png"
            
            assert grayscale_file.exists()
            assert rgb_file.exists()
            
            # Verify we can load the processed images
            grayscale_img = Image.open(grayscale_file)
            rgb_img = Image.open(rgb_file)
            
            assert grayscale_img.mode == 'L'
            assert rgb_img.mode == 'RGB'
            assert grayscale_img.size == (256, 256)
            assert rgb_img.size == (256, 256)
    
    def test_generate_processed_versions_no_overwrite(self):
        """Test that existing processed versions are not overwritten by default."""
        # Create test image
        image_paths = self.create_test_images('portraits', 1)
        
        # Generate processed versions first time
        self.dataset_manager.generate_processed_versions()
        
        # Get modification times
        portraits_path = self.temp_dir / 'portraits'
        grayscale_file = portraits_path / 'grayscale' / f"{image_paths[0].stem}_gray.png"
        rgb_file = portraits_path / 'rgb' / f"{image_paths[0].stem}_rgb.png"
        
        original_gray_mtime = grayscale_file.stat().st_mtime
        original_rgb_mtime = rgb_file.stat().st_mtime
        
        # Generate again without overwrite
        self.dataset_manager.generate_processed_versions(overwrite=False)
        
        # Check that files weren't modified
        assert grayscale_file.stat().st_mtime == original_gray_mtime
        assert rgb_file.stat().st_mtime == original_rgb_mtime
    
    def test_generate_manifest(self):
        """Test generating comprehensive data manifest."""
        # Create test images in multiple categories
        self.create_test_images('portraits', 2)
        self.create_test_images('landscapes', 1)
        
        manifest_df = self.dataset_manager.generate_manifest()
        
        # Check DataFrame structure
        assert isinstance(manifest_df, pd.DataFrame)
        assert len(manifest_df) == 3  # 2 portraits + 1 landscape
        
        expected_columns = [
            'filename', 'width', 'height', 'dataset_label', 
            'channels', 'file_size_bytes', 'original_path'
        ]
        for col in expected_columns:
            assert col in manifest_df.columns
        
        # Check data types and values
        assert manifest_df['width'].dtype in [np.int64, int]
        assert manifest_df['height'].dtype in [np.int64, int]
        assert all(manifest_df['width'] == 256)  # All resized to 256x256
        assert all(manifest_df['height'] == 256)
        
        # Check categories
        categories = set(manifest_df['dataset_label'])
        assert 'portraits' in categories
        assert 'landscapes' in categories
        
        # Check file sizes are positive
        assert all(manifest_df['file_size_bytes'] > 0)
    
    def test_generate_manifest_with_save(self):
        """Test generating and saving manifest to file."""
        # Create test images
        self.create_test_images('portraits', 1)
        
        # Generate and save manifest
        output_path = self.temp_dir / 'manifest.csv'
        manifest_df = self.dataset_manager.generate_manifest(output_path=output_path)
        
        # Check file was created
        assert output_path.exists()
        
        # Load and verify saved manifest
        loaded_df = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(manifest_df, loaded_df)
    
    def test_get_category_statistics(self):
        """Test getting statistics for each category."""
        # Create test images with different sizes
        self.create_test_images('portraits', 2)
        self.create_test_images('landscapes', 1)
        
        stats = self.dataset_manager.get_category_statistics()
        
        # Check structure
        assert 'portraits' in stats
        assert 'landscapes' in stats
        assert 'textures' in stats
        
        # Check portraits stats
        assert stats['portraits']['total_images'] == 2
        assert stats['portraits']['total_size_bytes'] > 0
        
        # Check landscapes stats
        assert stats['landscapes']['total_images'] == 1
        assert stats['landscapes']['total_size_bytes'] > 0
        
        # Check empty textures stats
        assert stats['textures']['total_images'] == 0
        assert stats['textures']['total_size_bytes'] == 0
    
    def test_validate_dataset_complete(self):
        """Test dataset validation with complete setup."""
        # Set up complete dataset
        self.dataset_manager.setup_directory_structure()
        self.create_test_images('portraits', 1)
        self.dataset_manager.generate_processed_versions()
        
        issues = self.dataset_manager.validate_dataset()
        
        # Should have no major issues
        assert len(issues['missing_directories']) == 0
        assert len(issues['corrupted_images']) == 0
        assert len(issues['missing_processed']) == 0
        
        # May have empty categories (landscapes, textures)
        assert 'landscapes' in issues['empty_categories']
        assert 'textures' in issues['empty_categories']
    
    def test_validate_dataset_missing_directories(self):
        """Test dataset validation with missing directories."""
        # Don't set up directory structure
        
        issues = self.dataset_manager.validate_dataset()
        
        # Should detect missing directories
        assert len(issues['missing_directories']) > 0
        
        # Check specific missing directories
        missing_dirs = [str(p) for p in issues['missing_directories']]
        assert any('portraits' in d for d in missing_dirs)
        assert any('landscapes' in d for d in missing_dirs)
        assert any('textures' in d for d in missing_dirs)
    
    def test_validate_dataset_missing_processed(self):
        """Test dataset validation with missing processed versions."""
        # Create original images but not processed versions
        self.create_test_images('portraits', 1)
        
        issues = self.dataset_manager.validate_dataset()
        
        # Should detect missing processed versions
        assert len(issues['missing_processed']) > 0
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end dataset management workflow."""
        # 1. Setup directory structure
        self.dataset_manager.setup_directory_structure()
        
        # 2. Add some test images
        self.create_test_images('portraits', 2)
        self.create_test_images('landscapes', 1)
        
        # 3. Generate processed versions
        self.dataset_manager.generate_processed_versions()
        
        # 4. Load datasets
        datasets = self.dataset_manager.load_datasets()
        
        # 5. Generate manifest
        manifest_path = self.temp_dir / 'data_manifest.csv'
        manifest_df = self.dataset_manager.generate_manifest(output_path=manifest_path)
        
        # 6. Get statistics
        stats = self.dataset_manager.get_category_statistics()
        
        # 7. Validate dataset
        issues = self.dataset_manager.validate_dataset()
        
        # Verify everything worked
        assert len(datasets['portraits']['grayscale']) == 2
        assert len(datasets['landscapes']['rgb']) == 1
        assert len(manifest_df) == 3
        assert manifest_path.exists()
        assert stats['portraits']['total_images'] == 2
        assert len(issues['corrupted_images']) == 0
        assert len(issues['missing_processed']) == 0
    
    def test_custom_target_size(self):
        """Test DatasetManager with custom target size."""
        # Create manager with custom size
        custom_manager = DatasetManager(data_root=self.temp_dir, target_size=(128, 128))
        
        # Create test image
        self.create_test_images('portraits', 1)
        
        # Load datasets
        datasets = custom_manager.load_datasets()
        
        # Check that images have custom size
        assert datasets['portraits']['grayscale'][0].shape == (128, 128)
        assert datasets['portraits']['rgb'][0].shape == (128, 128, 3)


if __name__ == "__main__":
    pytest.main([__file__])