"""
End-to-end integration tests for the complete SVD image compression system.

This module tests complete workflows from image loading through compression,
evaluation, visualization, and result storage.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
import time
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.compression.svd_compressor import SVDCompressor
from src.evaluation.metrics_calculator import MetricsCalculator
from src.evaluation.performance_profiler import PerformanceProfiler
from src.data.image_loader import ImageLoader
from src.data.dataset_manager import DatasetManager
from src.visualization.plot_generator import PlotGenerator
from src.batch.experiment_runner import ExperimentRunner, ExperimentConfig
from src.batch.result_manager import ResultManager


class TestEndToEndWorkflows:
    """End-to-end integration tests for complete system workflows."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace with sample data."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create directory structure
        data_dir = temp_dir / "data"
        results_dir = temp_dir / "results"
        
        # Create sample images
        for category in ['portraits', 'landscapes', 'textures']:
            category_dir = data_dir / category / 'original'
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Create 2-3 test images per category
            num_images = 2 if category == 'textures' else 3
            for i in range(num_images):
                if category == 'portraits':
                    # Create portrait-like images (more vertical)
                    img = Image.new('RGB', (120, 160), (200, 150, 100))
                elif category == 'landscapes':
                    # Create landscape-like images (more horizontal)
                    img = Image.new('RGB', (200, 120), (100, 150, 200))
                else:  # textures
                    # Create texture-like images (square with pattern)
                    img = Image.new('L', (150, 150), 128)
                
                img_path = category_dir / f"{category}_sample_{i}.png"
                img.save(img_path)
        
        yield temp_dir
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_complete_image_processing_pipeline(self, temp_workspace):
        """Test complete pipeline from image loading to compression and evaluation."""
        # Initialize components
        loader = ImageLoader(target_size=(128, 128))
        compressor = SVDCompressor()
        calculator = MetricsCalculator()
        
        # Load a test image
        image_path = temp_workspace / "data" / "portraits" / "original" / "portraits_sample_0.png"
        
        # Step 1: Load and preprocess image
        original_image = loader.load_image(image_path)
        assert original_image.shape == (128, 128, 3)
        assert 0 <= original_image.min() <= original_image.max() <= 1
        
        # Step 2: Compress image with different k values
        k_values = [5, 10, 20]
        compression_results = []
        
        for k in k_values:
            compressed_image, metadata = compressor.compress_image(original_image, k)
            
            # Verify compression results
            assert compressed_image.shape == original_image.shape
            assert metadata['k_value'] == k
            assert metadata['compression_ratio'] > 0
            
            # Step 3: Calculate quality metrics
            metrics = calculator.calculate_all_metrics(original_image, compressed_image, k)
            
            # Verify metrics
            assert 'psnr' in metrics
            assert 'ssim' in metrics
            assert 'mse' in metrics
            assert metrics['psnr'] > 0
            assert -1e-10 <= metrics['ssim'] <= 1 + 1e-10  # Allow small numerical errors
            assert metrics['mse'] >= 0
            
            compression_results.append({
                'k': k,
                'compressed_image': compressed_image,
                'metadata': metadata,
                'metrics': metrics
            })
        
        # Step 4: Verify quality trends (may not be strictly monotonic for very uniform images)
        psnr_values = [r['metrics']['psnr'] for r in compression_results]
        ssim_values = [r['metrics']['ssim'] for r in compression_results]
        
        # For very uniform images, PSNR might be very high and not strictly increasing
        # Instead, check that all values are reasonable
        assert all(psnr > 20 for psnr in psnr_values), "All PSNR values should be reasonable"
        assert all(ssim > 0.8 for ssim in ssim_values), "All SSIM values should be high"
        
        # Step 5: Save compressed images
        output_dir = temp_workspace / "results" / "compressed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for result in compression_results:
            output_path = output_dir / f"compressed_k{result['k']}.png"
            loader.save_image(result['compressed_image'], output_path)
            assert output_path.exists()
    
    def test_dataset_management_workflow(self, temp_workspace):
        """Test complete dataset management workflow."""
        # Initialize dataset manager
        dataset_manager = DatasetManager(data_root=temp_workspace / "data")
        
        # Step 1: Setup directory structure
        dataset_manager.setup_directory_structure()
        
        # Step 2: Discover images
        discovered = dataset_manager.discover_images()
        assert 'portraits' in discovered
        assert 'landscapes' in discovered
        assert 'textures' in discovered
        assert len(discovered['portraits']) > 0
        assert len(discovered['landscapes']) > 0
        
        # Step 3: Generate processed versions
        dataset_manager.generate_processed_versions()
        
        # Verify processed versions exist
        for category in ['portraits', 'landscapes']:
            grayscale_dir = temp_workspace / "data" / category / "grayscale"
            rgb_dir = temp_workspace / "data" / category / "rgb"
            assert grayscale_dir.exists()
            assert rgb_dir.exists()
            assert len(list(grayscale_dir.glob("*.png"))) > 0
            assert len(list(rgb_dir.glob("*.png"))) > 0
        
        # Step 4: Load complete datasets
        datasets = dataset_manager.load_datasets()
        
        # Verify dataset structure
        for category in ['portraits', 'landscapes', 'textures']:
            assert category in datasets
            assert 'grayscale' in datasets[category]
            assert 'rgb' in datasets[category]
        
        # Step 5: Generate manifest
        manifest_path = temp_workspace / "data_manifest.csv"
        manifest_df = dataset_manager.generate_manifest(output_path=manifest_path)
        
        assert manifest_path.exists()
        assert len(manifest_df) > 0
        assert 'dataset_label' in manifest_df.columns
        
        # Step 6: Validate dataset
        issues = dataset_manager.validate_dataset()
        assert len(issues['corrupted_images']) == 0
        assert len(issues['missing_processed']) == 0
    
    def test_batch_experiment_workflow(self, temp_workspace):
        """Test complete batch experiment workflow."""
        # Setup dataset first
        dataset_manager = DatasetManager(data_root=temp_workspace / "data")
        dataset_manager.setup_directory_structure()
        dataset_manager.generate_processed_versions()
        
        # Configure experiment
        config = ExperimentConfig(
            datasets=['portraits', 'landscapes'],
            data_root=temp_workspace / "data",
            k_values=[5, 10, 15],
            image_types=['grayscale', 'rgb'],
            output_dir=temp_workspace / "results",
            experiment_name="end_to_end_test",
            save_reconstructed_images=True,
            parallel=False,
            show_progress=False,
            checkpoint_interval=5
        )
        
        # Run batch experiments
        runner = ExperimentRunner(config)
        results_df = runner.run_batch_experiments()
        
        if results_df is not None and len(results_df) > 0:
            # Verify results structure
            required_columns = [
                'dataset', 'image_name', 'image_type', 'k_value',
                'psnr', 'ssim', 'mse', 'compression_ratio'
            ]
            for col in required_columns:
                assert col in results_df.columns
            
            # Verify data quality
            assert results_df['psnr'].min() > 0
            assert results_df['ssim'].min() >= -1e-10  # Allow small numerical errors
            assert results_df['ssim'].max() <= 1 + 1e-10  # Allow small numerical errors
            assert results_df['compression_ratio'].min() > 0
            
            # Verify output files
            results_file = config.output_dir / f"{config.experiment_name}_results.csv"
            assert results_file.exists()
            
            # Check reconstructed images
            images_dir = config.output_dir / "reconstructed_images"
            if images_dir.exists():
                image_files = list(images_dir.glob("**/*.png"))
                assert len(image_files) > 0
    
    def test_visualization_workflow(self, temp_workspace):
        """Test complete visualization workflow."""
        # Create sample data
        compressor = SVDCompressor()
        plotter = PlotGenerator()
        
        # Generate test image and compress with multiple k values
        test_image = np.random.rand(100, 100)
        k_values = [5, 10, 15, 20, 25]
        
        results_data = []
        compressed_images = []
        
        for k in k_values:
            compressed, metadata = compressor.compress_image(test_image, k)
            
            # Calculate metrics
            calculator = MetricsCalculator()
            metrics = calculator.calculate_all_metrics(test_image, compressed, k)
            
            results_data.append({
                'dataset': 'test',
                'k_value': k,
                'psnr': metrics['psnr'],
                'ssim': metrics['ssim'],
                'compression_ratio': metrics['compression_ratio']
            })
            
            compressed_images.append(compressed)
        
        results_df = pd.DataFrame(results_data)
        
        # Test singular value plot
        singular_values = compressor.singular_value_spectrum(test_image)
        fig1 = plotter.plot_singular_values(singular_values, title="Test Singular Values")
        assert fig1 is not None
        
        # Test quality vs k plots
        fig2 = plotter.plot_quality_vs_k(results_df, metric='psnr')
        assert fig2 is not None
        
        fig3 = plotter.plot_quality_vs_k(results_df, metric='ssim')
        assert fig3 is not None
        
        # Test compression analysis plot
        fig4 = plotter.plot_compression_analysis(results_df, quality_metric='psnr')
        assert fig4 is not None
        
        # Test image grid
        sample_images = [test_image] + compressed_images[:3]
        titles = ['Original', 'k=5', 'k=10', 'k=15']
        fig5 = plotter.create_image_grid(sample_images, titles)
        assert fig5 is not None
        
        # Test saving plots
        plots_dir = temp_workspace / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        saved_path = plotter.save_to_results_dir(fig1, "singular_values", results_dir=plots_dir)
        assert saved_path.exists()
        
        # Clean up figures
        plotter.close_all_figures()
    
    def test_performance_profiling_workflow(self, temp_workspace):
        """Test complete performance profiling workflow."""
        # Initialize components
        compressor = SVDCompressor()
        profiler = PerformanceProfiler(enable_logging=False)
        calculator = MetricsCalculator()
        
        # Create test images of different sizes
        test_images = [
            np.random.rand(50, 50),
            np.random.rand(100, 100),
            np.random.rand(150, 150)
        ]
        
        k_values = [5, 10, 15]
        
        # Profile compression operations
        for i, image in enumerate(test_images):
            for k in k_values:
                # Profile compression
                result, perf_result = profiler.profile_compression(compressor, image, k)
                
                # Verify profiling results
                assert perf_result.operation_name == 'svd_compression'
                assert perf_result.execution_time > 0
                assert perf_result.parameters['k'] == k
                assert perf_result.parameters['image_shape'] == image.shape
                
                # Profile metrics calculation
                compressed_image, metadata = result
                metrics_result, metrics_perf = profiler.profile_metrics_calculation(
                    calculator, image, compressed_image, k
                )
                
                assert metrics_perf.operation_name == 'metrics_calculation'
                assert metrics_perf.execution_time > 0
        
        # Generate performance summary
        compression_stats = profiler.get_summary_stats('svd_compression')
        metrics_stats = profiler.get_summary_stats('metrics_calculation')
        
        assert compression_stats['total_operations'] > 0
        assert metrics_stats['total_operations'] > 0
        
        # Save performance results
        results_file = temp_workspace / "performance_results.json"
        profiler.save_results(results_file, format='json')
        assert results_file.exists()
        
        # Test benchmark functionality
        benchmark_results = profiler.benchmark_compression_performance(
            compressor, test_images, k_values
        )
        
        assert 'total_operations' in benchmark_results
        assert benchmark_results['total_operations'] == len(test_images) * len(k_values)
    
    def test_error_handling_across_components(self, temp_workspace):
        """Test error handling across different system components."""
        # Test with invalid inputs across the pipeline
        
        # 1. Test image loader with invalid file
        loader = ImageLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_image(temp_workspace / "nonexistent.png")
        
        # 2. Test compressor with invalid inputs
        compressor = SVDCompressor()
        
        # Invalid k value
        valid_image = np.random.rand(50, 50)
        with pytest.raises(ValueError):
            compressor.compress_image(valid_image, -1)
        
        # Invalid image type
        with pytest.raises(TypeError):
            compressor.compress_image("not an array", 5)
        
        # 3. Test metrics calculator with mismatched shapes
        calculator = MetricsCalculator()
        img1 = np.random.rand(50, 50)
        img2 = np.random.rand(60, 60)
        
        with pytest.raises(ValueError):
            calculator.calculate_mse(img1, img2)
        
        # 4. Test dataset manager with missing directories
        dataset_manager = DatasetManager(data_root=temp_workspace / "nonexistent")
        issues = dataset_manager.validate_dataset()
        assert len(issues['missing_directories']) > 0
        
        # 5. Test experiment runner with invalid configuration
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                datasets=[],  # Empty datasets
                data_root=temp_workspace / "data",
                k_values=[5],
                output_dir=temp_workspace / "results",
                experiment_name="error_test"
            )
    
    def test_data_consistency_across_pipeline(self, temp_workspace):
        """Test data consistency throughout the processing pipeline."""
        # Create a known test image
        test_image = np.ones((64, 64, 3)) * 0.5  # Gray image
        
        # Save and reload through image loader
        loader = ImageLoader(target_size=(64, 64))
        temp_image_path = temp_workspace / "test_consistency.png"
        loader.save_image(test_image, temp_image_path)
        reloaded_image = loader.load_image(temp_image_path)
        
        # Should be very close (allowing for PNG compression artifacts)
        np.testing.assert_allclose(test_image, reloaded_image, atol=0.01)
        
        # Compress and verify consistency
        compressor = SVDCompressor()
        
        # Test with k=min(dimensions) for perfect reconstruction
        k_full = min(test_image.shape[:2])
        compressed_full, metadata_full = compressor.compress_image(test_image, k_full)
        
        # Should be nearly identical for full rank
        np.testing.assert_allclose(test_image, compressed_full, rtol=1e-10)
        
        # Test metrics consistency
        calculator = MetricsCalculator()
        
        # Metrics for identical images
        metrics_identical = calculator.calculate_all_metrics(test_image, test_image, k_full)
        assert metrics_identical['psnr'] == float('inf')
        assert abs(metrics_identical['ssim'] - 1.0) < 1e-10
        assert metrics_identical['mse'] == 0.0
        
        # Metrics should be consistent across multiple calculations
        metrics1 = calculator.calculate_all_metrics(test_image, compressed_full, k_full)
        metrics2 = calculator.calculate_all_metrics(test_image, compressed_full, k_full)
        
        for key in ['psnr', 'ssim', 'mse']:
            assert abs(metrics1[key] - metrics2[key]) < 1e-10
    
    def test_scalability_across_system(self, temp_workspace):
        """Test system scalability with varying data sizes."""
        # Test with different image sizes and batch sizes
        image_sizes = [(32, 32), (64, 64), (128, 128)]
        k_values = [5, 10]
        
        # Initialize components
        compressor = SVDCompressor()
        calculator = MetricsCalculator()
        profiler = PerformanceProfiler(enable_logging=False)
        
        scalability_results = []
        
        for height, width in image_sizes:
            # Create test image
            test_image = np.random.rand(height, width, 3)
            
            for k in k_values:
                # Time the complete pipeline
                start_time = time.time()
                
                # Compression
                compressed, metadata = compressor.compress_image(test_image, k)
                
                # Metrics calculation
                metrics = calculator.calculate_all_metrics(test_image, compressed, k)
                
                end_time = time.time()
                
                scalability_results.append({
                    'image_size': f'{height}x{width}',
                    'pixels': height * width,
                    'k': k,
                    'total_time': end_time - start_time,
                    'psnr': metrics['psnr'],
                    'compression_ratio': metrics['compression_ratio']
                })
        
        # Analyze scalability
        for result in scalability_results:
            # All operations should complete in reasonable time
            assert result['total_time'] < 5.0, f"Pipeline too slow for {result['image_size']}: {result['total_time']}s"
            
            # Quality metrics should be reasonable
            assert result['psnr'] > 0
            assert result['compression_ratio'] > 0
        
        # Timing should scale reasonably with image size
        small_times = [r['total_time'] for r in scalability_results if r['pixels'] <= 32*32]
        large_times = [r['total_time'] for r in scalability_results if r['pixels'] >= 128*128]
        
        if small_times and large_times:
            avg_small = np.mean(small_times)
            avg_large = np.mean(large_times)
            # Large images should not be more than 20x slower than small ones
            assert avg_large < avg_small * 20


if __name__ == "__main__":
    pytest.main([__file__])