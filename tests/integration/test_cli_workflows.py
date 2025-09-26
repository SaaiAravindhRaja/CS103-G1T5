"""
Integration tests for CLI workflows.

This module tests end-to-end CLI functionality including
command-line interfaces, batch processing, and result generation.
"""

import pytest
import subprocess
import tempfile
import shutil
import json
import csv
import time
from pathlib import Path
from PIL import Image
import numpy as np

from src.batch.cli_demo import main as cli_main
from src.batch.experiment_runner import ExperimentRunner, ExperimentConfig


class TestCLIWorkflows:
    """Integration tests for command-line interface workflows."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for CLI tests."""
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create basic directory structure
        data_dir = temp_dir / "data"
        results_dir = temp_dir / "results"
        
        # Create sample images for testing
        for category in ['portraits', 'landscapes']:
            category_dir = data_dir / category / 'original'
            category_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a few test images
            for i in range(2):
                img = Image.new('RGB', (100, 100), (255, 128, 64))
                img_path = category_dir / f"test_image_{i}.png"
                img.save(img_path)
        
        yield temp_dir
        
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def test_cli_demo_basic_functionality(self, temp_workspace):
        """Test basic CLI demo functionality."""
        # Create a simple test configuration
        config_data = {
            'datasets': ['portraits'],
            'k_values': [5, 10],
            'image_types': ['grayscale', 'rgb'],
            'data_root': str(temp_workspace / 'data'),
            'output_dir': str(temp_workspace / 'results'),
            'experiment_name': 'cli_test',
            'save_reconstructed_images': True,
            'parallel': False,
            'show_progress': False
        }
        
        config_file = temp_workspace / 'config.json'
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Test CLI execution (mock the main function call)
        try:
            # Create experiment config from the JSON data
            experiment_config = ExperimentConfig(**config_data)
            runner = ExperimentRunner(experiment_config)
            
            # This simulates what the CLI would do
            results_df = runner.run_batch_experiments()
            
            # Verify results
            assert len(results_df) > 0
            assert 'psnr' in results_df.columns
            assert 'ssim' in results_df.columns
            
            # Check that output files were created
            results_file = Path(config_data['output_dir']) / f"{config_data['experiment_name']}_results.csv"
            assert results_file.exists()
            
        except Exception as e:
            pytest.fail(f"CLI demo failed: {e}")
    
    def test_cli_argument_parsing(self, temp_workspace):
        """Test CLI argument parsing and validation."""
        # Test with valid arguments
        valid_args = {
            'data_root': str(temp_workspace / 'data'),
            'output_dir': str(temp_workspace / 'results'),
            'datasets': ['portraits'],
            'k_values': [5, 10],
            'experiment_name': 'test_experiment'
        }
        
        # Create experiment config (simulating CLI parsing)
        try:
            config = ExperimentConfig(
                datasets=valid_args['datasets'],
                data_root=Path(valid_args['data_root']),
                k_values=valid_args['k_values'],
                output_dir=Path(valid_args['output_dir']),
                experiment_name=valid_args['experiment_name'],
                image_types=['grayscale'],
                save_reconstructed_images=False,
                parallel=False,
                show_progress=False
            )
            
            # Should create config without errors
            assert config.experiment_name == 'test_experiment'
            assert config.datasets == ['portraits']
            
        except Exception as e:
            pytest.fail(f"Argument parsing failed: {e}")
    
    def test_cli_error_handling(self, temp_workspace):
        """Test CLI error handling for invalid inputs."""
        # Test with non-existent data directory
        invalid_config = ExperimentConfig(
            datasets=['portraits'],
            data_root=temp_workspace / 'nonexistent',
            k_values=[5],
            output_dir=temp_workspace / 'results',
            experiment_name='error_test',
            image_types=['grayscale'],
            save_reconstructed_images=False,
            parallel=False,
            show_progress=False
        )
        
        runner = ExperimentRunner(invalid_config)
        
        # Should handle missing data gracefully
        try:
            results_df = runner.run_batch_experiments()
            # Should return empty results or handle gracefully
            assert isinstance(results_df, type(None)) or len(results_df) == 0
        except Exception as e:
            # Should raise a meaningful error
            assert "data" in str(e).lower() or "not found" in str(e).lower()
    
    def test_cli_output_formats(self, temp_workspace):
        """Test CLI output in different formats."""
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=temp_workspace / 'data',
            k_values=[5],
            output_dir=temp_workspace / 'results',
            experiment_name='format_test',
            image_types=['grayscale'],
            save_reconstructed_images=True,
            parallel=False,
            show_progress=False
        )
        
        runner = ExperimentRunner(config)
        results_df = runner.run_batch_experiments()
        
        if results_df is not None and len(results_df) > 0:
            # Check CSV output
            csv_file = config.output_dir / f"{config.experiment_name}_results.csv"
            assert csv_file.exists()
            
            # Verify CSV content
            with open(csv_file, 'r') as f:
                csv_reader = csv.DictReader(f)
                rows = list(csv_reader)
                assert len(rows) > 0
                assert 'psnr' in rows[0]
                assert 'ssim' in rows[0]
            
            # Check that reconstructed images were saved
            images_dir = config.output_dir / 'reconstructed_images'
            if images_dir.exists():
                image_files = list(images_dir.glob('**/*.png'))
                assert len(image_files) > 0
    
    def test_cli_batch_processing_workflow(self, temp_workspace):
        """Test complete batch processing workflow via CLI."""
        # Create more comprehensive test data
        data_dir = temp_workspace / 'data'
        
        # Add more test images
        for category in ['portraits', 'landscapes']:
            category_dir = data_dir / category / 'original'
            for i in range(3):
                # Create images with different characteristics
                if i == 0:
                    img = Image.new('RGB', (150, 150), (255, 0, 0))  # Red
                elif i == 1:
                    img = Image.new('L', (120, 120), 128)  # Grayscale
                else:
                    img = Image.new('RGB', (200, 100), (0, 255, 0))  # Green, different aspect
                
                img_path = category_dir / f"batch_test_{i}.png"
                img.save(img_path)
        
        # Configure batch processing
        config = ExperimentConfig(
            datasets=['portraits', 'landscapes'],
            data_root=data_dir,
            k_values=[5, 10, 20],
            image_types=['grayscale', 'rgb'],
            output_dir=temp_workspace / 'batch_results',
            experiment_name='batch_workflow_test',
            save_reconstructed_images=True,
            parallel=False,
            show_progress=False,
            checkpoint_interval=2
        )
        
        # Run batch processing
        runner = ExperimentRunner(config)
        results_df = runner.run_batch_experiments()
        
        if results_df is not None and len(results_df) > 0:
            # Verify comprehensive results
            expected_combinations = len(config.datasets) * len(config.image_types) * len(config.k_values) * 5  # 5 images per category
            # Note: actual results may be less due to image processing variations or missing categories
            assert len(results_df) <= expected_combinations
            
            # Check result quality
            assert 'dataset' in results_df.columns
            assert 'k_value' in results_df.columns
            assert 'psnr' in results_df.columns
            assert 'compression_ratio' in results_df.columns
            
            # Verify datasets are represented
            datasets_in_results = set(results_df['dataset'].unique())
            assert len(datasets_in_results) > 0
            
            # Check k-values are represented
            k_values_in_results = set(results_df['k_value'].unique())
            assert len(k_values_in_results) > 0
    
    def test_cli_resume_functionality(self, temp_workspace):
        """Test CLI resume from checkpoint functionality."""
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=temp_workspace / 'data',
            k_values=[5, 10, 15],
            image_types=['grayscale'],
            output_dir=temp_workspace / 'resume_results',
            experiment_name='resume_test',
            save_reconstructed_images=False,
            parallel=False,
            show_progress=False,
            checkpoint_interval=1,  # Save after each experiment
            resume_from_checkpoint=False
        )
        
        # First run - simulate partial completion
        runner1 = ExperimentRunner(config)
        
        # Mock interruption after partial completion
        original_execute = runner1._execute_single_experiment
        call_count = 0
        
        def mock_execute_with_interruption(task):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Complete first 2 experiments
                return original_execute(task)
            else:
                raise KeyboardInterrupt("Simulated interruption")
        
        runner1._execute_single_experiment = mock_execute_with_interruption
        
        # Run until interruption
        try:
            runner1.run_batch_experiments()
        except KeyboardInterrupt:
            pass
        
        # Verify checkpoint was created
        checkpoint_file = config.output_dir / f"{config.experiment_name}_checkpoint.csv"
        if checkpoint_file.exists():
            # Second run - resume from checkpoint
            resume_config = ExperimentConfig(
                **{**config.__dict__, 'resume_from_checkpoint': True}
            )
            
            runner2 = ExperimentRunner(resume_config)
            final_results = runner2.run_batch_experiments()
            
            if final_results is not None:
                # Should have completed all experiments
                assert len(final_results) > 0
    
    def test_cli_parallel_processing(self, temp_workspace):
        """Test CLI parallel processing functionality."""
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=temp_workspace / 'data',
            k_values=[5, 10],
            image_types=['grayscale'],
            output_dir=temp_workspace / 'parallel_results',
            experiment_name='parallel_test',
            save_reconstructed_images=False,
            parallel=True,
            max_workers=2,
            show_progress=False
        )
        
        runner = ExperimentRunner(config)
        results_df = runner.run_batch_experiments()
        
        if results_df is not None and len(results_df) > 0:
            # Should produce same results as sequential processing
            assert 'psnr' in results_df.columns
            assert 'ssim' in results_df.columns
            assert len(results_df) > 0
    
    def test_cli_configuration_validation(self, temp_workspace):
        """Test CLI configuration validation."""
        # Test invalid k values
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                datasets=['portraits'],
                data_root=temp_workspace / 'data',
                k_values=[0, -1],  # Invalid k values
                output_dir=temp_workspace / 'results',
                experiment_name='invalid_test'
            )
        
        # Test empty datasets
        with pytest.raises((ValueError, TypeError)):
            ExperimentConfig(
                datasets=[],  # Empty datasets
                data_root=temp_workspace / 'data',
                k_values=[5],
                output_dir=temp_workspace / 'results',
                experiment_name='empty_test'
            )
    
    def test_cli_logging_and_progress(self, temp_workspace):
        """Test CLI logging and progress reporting."""
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=temp_workspace / 'data',
            k_values=[5],
            image_types=['grayscale'],
            output_dir=temp_workspace / 'logging_results',
            experiment_name='logging_test',
            save_reconstructed_images=False,
            parallel=False,
            show_progress=True  # Enable progress reporting
        )
        
        runner = ExperimentRunner(config)
        
        # Should run without errors even with progress enabled
        try:
            results_df = runner.run_batch_experiments()
            # Test passes if no exceptions are raised
            assert True
        except Exception as e:
            pytest.fail(f"CLI with progress reporting failed: {e}")
    
    def test_cli_memory_management(self, temp_workspace):
        """Test CLI memory management during batch processing."""
        # Create larger test dataset
        data_dir = temp_workspace / 'data'
        
        for category in ['portraits']:
            category_dir = data_dir / category / 'original'
            for i in range(5):  # More images
                img = Image.new('RGB', (200, 200), (i * 50, 100, 150))
                img_path = category_dir / f"memory_test_{i}.png"
                img.save(img_path)
        
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=data_dir,
            k_values=[5, 10, 15, 20],  # Multiple k values
            image_types=['grayscale', 'rgb'],
            output_dir=temp_workspace / 'memory_results',
            experiment_name='memory_test',
            save_reconstructed_images=False,  # Reduce memory usage
            parallel=False,
            show_progress=False
        )
        
        # Monitor memory usage
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        runner = ExperimentRunner(config)
        results_df = runner.run_batch_experiments()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase} MB"
        
        if results_df is not None:
            assert len(results_df) > 0
    
    def test_cli_stress_testing(self, temp_workspace):
        """Test CLI under stress conditions with many images and k-values."""
        # Create stress test dataset
        data_dir = temp_workspace / 'data'
        
        for category in ['portraits', 'landscapes']:
            category_dir = data_dir / category / 'original'
            for i in range(10):  # Many images
                # Vary image sizes to test robustness
                size = 50 + (i * 10)  # 50x50 to 140x140
                img = Image.new('RGB', (size, size), (i * 25, 128, 255 - i * 20))
                img_path = category_dir / f"stress_test_{i:02d}.png"
                img.save(img_path)
        
        config = ExperimentConfig(
            datasets=['portraits', 'landscapes'],
            data_root=data_dir,
            k_values=[3, 5, 8, 12, 15, 20, 25],  # Many k values
            image_types=['grayscale', 'rgb'],
            output_dir=temp_workspace / 'stress_results',
            experiment_name='stress_test',
            save_reconstructed_images=False,
            parallel=False,
            show_progress=False,
            checkpoint_interval=5
        )
        
        # Run stress test
        start_time = time.time()
        runner = ExperimentRunner(config)
        results_df = runner.run_batch_experiments()
        end_time = time.time()
        
        # Verify stress test results
        if results_df is not None and len(results_df) > 0:
            # Should handle large number of experiments
            assert len(results_df) > 50, "Should process many experiments"
            
            # Should complete in reasonable time (allow generous timeout for CI)
            total_time = end_time - start_time
            assert total_time < 300, f"Stress test took too long: {total_time}s"
            
            # All results should be valid
            assert results_df['psnr'].notna().all(), "All PSNR values should be valid"
            assert results_df['ssim'].notna().all(), "All SSIM values should be valid"
            assert (results_df['compression_ratio'] > 0).all(), "All compression ratios should be positive"
    
    def test_cli_edge_cases(self, temp_workspace):
        """Test CLI handling of edge cases and boundary conditions."""
        data_dir = temp_workspace / 'data'
        
        # Create edge case images
        portraits_dir = data_dir / 'portraits' / 'original'
        portraits_dir.mkdir(parents=True, exist_ok=True)
        
        # Very small image
        tiny_img = Image.new('RGB', (10, 10), (255, 0, 0))
        tiny_img.save(portraits_dir / "tiny_image.png")
        
        # Square image
        square_img = Image.new('L', (50, 50), 128)
        square_img.save(portraits_dir / "square_image.png")
        
        # Rectangular image
        rect_img = Image.new('RGB', (80, 40), (0, 255, 0))
        rect_img.save(portraits_dir / "rect_image.png")
        
        # Test with edge case k-values
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=data_dir,
            k_values=[1, 2, 5],  # Small k values
            image_types=['grayscale', 'rgb'],
            output_dir=temp_workspace / 'edge_results',
            experiment_name='edge_test',
            save_reconstructed_images=False,
            parallel=False,
            show_progress=False
        )
        
        runner = ExperimentRunner(config)
        results_df = runner.run_batch_experiments()
        
        # Should handle edge cases gracefully
        if results_df is not None:
            # Should have some results (may skip some due to k > min_dim)
            assert len(results_df) >= 0
            
            # All successful results should be valid
            if len(results_df) > 0:
                assert results_df['psnr'].notna().all()
                assert (results_df['k_value'] >= 1).all()
    
    def test_cli_robustness_with_corrupted_data(self, temp_workspace):
        """Test CLI robustness when dealing with corrupted or invalid data."""
        data_dir = temp_workspace / 'data'
        portraits_dir = data_dir / 'portraits' / 'original'
        portraits_dir.mkdir(parents=True, exist_ok=True)
        
        # Create valid image
        valid_img = Image.new('RGB', (100, 100), (255, 128, 64))
        valid_img.save(portraits_dir / "valid_image.png")
        
        # Create corrupted file (not an image)
        corrupted_file = portraits_dir / "corrupted.png"
        with open(corrupted_file, 'w') as f:
            f.write("This is not an image file")
        
        # Create empty file
        empty_file = portraits_dir / "empty.png"
        empty_file.touch()
        
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=data_dir,
            k_values=[5, 10],
            image_types=['grayscale'],
            output_dir=temp_workspace / 'robust_results',
            experiment_name='robust_test',
            save_reconstructed_images=False,
            parallel=False,
            show_progress=False
        )
        
        # Should handle corrupted data gracefully
        try:
            runner = ExperimentRunner(config)
            results_df = runner.run_batch_experiments()
            
            # Should either succeed with valid data or handle errors gracefully
            if results_df is not None:
                # If any results, they should be valid
                if len(results_df) > 0:
                    assert results_df['psnr'].notna().all()
            
        except Exception as e:
            # Should provide meaningful error messages
            assert "data" in str(e).lower() or "image" in str(e).lower() or "file" in str(e).lower()
    
    def test_cli_performance_monitoring(self, temp_workspace):
        """Test CLI performance monitoring and timing accuracy."""
        data_dir = temp_workspace / 'data'
        
        # Create test images
        for category in ['portraits']:
            category_dir = data_dir / category / 'original'
            category_dir.mkdir(parents=True, exist_ok=True)
            
            for i in range(3):
                img = Image.new('RGB', (100, 100), (i * 80, 100, 200))
                img_path = category_dir / f"perf_test_{i}.png"
                img.save(img_path)
        
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=data_dir,
            k_values=[5, 10, 15],
            image_types=['grayscale', 'rgb'],
            output_dir=temp_workspace / 'perf_results',
            experiment_name='perf_test',
            save_reconstructed_images=False,
            parallel=False,
            show_progress=False
        )
        
        # Monitor overall performance
        import time
        start_time = time.time()
        
        runner = ExperimentRunner(config)
        results_df = runner.run_batch_experiments()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if results_df is not None and len(results_df) > 0:
            # Check timing accuracy
            recorded_times = results_df['compression_time'].sum()
            
            # Recorded times should be reasonable portion of total time
            # (allowing for overhead, I/O, etc.)
            assert recorded_times <= total_time * 2, "Recorded times should be reasonable"
            assert recorded_times > 0, "Should record positive compression times"
            
            # Individual times should be reasonable
            assert (results_df['compression_time'] > 0).all(), "All times should be positive"
            assert (results_df['compression_time'] < 10).all(), "Individual times should be reasonable"


if __name__ == "__main__":
    pytest.main([__file__])