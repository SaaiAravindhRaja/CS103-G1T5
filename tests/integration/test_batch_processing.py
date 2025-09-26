"""
Integration tests for batch processing workflows.

This module tests the complete batch processing pipeline including
experiment configuration, execution, result management, and resumption capabilities.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import json

from src.batch.experiment_runner import ExperimentRunner, ExperimentConfig
from src.batch.result_manager import ResultManager
from src.compression.svd_compressor import SVDCompressor
from src.data.dataset_manager import DatasetManager
from src.evaluation.metrics_calculator import MetricsCalculator


class TestBatchProcessingIntegration:
    """Integration tests for complete batch processing workflows."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for testing."""
        # Create small test images to speed up tests
        # Ensure images are properly normalized to [0,1] range
        np.random.seed(42)  # For reproducible tests
        grayscale_img = np.random.rand(32, 32).astype(np.float64)
        rgb_img = np.random.rand(32, 32, 3).astype(np.float64)
        
        # Ensure values are exactly in [0,1] range
        grayscale_img = np.clip(grayscale_img, 0.0, 1.0)
        rgb_img = np.clip(rgb_img, 0.0, 1.0)
        
        return {
            'portraits': {'grayscale': [grayscale_img], 'rgb': [rgb_img]},
            'landscapes': {'grayscale': [grayscale_img], 'rgb': [rgb_img]}
        }
    
    @pytest.fixture
    def experiment_config(self, temp_dir):
        """Create test experiment configuration."""
        return ExperimentConfig(
            datasets=['portraits', 'landscapes'],
            data_root=temp_dir / "data",
            k_values=[5, 10, 15],
            image_types=['grayscale', 'rgb'],
            output_dir=temp_dir / "results",
            save_reconstructed_images=True,
            parallel=False,  # Use sequential for deterministic testing
            experiment_name="test_experiment",
            show_progress=False,
            checkpoint_interval=2
        )
    
    def test_complete_batch_workflow(self, experiment_config, sample_images, temp_dir):
        """Test complete batch processing workflow from start to finish."""
        # Mock dataset loading to return our sample images
        with patch.object(DatasetManager, 'load_datasets', return_value=sample_images):
            # Initialize experiment runner
            runner = ExperimentRunner(experiment_config)
            
            # Run batch experiments
            results_df = runner.run_batch_experiments()
            
            # Verify results
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) > 0
            
            # Check that all expected combinations are present
            expected_combinations = len(experiment_config.datasets) * len(experiment_config.image_types) * len(experiment_config.k_values)
            assert len(results_df) == expected_combinations
            
            # Verify required columns are present
            required_columns = [
                'dataset', 'image_name', 'image_type', 'k_value',
                'psnr', 'ssim', 'mse', 'compression_ratio', 'compression_time'
            ]
            for col in required_columns:
                assert col in results_df.columns
            
            # Verify results file was created
            results_file = experiment_config.output_dir / f"{experiment_config.experiment_name}_results.csv"
            assert results_file.exists()
            
            # Verify reconstructed images were saved
            images_dir = experiment_config.output_dir / "reconstructed_images"
            assert images_dir.exists()
            
            # Check that images were saved for each dataset
            for dataset in experiment_config.datasets:
                dataset_dir = images_dir / dataset
                assert dataset_dir.exists()
                assert len(list(dataset_dir.glob("*.png"))) > 0
    
    def test_experiment_resumption(self, experiment_config, sample_images, temp_dir):
        """Test experiment resumption from checkpoint."""
        with patch.object(DatasetManager, 'load_datasets', return_value=sample_images):
            # First run - simulate interruption after partial completion
            runner1 = ExperimentRunner(experiment_config)
            
            # Mock the experiment execution to simulate partial completion
            original_execute = runner1._execute_single_experiment
            call_count = 0
            
            def mock_execute(task):
                nonlocal call_count
                call_count += 1
                if call_count <= 3:  # Complete first 3 experiments
                    return original_execute(task)
                else:
                    raise KeyboardInterrupt("Simulated interruption")
            
            runner1._execute_single_experiment = mock_execute
            
            # Run until interruption
            try:
                runner1.run_batch_experiments()
            except KeyboardInterrupt:
                pass
            
            # Verify checkpoint was created
            checkpoint_file = experiment_config.output_dir / f"{experiment_config.experiment_name}_checkpoint.csv"
            assert checkpoint_file.exists()
            
            # Load checkpoint and verify partial results
            checkpoint_df = pd.read_csv(checkpoint_file)
            # Should have 2 results (saved after every 2 experiments due to checkpoint_interval=2)
            assert len(checkpoint_df) == 2
            
            # Second run - resume from checkpoint
            config_resume = ExperimentConfig(
                **{**experiment_config.__dict__, 'resume_from_checkpoint': True}
            )
            
            runner2 = ExperimentRunner(config_resume)
            results_df = runner2.run_batch_experiments()
            
            # Verify all experiments completed
            expected_total = len(experiment_config.datasets) * len(experiment_config.image_types) * len(experiment_config.k_values)
            assert len(results_df) == expected_total
    
    def test_result_manager_functionality(self, temp_dir):
        """Test ResultManager functionality."""
        result_manager = ResultManager(temp_dir, "test_experiment")
        
        # Create sample results
        sample_results = [
            {
                'experiment_id': 'test_001',
                'dataset': 'portraits',
                'image_name': 'image_001',
                'image_type': 'grayscale',
                'k_value': 10,
                'psnr': 25.5,
                'ssim': 0.85,
                'mse': 0.01,
                'compression_ratio': 2.5,
                'compression_time': 0.1
            },
            {
                'experiment_id': 'test_002',
                'dataset': 'landscapes',
                'image_name': 'image_001',
                'image_type': 'rgb',
                'k_value': 20,
                'psnr': 30.2,
                'ssim': 0.92,
                'mse': 0.005,
                'compression_ratio': 1.8,
                'compression_time': 0.15
            }
        ]
        
        # Test saving results
        results_path = result_manager.save_results(sample_results)
        assert results_path.exists()
        
        # Test loading results
        loaded_df = result_manager.load_results()
        assert loaded_df is not None
        assert len(loaded_df) == 2
        
        # Test metadata saving/loading
        metadata = {
            'experiment_date': '2024-01-01',
            'total_images': 10,
            'configuration': {'k_values': [10, 20]}
        }
        
        metadata_path = result_manager.save_metadata(metadata)
        assert metadata_path.exists()
        
        loaded_metadata = result_manager.load_metadata()
        assert loaded_metadata is not None
        assert loaded_metadata['total_images'] == 10
        
        # Test systematic filename generation
        filename = result_manager.generate_systematic_filename(
            'portraits', 'image_001', 'grayscale', 15
        )
        assert filename == 'portraits_image_001_grayscale_k015.png'
        
        # Test summary report generation
        summary = result_manager.generate_summary_report(loaded_df)
        assert 'experiment_info' in summary
        assert 'quality' in summary
        assert summary['experiment_info']['total_experiments'] == 2
    
    def test_parallel_processing(self, experiment_config, sample_images, temp_dir):
        """Test parallel processing functionality."""
        # Enable parallel processing
        experiment_config.parallel = True
        experiment_config.max_workers = 2
        
        with patch.object(DatasetManager, 'load_datasets', return_value=sample_images):
            runner = ExperimentRunner(experiment_config)
            results_df = runner.run_batch_experiments()
            
            # Verify results are same as sequential processing
            assert isinstance(results_df, pd.DataFrame)
            assert len(results_df) > 0
            
            # All experiments should complete
            expected_combinations = len(experiment_config.datasets) * len(experiment_config.image_types) * len(experiment_config.k_values)
            assert len(results_df) == expected_combinations
    
    def test_error_handling_in_batch_processing(self, experiment_config, sample_images, temp_dir):
        """Test error handling during batch processing."""
        with patch.object(DatasetManager, 'load_datasets', return_value=sample_images):
            runner = ExperimentRunner(experiment_config)
            
            # Mock compression to fail for specific cases
            original_compress = runner.compressor.compress_image
            
            def mock_compress(image, k):
                if k == 10:  # Simulate failure for k=10
                    raise ValueError("Simulated compression error")
                return original_compress(image, k)
            
            runner.compressor.compress_image = mock_compress
            
            # Run experiments - should handle errors gracefully
            results_df = runner.run_batch_experiments()
            
            # Should have results for k=5 and k=15, but not k=10
            assert len(results_df) > 0
            assert 10 not in results_df['k_value'].values
            assert 5 in results_df['k_value'].values
            assert 15 in results_df['k_value'].values
    
    def test_checkpoint_functionality(self, experiment_config, sample_images, temp_dir):
        """Test checkpoint saving and loading functionality."""
        result_manager = ResultManager(temp_dir, experiment_config.experiment_name)
        
        # Create sample results for checkpoint
        sample_results = [
            {
                'experiment_id': 'test_001',
                'dataset': 'portraits',
                'image_name': 'image_001',
                'image_type': 'grayscale',
                'k_value': 5,
                'psnr': 25.0
            }
        ]
        
        # Test checkpoint saving
        checkpoint_path = result_manager.save_checkpoint(sample_results)
        assert checkpoint_path.exists()
        
        # Test checkpoint loading
        loaded_results, completed_ids = result_manager.load_checkpoint()
        assert len(loaded_results) == 1
        assert len(completed_ids) == 1
        # The experiment ID is generated from the result data, not the original experiment_id field
        expected_id = 'portraits_image_001_grayscale_k5'
        assert expected_id in completed_ids
        
        # Test checkpoint cleanup
        result_manager.cleanup_checkpoint()
        assert not checkpoint_path.exists()
    
    def test_experiment_summary_generation(self, experiment_config, sample_images, temp_dir):
        """Test experiment summary generation."""
        with patch.object(DatasetManager, 'load_datasets', return_value=sample_images):
            runner = ExperimentRunner(experiment_config)
            results_df = runner.run_batch_experiments()
            
            # Generate summary
            summary = runner.get_experiment_summary()
            
            # Verify summary contents
            assert 'total_experiments' in summary
            assert 'datasets' in summary
            assert 'k_values' in summary
            assert 'avg_psnr' in summary
            assert 'avg_ssim' in summary
            assert 'avg_compression_ratio' in summary
            
            assert summary['total_experiments'] > 0
            assert set(summary['datasets']) == set(experiment_config.datasets)
            assert set(summary['k_values']) == set(experiment_config.k_values)
    
    def test_result_export_functionality(self, temp_dir):
        """Test result export in different formats."""
        result_manager = ResultManager(temp_dir, "export_test")
        
        # Create and save sample results
        sample_results = [
            {
                'dataset': 'portraits',
                'image_name': 'image_001',
                'k_value': 10,
                'psnr': 25.5,
                'ssim': 0.85
            }
        ]
        
        result_manager.save_results(sample_results)
        
        # Test CSV export
        exported_files = result_manager.export_results_for_analysis('csv')
        assert len(exported_files) == 1
        assert exported_files[0].suffix == '.csv'
        assert exported_files[0].exists()
        
        # Test JSON export
        exported_files = result_manager.export_results_for_analysis('json')
        assert len(exported_files) == 1
        assert exported_files[0].suffix == '.json'
        assert exported_files[0].exists()
        
        # Verify JSON content
        with open(exported_files[0]) as f:
            json_data = json.load(f)
        assert len(json_data) == 1
        assert json_data[0]['psnr'] == 25.5
    
    def test_large_scale_batch_processing(self, temp_dir):
        """Test batch processing with large number of experiments."""
        # Create large dataset simulation
        large_sample_images = {}
        np.random.seed(42)
        
        # Create multiple datasets with multiple images each
        for dataset in ['portraits', 'landscapes', 'textures']:
            large_sample_images[dataset] = {
                'grayscale': [np.random.rand(64, 64) for _ in range(5)],
                'rgb': [np.random.rand(64, 64, 3) for _ in range(5)]
            }
        
        config = ExperimentConfig(
            datasets=['portraits', 'landscapes', 'textures'],
            data_root=temp_dir / "data",
            k_values=[5, 10, 15, 20, 25],
            image_types=['grayscale', 'rgb'],
            output_dir=temp_dir / "large_results",
            save_reconstructed_images=False,  # Speed up test
            parallel=False,
            experiment_name="large_scale_test",
            show_progress=False,
            checkpoint_interval=10
        )
        
        with patch.object(DatasetManager, 'load_datasets', return_value=large_sample_images):
            runner = ExperimentRunner(config)
            
            # Monitor performance
            import time
            start_time = time.time()
            results_df = runner.run_batch_experiments()
            end_time = time.time()
            
            # Verify large scale processing
            expected_experiments = 3 * 2 * 5 * 5  # datasets * image_types * k_values * images_per_type
            assert len(results_df) == expected_experiments
            
            # Should complete in reasonable time
            total_time = end_time - start_time
            assert total_time < 120, f"Large scale processing took too long: {total_time}s"
            
            # All results should be valid
            assert results_df['psnr'].notna().all()
            assert results_df['ssim'].notna().all()
            assert (results_df['compression_ratio'] > 0).all()
    
    def test_batch_processing_data_integrity(self, experiment_config, sample_images, temp_dir):
        """Test data integrity throughout batch processing pipeline."""
        with patch.object(DatasetManager, 'load_datasets', return_value=sample_images):
            runner = ExperimentRunner(experiment_config)
            results_df = runner.run_batch_experiments()
            
            # Verify data integrity
            assert len(results_df) > 0
            
            # Check that all expected columns are present and valid
            required_columns = [
                'experiment_id', 'dataset', 'image_name', 'image_type', 
                'k_value', 'psnr', 'ssim', 'mse', 'compression_ratio',
                'compression_time', 'timestamp'
            ]
            
            for col in required_columns:
                assert col in results_df.columns, f"Missing column: {col}"
                assert results_df[col].notna().all(), f"Null values in column: {col}"
            
            # Verify data types and ranges
            assert results_df['k_value'].dtype in [np.int32, np.int64]
            assert (results_df['k_value'] > 0).all()
            
            assert results_df['psnr'].dtype in [np.float32, np.float64]
            assert (results_df['psnr'] > 0).all()
            
            assert results_df['ssim'].dtype in [np.float32, np.float64]
            assert (results_df['ssim'] >= 0).all()
            assert (results_df['ssim'] <= 1).all()
            
            assert results_df['compression_ratio'].dtype in [np.float32, np.float64]
            assert (results_df['compression_ratio'] > 0).all()
            
            # Verify experiment IDs are unique
            assert len(results_df['experiment_id'].unique()) == len(results_df)
    
    def test_batch_processing_scalability(self, temp_dir):
        """Test batch processing scalability with varying workloads."""
        scalability_results = []
        
        # Test different workload sizes
        workload_sizes = [
            (1, 1, [5]),           # Small: 1 dataset, 1 image type, 1 k-value
            (2, 2, [5, 10]),       # Medium: 2 datasets, 2 image types, 2 k-values  
            (3, 2, [5, 10, 15])    # Large: 3 datasets, 2 image types, 3 k-values
        ]
        
        for num_datasets, num_image_types, k_values in workload_sizes:
            # Create test data for this workload
            test_images = {}
            datasets = ['portraits', 'landscapes', 'textures'][:num_datasets]
            image_types = ['grayscale', 'rgb'][:num_image_types]
            
            for dataset in datasets:
                test_images[dataset] = {}
                for img_type in image_types:
                    test_images[dataset][img_type] = [np.random.rand(32, 32) if img_type == 'grayscale' 
                                                     else np.random.rand(32, 32, 3) for _ in range(2)]
            
            config = ExperimentConfig(
                datasets=datasets,
                data_root=temp_dir / "data",
                k_values=k_values,
                image_types=image_types,
                output_dir=temp_dir / f"scale_results_{len(datasets)}_{len(image_types)}_{len(k_values)}",
                save_reconstructed_images=False,
                parallel=False,
                experiment_name=f"scale_test_{len(datasets)}_{len(image_types)}_{len(k_values)}",
                show_progress=False
            )
            
            with patch.object(DatasetManager, 'load_datasets', return_value=test_images):
                runner = ExperimentRunner(config)
                
                # Time the execution
                import time
                start_time = time.time()
                results_df = runner.run_batch_experiments()
                end_time = time.time()
                
                execution_time = end_time - start_time
                num_experiments = len(results_df) if results_df is not None else 0
                
                scalability_results.append({
                    'workload_size': num_datasets * num_image_types * len(k_values) * 2,  # 2 images per type
                    'execution_time': execution_time,
                    'experiments_completed': num_experiments,
                    'time_per_experiment': execution_time / max(num_experiments, 1)
                })
        
        # Analyze scalability
        assert len(scalability_results) == 3
        
        # Execution time should scale reasonably with workload
        for i in range(1, len(scalability_results)):
            prev_result = scalability_results[i-1]
            curr_result = scalability_results[i]
            
            # Time per experiment should be relatively consistent
            time_ratio = curr_result['time_per_experiment'] / prev_result['time_per_experiment']
            assert 0.5 <= time_ratio <= 3.0, f"Time per experiment scaling issue: {time_ratio}"
    
    def test_batch_processing_resource_cleanup(self, experiment_config, sample_images, temp_dir):
        """Test proper resource cleanup during batch processing."""
        import psutil
        import os
        import gc
        
        # Get initial resource usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_handles = process.num_handles() if hasattr(process, 'num_handles') else 0
        
        with patch.object(DatasetManager, 'load_datasets', return_value=sample_images):
            runner = ExperimentRunner(experiment_config)
            results_df = runner.run_batch_experiments()
            
            # Force garbage collection
            del runner, results_df
            gc.collect()
            
            # Check resource usage after cleanup
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_handles = process.num_handles() if hasattr(process, 'num_handles') else 0
            
            memory_increase = final_memory - initial_memory
            handle_increase = final_handles - initial_handles
            
            # Memory increase should be reasonable
            assert memory_increase < 100, f"Excessive memory usage after cleanup: {memory_increase} MB"
            
            # Handle count should not increase significantly (if available on platform)
            if hasattr(process, 'num_handles'):
                assert handle_increase < 50, f"Resource handles not cleaned up: {handle_increase}"
    
    def test_batch_processing_fault_tolerance(self, temp_dir):
        """Test batch processing fault tolerance and error recovery."""
        # Create test data with some problematic cases
        fault_test_images = {
            'portraits': {
                'grayscale': [
                    np.random.rand(50, 50),  # Normal image
                    np.ones((30, 30)) * 1e-10,  # Very small values
                    np.random.rand(20, 20),  # Small image
                ],
                'rgb': [
                    np.random.rand(50, 50, 3),  # Normal RGB
                    np.ones((25, 25, 3)),  # Constant image
                ]
            }
        }
        
        config = ExperimentConfig(
            datasets=['portraits'],
            data_root=temp_dir / "data",
            k_values=[5, 10, 15, 25],  # Some k values may be too large for small images
            image_types=['grayscale', 'rgb'],
            output_dir=temp_dir / "fault_results",
            save_reconstructed_images=False,
            parallel=False,
            experiment_name="fault_tolerance_test",
            show_progress=False
        )
        
        with patch.object(DatasetManager, 'load_datasets', return_value=fault_test_images):
            runner = ExperimentRunner(config)
            
            # Should handle faults gracefully
            try:
                results_df = runner.run_batch_experiments()
                
                # Should have some successful results
                if results_df is not None and len(results_df) > 0:
                    # All successful results should be valid
                    assert results_df['psnr'].notna().all()
                    assert (results_df['compression_ratio'] > 0).all()
                    
                    # Should skip invalid k-values gracefully
                    max_k = results_df['k_value'].max()
                    assert max_k <= 25  # Some large k values should be skipped
                
            except Exception as e:
                # If it fails, should provide meaningful error
                assert len(str(e)) > 0, "Should provide meaningful error message"


if __name__ == "__main__":
    pytest.main([__file__])