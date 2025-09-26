"""
Experiment configuration and runner for batch SVD compression experiments.

This module provides the ExperimentRunner class for systematic batch processing
of image compression experiments with parallel processing support and
comprehensive progress tracking.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
from tqdm import tqdm

from ..compression.svd_compressor import SVDCompressor
from ..data.dataset_manager import DatasetManager
from ..evaluation.metrics_calculator import MetricsCalculator
from ..evaluation.performance_profiler import PerformanceProfiler


def _execute_experiment_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standalone function for executing a single experiment task.
    
    This function is defined at module level to be picklable for multiprocessing.
    
    Args:
        task_data: Dictionary containing task information and components
        
    Returns:
        Dictionary containing experiment results
    """
    task = task_data['task']
    save_images = task_data.get('save_images', False)
    output_dir = task_data.get('output_dir')
    
    # Initialize components (they need to be recreated in each process)
    compressor = SVDCompressor()
    metrics_calculator = MetricsCalculator()
    
    image = task['image']
    k_value = task['k_value']
    
    # Perform compression with timing
    start_time = time.time()
    
    try:
        compressed_image, compression_metadata = compressor.compress_image(image, k_value)
        compression_time = time.time() - start_time
        
        # Calculate quality metrics
        metrics = metrics_calculator.calculate_all_metrics(image, compressed_image, k_value)
        
        # Save reconstructed image if requested
        if save_images and output_dir:
            _save_reconstructed_image_standalone(compressed_image, task, output_dir)
        
        # Compile results
        result = {
            'experiment_id': task['experiment_id'],
            'dataset': task['dataset'],
            'image_name': task['image_name'],
            'image_type': task['image_type'],
            'image_index': task['image_index'],
            'k_value': k_value,
            'compression_time': compression_time,
            'timestamp': datetime.now().isoformat(),
            **metrics,
            **compression_metadata
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Compression failed for {task['experiment_id']}: {e}")
        raise


def _save_reconstructed_image_standalone(compressed_image: np.ndarray, task: Dict[str, Any], output_dir: Path) -> None:
    """Standalone function for saving reconstructed images."""
    try:
        from ..data.image_loader import ImageLoader
        
        # Create output directory for reconstructed images
        reconstructed_dir = Path(output_dir) / "reconstructed_images" / task['dataset']
        reconstructed_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = f"{task['image_name']}_{task['image_type']}_k{task['k_value']}.png"
        output_path = reconstructed_dir / filename
        
        # Save image using image loader
        image_loader = ImageLoader()
        image_loader.save_image(compressed_image, output_path)
        
    except Exception as e:
        logger.warning(f"Failed to save reconstructed image for {task['experiment_id']}: {e}")

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """
    Configuration for batch compression experiments.
    
    This dataclass defines all parameters needed for systematic
    batch experiments across datasets, images, and k-values.
    """
    
    # Dataset configuration
    datasets: List[str] = field(default_factory=lambda: ['portraits', 'landscapes', 'textures'])
    data_root: Union[str, Path] = "data"
    
    # Compression parameters
    k_values: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 50, 75, 100])
    image_types: List[str] = field(default_factory=lambda: ['grayscale', 'rgb'])
    
    # Output configuration
    output_dir: Union[str, Path] = "results"
    save_reconstructed_images: bool = True
    save_intermediate_results: bool = True
    
    # Processing configuration
    parallel: bool = True
    max_workers: Optional[int] = None
    chunk_size: int = 1
    
    # Experiment metadata
    experiment_name: str = "svd_compression_experiment"
    random_seed: int = 42
    
    # Progress and logging
    show_progress: bool = True
    log_level: str = "INFO"
    
    # Resume capability
    resume_from_checkpoint: bool = False
    checkpoint_interval: int = 10  # Save checkpoint every N images
    
    def __post_init__(self):
        """Validate and normalize configuration parameters."""
        # Convert paths to Path objects
        self.data_root = Path(self.data_root)
        self.output_dir = Path(self.output_dir)
        
        # Validate k_values
        if not self.k_values or not all(isinstance(k, int) and k > 0 for k in self.k_values):
            raise ValueError("k_values must be a list of positive integers")
        
        # Validate datasets
        if not self.datasets:
            raise ValueError("At least one dataset must be specified")
        
        # Validate image types
        valid_types = {'grayscale', 'rgb'}
        if not all(img_type in valid_types for img_type in self.image_types):
            raise ValueError(f"image_types must be subset of {valid_types}")
        
        # Set default max_workers if not specified
        if self.max_workers is None:
            self.max_workers = min(mp.cpu_count(), 4)  # Reasonable default
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ExperimentRunner:
    """
    Runner for systematic SVD compression experiments.
    
    This class orchestrates batch experiments across multiple datasets,
    images, and compression parameters with parallel processing support
    and comprehensive result tracking.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        
        # Initialize components
        self.compressor = SVDCompressor()
        self.dataset_manager = DatasetManager(
            data_root=config.data_root,
            target_size=(256, 256)
        )
        self.metrics_calculator = MetricsCalculator()
        self.profiler = PerformanceProfiler()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize result tracking
        self.results = []
        self.checkpoint_file = self.config.output_dir / f"{config.experiment_name}_checkpoint.csv"
        self.completed_experiments = set()
        
        # Load checkpoint if resuming
        if config.resume_from_checkpoint and self.checkpoint_file.exists():
            self._load_checkpoint()
        
        logger.info(f"Initialized ExperimentRunner with config: {config.experiment_name}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _load_checkpoint(self) -> None:
        """Load previous experiment results from checkpoint."""
        try:
            checkpoint_df = pd.read_csv(self.checkpoint_file)
            self.results = checkpoint_df.to_dict('records')
            
            # Track completed experiments
            for result in self.results:
                exp_id = self._get_experiment_id(
                    result['dataset'], result['image_name'], 
                    result['image_type'], result['k_value']
                )
                self.completed_experiments.add(exp_id)
            
            logger.info(f"Loaded {len(self.results)} results from checkpoint")
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            self.results = []
            self.completed_experiments = set()
    
    def _get_experiment_id(self, dataset: str, image_name: str, 
                          image_type: str, k_value: int) -> str:
        """Generate unique experiment identifier."""
        return f"{dataset}_{image_name}_{image_type}_k{k_value}"
    
    def _save_checkpoint(self) -> None:
        """Save current results as checkpoint."""
        if self.results:
            results_df = pd.DataFrame(self.results)
            results_df.to_csv(self.checkpoint_file, index=False)
            logger.debug(f"Saved checkpoint with {len(self.results)} results")
    
    def run_batch_experiments(self) -> pd.DataFrame:
        """
        Run systematic batch experiments across all configured parameters.
        
        Returns:
            DataFrame containing all experiment results
        """
        logger.info("Starting batch experiments")
        start_time = time.time()
        
        # Load datasets
        logger.info("Loading datasets...")
        datasets = self.dataset_manager.load_datasets()
        
        # Generate experiment tasks
        tasks = self._generate_experiment_tasks(datasets)
        logger.info(f"Generated {len(tasks)} experiment tasks")
        
        # Filter out completed tasks if resuming
        if self.config.resume_from_checkpoint:
            original_count = len(tasks)
            tasks = [task for task in tasks if task['experiment_id'] not in self.completed_experiments]
            logger.info(f"Resuming: {len(tasks)} remaining tasks (skipped {original_count - len(tasks)})")
        
        if not tasks:
            logger.info("No tasks to execute")
            return pd.DataFrame(self.results)
        
        # Execute experiments
        if self.config.parallel and len(tasks) > 1:
            self._run_parallel_experiments(tasks)
        else:
            self._run_sequential_experiments(tasks)
        
        # Create final results DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Save final results
        final_results_path = self.config.output_dir / f"{self.config.experiment_name}_results.csv"
        results_df.to_csv(final_results_path, index=False)
        
        total_time = time.time() - start_time
        logger.info(f"Completed batch experiments in {total_time:.2f} seconds")
        logger.info(f"Results saved to {final_results_path}")
        
        return results_df
    
    def _generate_experiment_tasks(self, datasets: Dict[str, Dict[str, List[np.ndarray]]]) -> List[Dict[str, Any]]:
        """
        Generate list of experiment tasks.
        
        Args:
            datasets: Loaded datasets
            
        Returns:
            List of experiment task dictionaries
        """
        tasks = []
        
        for dataset_name in self.config.datasets:
            if dataset_name not in datasets:
                logger.warning(f"Dataset {dataset_name} not found, skipping")
                continue
            
            dataset = datasets[dataset_name]
            
            for image_type in self.config.image_types:
                if image_type not in dataset:
                    logger.warning(f"Image type {image_type} not found in {dataset_name}, skipping")
                    continue
                
                images = dataset[image_type]
                
                for img_idx, image in enumerate(images):
                    image_name = f"image_{img_idx:03d}"
                    
                    for k_value in self.config.k_values:
                        # Check if k_value is valid for this image
                        min_dim = min(image.shape[:2])
                        if k_value > min_dim:
                            logger.debug(f"Skipping k={k_value} for {image_name} (exceeds min dimension {min_dim})")
                            continue
                        
                        experiment_id = self._get_experiment_id(dataset_name, image_name, image_type, k_value)
                        
                        task = {
                            'experiment_id': experiment_id,
                            'dataset': dataset_name,
                            'image_name': image_name,
                            'image_type': image_type,
                            'image': image,
                            'k_value': k_value,
                            'image_index': img_idx
                        }
                        tasks.append(task)
        
        return tasks
    
    def _run_sequential_experiments(self, tasks: List[Dict[str, Any]]) -> None:
        """Run experiments sequentially with progress tracking."""
        logger.info("Running experiments sequentially")
        
        if self.config.show_progress:
            tasks_iter = tqdm(tasks, desc="Processing experiments")
        else:
            tasks_iter = tasks
        
        for i, task in enumerate(tasks_iter):
            try:
                result = self._execute_single_experiment(task)
                self.results.append(result)
                
                # Save checkpoint periodically
                if (i + 1) % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()
                    
            except Exception as e:
                logger.error(f"Failed to execute task {task['experiment_id']}: {e}")
                logger.debug(traceback.format_exc())
                continue
        
        # Final checkpoint save
        self._save_checkpoint()
    
    def _run_parallel_experiments(self, tasks: List[Dict[str, Any]]) -> None:
        """Run experiments in parallel with progress tracking."""
        logger.info(f"Running experiments in parallel with {self.config.max_workers} workers")
        
        # Prepare task data for parallel processing
        task_data_list = []
        for task in tasks:
            task_data = {
                'task': task,
                'save_images': self.config.save_reconstructed_images,
                'output_dir': self.config.output_dir
            }
            task_data_list.append(task_data)
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks using the standalone function
            future_to_task = {
                executor.submit(_execute_experiment_task, task_data): task_data['task']
                for task_data in task_data_list
            }
            
            # Process completed tasks with progress bar
            if self.config.show_progress:
                futures_iter = tqdm(as_completed(future_to_task), total=len(tasks), desc="Processing experiments")
            else:
                futures_iter = as_completed(future_to_task)
            
            completed_count = 0
            for future in futures_iter:
                task = future_to_task[future]
                
                try:
                    result = future.result()
                    self.results.append(result)
                    completed_count += 1
                    
                    # Save checkpoint periodically
                    if completed_count % self.config.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                except Exception as e:
                    logger.error(f"Failed to execute task {task['experiment_id']}: {e}")
                    logger.debug(traceback.format_exc())
                    continue
        
        # Final checkpoint save
        self._save_checkpoint()
    
    def _execute_single_experiment(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single compression experiment.
        
        Args:
            task: Experiment task dictionary
            
        Returns:
            Dictionary containing experiment results
        """
        image = task['image']
        k_value = task['k_value']
        
        # Perform compression with timing
        start_time = time.time()
        
        try:
            compressed_image, compression_metadata = self.compressor.compress_image(image, k_value)
            compression_time = time.time() - start_time
            
            # Calculate quality metrics
            metrics = self.metrics_calculator.calculate_all_metrics(image, compressed_image, k_value)
            
            # Save reconstructed image if requested
            if self.config.save_reconstructed_images:
                self._save_reconstructed_image(compressed_image, task)
            
            # Compile results
            result = {
                'experiment_id': task['experiment_id'],
                'dataset': task['dataset'],
                'image_name': task['image_name'],
                'image_type': task['image_type'],
                'image_index': task['image_index'],
                'k_value': k_value,
                'compression_time': compression_time,
                'timestamp': datetime.now().isoformat(),
                **metrics,
                **compression_metadata
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Compression failed for {task['experiment_id']}: {e}")
            raise
    
    def _save_reconstructed_image(self, compressed_image: np.ndarray, task: Dict[str, Any]) -> None:
        """Save reconstructed image to disk."""
        try:
            # Create output directory for reconstructed images
            reconstructed_dir = self.config.output_dir / "reconstructed_images" / task['dataset']
            reconstructed_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = f"{task['image_name']}_{task['image_type']}_k{task['k_value']}.png"
            output_path = reconstructed_dir / filename
            
            # Save image using dataset manager's image loader
            self.dataset_manager.image_loader.save_image(compressed_image, output_path)
            
        except Exception as e:
            logger.warning(f"Failed to save reconstructed image for {task['experiment_id']}: {e}")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of completed experiments.
        
        Returns:
            Dictionary containing experiment summary
        """
        if not self.results:
            return {"message": "No experiments completed"}
        
        results_df = pd.DataFrame(self.results)
        
        summary = {
            'total_experiments': len(results_df),
            'datasets': results_df['dataset'].unique().tolist(),
            'image_types': results_df['image_type'].unique().tolist(),
            'k_values': sorted(results_df['k_value'].unique().tolist()),
            'avg_compression_time': results_df['compression_time'].mean(),
            'avg_psnr': results_df['psnr'].mean(),
            'avg_ssim': results_df['ssim'].mean(),
            'avg_compression_ratio': results_df['compression_ratio'].mean(),
            'experiment_duration': (
                pd.to_datetime(results_df['timestamp']).max() - 
                pd.to_datetime(results_df['timestamp']).min()
            ).total_seconds() if len(results_df) > 1 else 0
        }
        
        return summary