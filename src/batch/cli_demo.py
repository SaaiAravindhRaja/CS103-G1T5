#!/usr/bin/env python3
"""
CLI demonstration script for batch SVD compression experiments.

This script provides a simple command-line interface to run batch experiments
using the SVD compression framework.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from typing import Dict, List

from .experiment_runner import ExperimentRunner, ExperimentConfig
from .result_manager import ResultManager


def create_sample_data(output_dir: Path, num_images: int = 3) -> None:
    """Create sample image data for demonstration."""
    print("Creating sample image data...")
    
    # Create data directories
    for category in ['portraits', 'landscapes', 'textures']:
        category_dir = output_dir / 'data' / category / 'original'
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample images
        for i in range(num_images):
            # Create random image data
            img_data = np.random.rand(64, 64, 3)  # Small RGB images for demo
            
            # Save as numpy array (in real scenario, these would be actual image files)
            np.save(category_dir / f"sample_{i:02d}.npy", img_data)
    
    print(f"Created {num_images} sample images per category")


def run_demo_experiment(data_dir: Path, output_dir: Path, 
                       k_values: List[int] = None, parallel: bool = False) -> None:
    """Run a demonstration batch experiment."""
    
    if k_values is None:
        k_values = [5, 10, 20, 30]
    
    print(f"Running batch experiment with k_values: {k_values}")
    
    # Create experiment configuration
    config = ExperimentConfig(
        datasets=['portraits', 'landscapes', 'textures'],
        data_root=data_dir,
        k_values=k_values,
        image_types=['grayscale', 'rgb'],
        output_dir=output_dir,
        save_reconstructed_images=True,
        parallel=parallel,
        experiment_name="demo_batch_experiment",
        show_progress=True,
        checkpoint_interval=5
    )
    
    # Initialize and run experiment
    runner = ExperimentRunner(config)
    results_df = runner.run_batch_experiments()
    
    # Print summary
    summary = runner.get_experiment_summary()
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Datasets: {summary['datasets']}")
    print(f"K-values: {summary['k_values']}")
    print(f"Average PSNR: {summary['avg_psnr']:.2f} dB")
    print(f"Average SSIM: {summary['avg_ssim']:.3f}")
    print(f"Average compression ratio: {summary['avg_compression_ratio']:.2f}")
    print(f"Average processing time: {summary['avg_compression_time']:.3f} seconds")
    
    # Generate detailed report
    result_manager = ResultManager(output_dir, config.experiment_name)
    detailed_summary = result_manager.generate_summary_report(results_df)
    
    print("\n" + "="*50)
    print("DETAILED ANALYSIS")
    print("="*50)
    
    if 'quality' in detailed_summary:
        for metric, stats in detailed_summary['quality'].items():
            print(f"{metric.upper()}:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Std:  {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
    
    # Export results
    exported_files = result_manager.export_results_for_analysis('csv')
    print(f"\nResults exported to: {exported_files}")
    
    return results_df


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="SVD Image Compression Batch Experiment Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic demo with sample data
  python -m src.batch.cli_demo --create-sample-data --output-dir ./demo_results
  
  # Run with custom k-values
  python -m src.batch.cli_demo --k-values 5 10 15 25 --output-dir ./demo_results
  
  # Run with parallel processing
  python -m src.batch.cli_demo --parallel --output-dir ./demo_results
        """
    )
    
    parser.add_argument(
        '--output-dir', 
        type=Path, 
        default=Path('./batch_demo_results'),
        help='Output directory for results (default: ./batch_demo_results)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=Path,
        help='Data directory (default: output-dir/data)'
    )
    
    parser.add_argument(
        '--create-sample-data',
        action='store_true',
        help='Create sample image data for demonstration'
    )
    
    parser.add_argument(
        '--k-values',
        type=int,
        nargs='+',
        default=[5, 10, 20],
        help='K-values for compression (default: 5 10 20)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Use parallel processing'
    )
    
    parser.add_argument(
        '--num-sample-images',
        type=int,
        default=2,
        help='Number of sample images per category (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Set default data directory
    if args.data_dir is None:
        args.data_dir = args.output_dir / 'data'
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create sample data if requested
        if args.create_sample_data:
            create_sample_data(args.output_dir, args.num_sample_images)
        
        # Check if data exists
        if not args.data_dir.exists():
            print(f"Error: Data directory {args.data_dir} does not exist.")
            print("Use --create-sample-data to generate sample data.")
            sys.exit(1)
        
        # Run experiment
        print(f"Starting batch experiment...")
        print(f"Data directory: {args.data_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"K-values: {args.k_values}")
        print(f"Parallel processing: {args.parallel}")
        
        results_df = run_demo_experiment(
            args.data_dir, 
            args.output_dir, 
            args.k_values, 
            args.parallel
        )
        
        print(f"\nDemo completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()