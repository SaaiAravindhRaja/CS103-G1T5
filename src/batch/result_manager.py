"""
Result management and storage utilities for batch experiments.

This module provides the ResultManager class for structured result CSV generation,
systematic file naming, and batch experiment resumption capabilities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import json
import shutil
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)


class ResultManager:
    """
    Manager for batch experiment results and storage.
    
    This class handles structured result CSV generation, systematic file naming
    for reconstructed images, and provides batch experiment resumption capabilities.
    """
    
    def __init__(self, output_dir: Union[str, Path], experiment_name: str = "svd_experiment"):
        """
        Initialize the result manager.
        
        Args:
            output_dir: Base directory for storing results
            experiment_name: Name of the experiment for file naming
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        
        # Create directory structure
        self.results_dir = self.output_dir / "results"
        self.images_dir = self.output_dir / "reconstructed_images"
        self.metadata_dir = self.output_dir / "metadata"
        
        self._create_directory_structure()
        
        # Result tracking
        self.results_file = self.results_dir / f"{experiment_name}_results.csv"
        self.metadata_file = self.metadata_dir / f"{experiment_name}_metadata.json"
        self.checkpoint_file = self.results_dir / f"{experiment_name}_checkpoint.csv"
        
        logger.info(f"Initialized ResultManager for experiment: {experiment_name}")
    
    def _create_directory_structure(self) -> None:
        """Create the directory structure for storing results."""
        directories = [
            self.results_dir,
            self.images_dir,
            self.metadata_dir,
            self.images_dir / "portraits",
            self.images_dir / "landscapes", 
            self.images_dir / "textures"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug("Created result directory structure")
    
    def save_results(self, results: List[Dict[str, Any]], 
                    metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save experiment results to structured CSV file.
        
        Args:
            results: List of experiment result dictionaries
            metadata: Optional experiment metadata
            
        Returns:
            Path to saved results file
        """
        if not results:
            logger.warning("No results to save")
            return self.results_file
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add derived columns for analysis
        results_df = self._add_derived_columns(results_df)
        
        # Sort by dataset, image, and k_value for consistency
        sort_columns = ['dataset', 'image_name', 'image_type', 'k_value']
        available_columns = [col for col in sort_columns if col in results_df.columns]
        if available_columns:
            results_df = results_df.sort_values(available_columns)
        
        # Save results
        results_df.to_csv(self.results_file, index=False)
        logger.info(f"Saved {len(results_df)} results to {self.results_file}")
        
        # Save metadata if provided
        if metadata:
            self.save_metadata(metadata)
        
        return self.results_file
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns for enhanced analysis."""
        df = df.copy()
        
        # Add quality categories based on PSNR
        if 'psnr' in df.columns:
            df['quality_category'] = pd.cut(
                df['psnr'],
                bins=[0, 20, 30, 40, float('inf')],
                labels=['Low', 'Medium', 'High', 'Excellent']
            )
        
        # Add compression efficiency metric
        if 'compression_ratio' in df.columns and 'psnr' in df.columns:
            df['compression_efficiency'] = df['psnr'] / df['compression_ratio']
        
        # Add relative k value (k as percentage of max possible)
        if 'k_value' in df.columns and 'image_shape' in df.columns:
            def calc_relative_k(row):
                try:
                    shape = eval(row['image_shape']) if isinstance(row['image_shape'], str) else row['image_shape']
                    max_k = min(shape[:2])
                    return (row['k_value'] / max_k) * 100
                except:
                    return None
            
            df['relative_k_percent'] = df.apply(calc_relative_k, axis=1)
        
        return df
    
    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """
        Save experiment metadata to JSON file.
        
        Args:
            metadata: Experiment metadata dictionary
            
        Returns:
            Path to saved metadata file
        """
        # Add timestamp and system info
        enhanced_metadata = {
            'experiment_name': self.experiment_name,
            'save_timestamp': datetime.now().isoformat(),
            'result_manager_version': '1.0',
            **metadata
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(enhanced_metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {self.metadata_file}")
        return self.metadata_file
    
    def load_results(self) -> Optional[pd.DataFrame]:
        """
        Load existing results from CSV file.
        
        Returns:
            DataFrame with results or None if file doesn't exist
        """
        if not self.results_file.exists():
            logger.info("No existing results file found")
            return None
        
        try:
            results_df = pd.read_csv(self.results_file)
            logger.info(f"Loaded {len(results_df)} results from {self.results_file}")
            return results_df
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return None
    
    def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load experiment metadata from JSON file.
        
        Returns:
            Metadata dictionary or None if file doesn't exist
        """
        if not self.metadata_file.exists():
            logger.info("No existing metadata file found")
            return None
        
        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {self.metadata_file}")
            return metadata
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None
    
    def generate_systematic_filename(self, dataset: str, image_name: str, 
                                   image_type: str, k_value: int, 
                                   extension: str = "png") -> str:
        """
        Generate systematic filename for reconstructed images.
        
        Args:
            dataset: Dataset name (portraits, landscapes, textures)
            image_name: Base image name
            image_type: Image type (grayscale, rgb)
            k_value: Compression parameter
            extension: File extension
            
        Returns:
            Systematic filename string
        """
        # Sanitize inputs
        dataset = str(dataset).lower().replace(' ', '_')
        image_name = str(image_name).replace(' ', '_')
        image_type = str(image_type).lower()
        
        # Generate filename: dataset_imagename_type_kXX.ext
        filename = f"{dataset}_{image_name}_{image_type}_k{k_value:03d}.{extension}"
        
        return filename
    
    def get_image_save_path(self, dataset: str, image_name: str, 
                           image_type: str, k_value: int) -> Path:
        """
        Get full path for saving reconstructed image.
        
        Args:
            dataset: Dataset name
            image_name: Base image name
            image_type: Image type
            k_value: Compression parameter
            
        Returns:
            Full path for saving the image
        """
        filename = self.generate_systematic_filename(dataset, image_name, image_type, k_value)
        dataset_dir = self.images_dir / dataset.lower()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        return dataset_dir / filename
    
    def save_checkpoint(self, results: List[Dict[str, Any]]) -> Path:
        """
        Save checkpoint for experiment resumption.
        
        Args:
            results: Current experiment results
            
        Returns:
            Path to checkpoint file
        """
        if not results:
            return self.checkpoint_file
        
        checkpoint_df = pd.DataFrame(results)
        checkpoint_df.to_csv(self.checkpoint_file, index=False)
        
        logger.debug(f"Saved checkpoint with {len(results)} results")
        return self.checkpoint_file
    
    def load_checkpoint(self) -> Tuple[List[Dict[str, Any]], set]:
        """
        Load checkpoint for experiment resumption.
        
        Returns:
            Tuple of (results_list, completed_experiment_ids)
        """
        if not self.checkpoint_file.exists():
            logger.info("No checkpoint file found")
            return [], set()
        
        try:
            checkpoint_df = pd.read_csv(self.checkpoint_file)
            results = checkpoint_df.to_dict('records')
            
            # Generate set of completed experiment IDs
            completed_ids = set()
            for result in results:
                exp_id = self._generate_experiment_id(result)
                completed_ids.add(exp_id)
            
            logger.info(f"Loaded checkpoint with {len(results)} completed experiments")
            return results, completed_ids
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return [], set()
    
    def _generate_experiment_id(self, result: Dict[str, Any]) -> str:
        """Generate experiment ID from result dictionary."""
        return f"{result.get('dataset', '')}_{result.get('image_name', '')}_{result.get('image_type', '')}_k{result.get('k_value', 0)}"
    
    def cleanup_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Removed checkpoint file")
    
    def generate_summary_report(self, results_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Generate comprehensive summary report of experiment results.
        
        Args:
            results_df: Results DataFrame, or None to load from file
            
        Returns:
            Dictionary containing summary statistics
        """
        if results_df is None:
            results_df = self.load_results()
        
        if results_df is None or results_df.empty:
            return {"error": "No results available for summary"}
        
        summary = {
            'experiment_info': {
                'name': self.experiment_name,
                'total_experiments': len(results_df),
                'datasets': results_df['dataset'].unique().tolist() if 'dataset' in results_df.columns else [],
                'image_types': results_df['image_type'].unique().tolist() if 'image_type' in results_df.columns else [],
                'k_values': sorted(results_df['k_value'].unique().tolist()) if 'k_value' in results_df.columns else []
            }
        }
        
        # Performance metrics
        if 'compression_time' in results_df.columns:
            summary['performance'] = {
                'avg_compression_time': results_df['compression_time'].mean(),
                'total_processing_time': results_df['compression_time'].sum(),
                'min_compression_time': results_df['compression_time'].min(),
                'max_compression_time': results_df['compression_time'].max()
            }
        
        # Quality metrics
        quality_metrics = ['psnr', 'ssim', 'mse']
        summary['quality'] = {}
        for metric in quality_metrics:
            if metric in results_df.columns:
                summary['quality'][metric] = {
                    'mean': results_df[metric].mean(),
                    'std': results_df[metric].std(),
                    'min': results_df[metric].min(),
                    'max': results_df[metric].max(),
                    'median': results_df[metric].median()
                }
        
        # Compression metrics
        if 'compression_ratio' in results_df.columns:
            summary['compression'] = {
                'avg_compression_ratio': results_df['compression_ratio'].mean(),
                'min_compression_ratio': results_df['compression_ratio'].min(),
                'max_compression_ratio': results_df['compression_ratio'].max()
            }
        
        # Best performing experiments
        if 'psnr' in results_df.columns:
            best_psnr_idx = results_df['psnr'].idxmax()
            summary['best_quality'] = results_df.loc[best_psnr_idx].to_dict()
        
        if 'compression_ratio' in results_df.columns:
            best_compression_idx = results_df['compression_ratio'].idxmax()
            summary['best_compression'] = results_df.loc[best_compression_idx].to_dict()
        
        return summary
    
    def export_results_for_analysis(self, output_format: str = 'csv') -> List[Path]:
        """
        Export results in various formats for external analysis.
        
        Args:
            output_format: Export format ('csv', 'json', 'excel', 'all')
            
        Returns:
            List of paths to exported files
        """
        results_df = self.load_results()
        if results_df is None:
            logger.warning("No results to export")
            return []
        
        exported_files = []
        base_name = f"{self.experiment_name}_export"
        
        if output_format in ['csv', 'all']:
            csv_path = self.results_dir / f"{base_name}.csv"
            results_df.to_csv(csv_path, index=False)
            exported_files.append(csv_path)
        
        if output_format in ['json', 'all']:
            json_path = self.results_dir / f"{base_name}.json"
            results_df.to_json(json_path, orient='records', indent=2)
            exported_files.append(json_path)
        
        if output_format in ['excel', 'all']:
            try:
                excel_path = self.results_dir / f"{base_name}.xlsx"
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    results_df.to_excel(writer, sheet_name='Results', index=False)
                    
                    # Add summary sheet
                    summary = self.generate_summary_report(results_df)
                    summary_df = pd.json_normalize(summary)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                exported_files.append(excel_path)
            except ImportError:
                logger.warning("openpyxl not available, skipping Excel export")
        
        logger.info(f"Exported results to {len(exported_files)} files")
        return exported_files
    
    def archive_experiment(self, archive_name: Optional[str] = None) -> Path:
        """
        Create archive of complete experiment results.
        
        Args:
            archive_name: Name for archive, or None for auto-generated name
            
        Returns:
            Path to created archive
        """
        if archive_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"{self.experiment_name}_{timestamp}"
        
        archive_path = self.output_dir / f"{archive_name}.zip"
        
        # Create temporary directory for archive contents
        temp_dir = self.output_dir / "temp_archive"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Copy all result files to temp directory
            if self.results_file.exists():
                shutil.copy2(self.results_file, temp_dir)
            if self.metadata_file.exists():
                shutil.copy2(self.metadata_file, temp_dir)
            
            # Copy reconstructed images
            if self.images_dir.exists():
                shutil.copytree(self.images_dir, temp_dir / "reconstructed_images", dirs_exist_ok=True)
            
            # Create archive
            shutil.make_archive(str(archive_path.with_suffix('')), 'zip', temp_dir)
            
            logger.info(f"Created experiment archive: {archive_path}")
            
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        
        return archive_path