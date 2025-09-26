"""
Plot generation module for SVD image compression analysis.

This module provides the PlotGenerator class for creating professional,
publication-quality visualizations of SVD compression results.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import seaborn as sns
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple, Union
import warnings

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class PlotGenerator:
    """
    Professional plot generator for SVD image compression analysis.
    
    This class provides methods to create publication-quality visualizations
    including singular value decay plots, quality metrics analysis, and
    image comparison grids with consistent styling.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        """
        Initialize the PlotGenerator with professional styling.
        
        Args:
            style: Matplotlib style to use for plots
            figsize: Default figure size as (width, height)
        """
        self.style = style
        self.figsize = figsize
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Configure matplotlib and seaborn for professional appearance."""
        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except OSError:
            # Fallback to default if seaborn style not available
            plt.style.use('default')
        
        # Configure seaborn color palette
        sns.set_palette("husl")
        
        # Set matplotlib parameters for publication quality
        plt.rcParams.update({
            'figure.figsize': self.figsize,
            'figure.dpi': 100,
            'savefig.dpi': 300,
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })
        
        # Enable LaTeX rendering if available
        try:
            # Test if LaTeX is available by trying to render a simple expression
            import subprocess
            subprocess.check_output(['latex', '--version'], stderr=subprocess.DEVNULL)
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'serif'
        except (FileNotFoundError, subprocess.CalledProcessError, OSError):
            # Fallback to mathtext if LaTeX not available
            plt.rcParams['text.usetex'] = False
            plt.rcParams['mathtext.default'] = 'regular'
    
    def plot_singular_values(
        self, 
        singular_values: np.ndarray, 
        title: str = "Singular Value Decay",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create a logarithmic plot of singular value decay.
        
        Args:
            singular_values: Array of singular values in descending order
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create x-axis (singular value indices)
        indices = np.arange(1, len(singular_values) + 1)
        
        # Plot singular values on log scale
        ax.semilogy(indices, singular_values, 'b-', linewidth=2, alpha=0.8)
        ax.scatter(indices[::max(1, len(indices)//20)], 
                  singular_values[::max(1, len(indices)//20)], 
                  c='red', s=30, alpha=0.7, zorder=5)
        
        # Styling
        ax.set_xlabel('Singular Value Index')
        ax.set_ylabel('Singular Value (log scale)')
        ax.set_title(title, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key points
        if len(singular_values) > 10:
            # Annotate the "elbow" point (approximate)
            elbow_idx = self._find_elbow_point(singular_values)
            ax.annotate(f'Elbow: k={elbow_idx}', 
                       xy=(elbow_idx, singular_values[elbow_idx-1]),
                       xytext=(elbow_idx + len(singular_values)*0.1, 
                              singular_values[elbow_idx-1]*2),
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        
        return fig
    
    def _find_elbow_point(self, singular_values: np.ndarray) -> int:
        """
        Find the approximate elbow point in singular value decay.
        
        Uses the method of maximum curvature to identify the optimal
        number of singular values to retain.
        
        Args:
            singular_values: Array of singular values in descending order
            
        Returns:
            Index of the elbow point (1-based)
        """
        if len(singular_values) < 3:
            return 1
        
        # Normalize values to [0, 1] for curvature calculation
        normalized = (singular_values - singular_values.min()) / (
            singular_values.max() - singular_values.min() + 1e-10
        )
        
        # Calculate second derivative (curvature)
        x = np.arange(len(normalized))
        curvature = np.gradient(np.gradient(normalized))
        
        # Find point of maximum curvature (most negative second derivative)
        elbow_idx = np.argmin(curvature[1:-1]) + 1  # Exclude endpoints
        
        return elbow_idx + 1  # Convert to 1-based indexing
    
    def create_subplot_grid(
        self, 
        nrows: int, 
        ncols: int, 
        figsize: Optional[Tuple[int, int]] = None
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a subplot grid with consistent styling.
        
        Args:
            nrows: Number of subplot rows
            ncols: Number of subplot columns
            figsize: Optional figure size override
            
        Returns:
            Tuple of (figure, axes array)
        """
        if figsize is None:
            figsize = (self.figsize[0] * ncols * 0.8, self.figsize[1] * nrows * 0.8)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        # Ensure axes is always a numpy array
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        elif nrows == 1 or ncols == 1:
            axes = np.array(axes)
        
        return fig, axes
    
    def save_plot(
        self, 
        fig: plt.Figure, 
        filepath: Path, 
        format: str = 'png',
        close_after_save: bool = True
    ) -> None:
        """
        Save a plot with consistent settings.
        
        Args:
            fig: matplotlib Figure to save
            filepath: Path where to save the plot
            format: File format ('png', 'pdf', 'svg', etc.)
            close_after_save: Whether to close the figure after saving
        """
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with high quality settings
        fig.savefig(
            filepath,
            format=format,
            dpi=300,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        
        if close_after_save:
            plt.close(fig)
    
    def get_color_palette(self, n_colors: int) -> List[str]:
        """
        Get a consistent color palette for multiple series.
        
        Args:
            n_colors: Number of colors needed
            
        Returns:
            List of color hex codes
        """
        return sns.color_palette("husl", n_colors).as_hex()
    
    def plot_quality_vs_k(
        self,
        results_df,
        metric: str = 'psnr',
        dataset_column: str = 'dataset',
        k_column: str = 'k_value',
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create quality metric vs k-value plots grouped by dataset.
        
        Args:
            results_df: DataFrame with experiment results
            metric: Quality metric to plot ('psnr', 'ssim', 'mse')
            dataset_column: Column name for dataset grouping
            k_column: Column name for k-values
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get unique datasets and colors
        datasets = results_df[dataset_column].unique()
        colors = self.get_color_palette(len(datasets))
        
        # Plot each dataset
        for i, dataset in enumerate(datasets):
            dataset_data = results_df[results_df[dataset_column] == dataset]
            
            # Group by k-value and calculate mean and std
            grouped = dataset_data.groupby(k_column)[metric].agg(['mean', 'std']).reset_index()
            
            # Plot line with error bars
            ax.errorbar(
                grouped[k_column], 
                grouped['mean'],
                yerr=grouped['std'],
                label=dataset.title(),
                color=colors[i],
                marker='o',
                markersize=6,
                linewidth=2,
                capsize=4,
                alpha=0.8
            )
        
        # Styling
        metric_labels = {
            'psnr': 'PSNR (dB)',
            'ssim': 'SSIM',
            'mse': 'MSE'
        }
        
        ax.set_xlabel('Number of Singular Values (k)')
        ax.set_ylabel(metric_labels.get(metric, metric.upper()))
        ax.set_title(f'{metric_labels.get(metric, metric.upper())} vs k-value by Dataset', 
                    fontweight='bold', pad=20)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        if metric == 'psnr':
            ax.set_ylim(bottom=0)
        elif metric == 'ssim':
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path, close_after_save=False)
        
        return fig
    
    def plot_compression_analysis(
        self,
        results_df,
        quality_metric: str = 'psnr',
        compression_column: str = 'compression_ratio',
        dataset_column: str = 'dataset',
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create scatter plot of compression ratio vs quality metric.
        
        Args:
            results_df: DataFrame with experiment results
            quality_metric: Quality metric for y-axis ('psnr', 'ssim')
            compression_column: Column name for compression ratios
            dataset_column: Column name for dataset grouping
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get unique datasets and colors
        datasets = results_df[dataset_column].unique()
        colors = self.get_color_palette(len(datasets))
        
        # Plot each dataset
        for i, dataset in enumerate(datasets):
            dataset_data = results_df[results_df[dataset_column] == dataset]
            
            ax.scatter(
                dataset_data[compression_column],
                dataset_data[quality_metric],
                label=dataset.title(),
                color=colors[i],
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5
            )
        
        # Styling
        quality_labels = {
            'psnr': 'PSNR (dB)',
            'ssim': 'SSIM',
            'mse': 'MSE'
        }
        
        ax.set_xlabel('Compression Ratio')
        ax.set_ylabel(quality_labels.get(quality_metric, quality_metric.upper()))
        ax.set_title(f'Compression Ratio vs {quality_labels.get(quality_metric, quality_metric.upper())}',
                    fontweight='bold', pad=20)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        # Set reasonable axis limits
        if quality_metric == 'ssim':
            ax.set_ylim(0, 1)
        elif quality_metric == 'psnr':
            ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path, close_after_save=False)
        
        return fig
    
    def create_image_grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create a grid of images for visual comparison.
        
        Args:
            images: List of image arrays to display
            titles: List of titles for each image
            nrows: Number of rows (auto-calculated if None)
            ncols: Number of columns (auto-calculated if None)
            figsize: Figure size override
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        n_images = len(images)
        
        if n_images == 0:
            raise ValueError("No images provided")
        
        if len(titles) != n_images:
            raise ValueError("Number of titles must match number of images")
        
        # Auto-calculate grid dimensions
        if nrows is None and ncols is None:
            ncols = min(4, n_images)  # Max 4 columns
            nrows = (n_images + ncols - 1) // ncols
        elif nrows is None:
            nrows = (n_images + ncols - 1) // ncols
        elif ncols is None:
            ncols = (n_images + nrows - 1) // nrows
        
        # Calculate figure size
        if figsize is None:
            figsize = (ncols * 4, nrows * 4)
        
        fig, axes = self.create_subplot_grid(nrows, ncols, figsize)
        
        # Flatten axes array for easier indexing
        if nrows == 1 and ncols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Display images
        for i in range(n_images):
            ax = axes[i]
            
            # Handle grayscale vs RGB images
            if len(images[i].shape) == 2:
                # Grayscale
                im = ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
            else:
                # RGB
                im = ax.imshow(np.clip(images[i], 0, 1))
            
            ax.set_title(titles[i], fontweight='bold', pad=10)
            ax.axis('off')
            
            # Add colorbar for grayscale images
            if len(images[i].shape) == 2:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path, close_after_save=False)
        
        return fig
    
    def plot_multiple_metrics(
        self,
        results_df,
        metrics: List[str] = ['psnr', 'ssim'],
        dataset_column: str = 'dataset',
        k_column: str = 'k_value',
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create subplots showing multiple quality metrics vs k-value.
        
        Args:
            results_df: DataFrame with experiment results
            metrics: List of metrics to plot
            dataset_column: Column name for dataset grouping
            k_column: Column name for k-values
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        n_metrics = len(metrics)
        ncols = min(2, n_metrics)
        nrows = (n_metrics + ncols - 1) // ncols
        
        fig, axes = self.create_subplot_grid(nrows, ncols, 
                                           figsize=(ncols * 6, nrows * 5))
        
        # Handle different axes configurations
        if n_metrics == 1:
            if isinstance(axes, np.ndarray) and axes.size == 1:
                axes = [axes.item()]
            elif not isinstance(axes, (list, np.ndarray)):
                axes = [axes]
        else:
            if isinstance(axes, np.ndarray):
                axes = axes.flatten()
            else:
                axes = [axes]
        
        # Get unique datasets and colors
        datasets = results_df[dataset_column].unique()
        colors = self.get_color_palette(len(datasets))
        
        metric_labels = {
            'psnr': 'PSNR (dB)',
            'ssim': 'SSIM',
            'mse': 'MSE',
            'compression_ratio': 'Compression Ratio'
        }
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Plot each dataset
            for j, dataset in enumerate(datasets):
                dataset_data = results_df[results_df[dataset_column] == dataset]
                
                # Group by k-value and calculate mean
                grouped = dataset_data.groupby(k_column)[metric].mean().reset_index()
                
                ax.plot(
                    grouped[k_column], 
                    grouped[metric],
                    label=dataset.title(),
                    color=colors[j],
                    marker='o',
                    markersize=4,
                    linewidth=2,
                    alpha=0.8
                )
            
            # Styling
            ax.set_xlabel('Number of Singular Values (k)')
            ax.set_ylabel(metric_labels.get(metric, metric.upper()))
            ax.set_title(f'{metric_labels.get(metric, metric.upper())} vs k-value',
                        fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Set reasonable axis limits
            if metric == 'ssim':
                ax.set_ylim(0, 1)
            elif metric == 'psnr':
                ax.set_ylim(bottom=0)
            
            # Add legend to first subplot only
            if i == 0:
                ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            self.save_plot(fig, save_path, close_after_save=False)
        
        return fig
    
    def save_to_results_dir(
        self,
        fig: plt.Figure,
        filename: str,
        format: str = 'png',
        results_dir: Path = Path('results/plots'),
        close_after_save: bool = True
    ) -> Path:
        """
        Save a plot to the results/plots directory with standardized naming.
        
        Args:
            fig: matplotlib Figure to save
            filename: Base filename (without extension)
            format: File format ('png', 'pdf', 'svg', etc.)
            results_dir: Directory to save plots in
            close_after_save: Whether to close the figure after saving
            
        Returns:
            Path to the saved file
        """
        # Ensure results directory exists
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create full filepath
        filepath = results_dir / f"{filename}.{format}"
        
        # Save the plot
        self.save_plot(fig, filepath, format=format, close_after_save=close_after_save)
        
        return filepath
    
    def close_all_figures(self) -> None:
        """Close all open matplotlib figures to free memory."""
        plt.close('all')