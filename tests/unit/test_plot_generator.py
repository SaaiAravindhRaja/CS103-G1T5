"""
Unit tests for the PlotGenerator class.

Tests the core plotting functionality including singular value plots,
styling configuration, and utility methods.
"""

import unittest
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from src.visualization.plot_generator import PlotGenerator


class TestPlotGenerator(unittest.TestCase):
    """Test cases for PlotGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.plot_generator = PlotGenerator()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create test singular values (exponential decay)
        self.test_singular_values = np.array([
            100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125, 0.390625, 0.1953125
        ])
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Close all matplotlib figures
        plt.close('all')
        
        # Remove temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test PlotGenerator initialization."""
        # Test default initialization
        pg = PlotGenerator()
        self.assertEqual(pg.figsize, (10, 6))
        self.assertEqual(pg.style, 'seaborn-v0_8')
        
        # Test custom initialization
        custom_pg = PlotGenerator(style='default', figsize=(8, 5))
        self.assertEqual(custom_pg.figsize, (8, 5))
        self.assertEqual(custom_pg.style, 'default')
    
    def test_plot_singular_values_basic(self):
        """Test basic singular value plotting functionality."""
        fig = self.plot_generator.plot_singular_values(
            self.test_singular_values,
            title="Test Singular Values"
        )
        
        # Check that figure was created
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that figure has one axes
        axes = fig.get_axes()
        self.assertEqual(len(axes), 1)
        
        ax = axes[0]
        
        # Check axes labels and title
        self.assertEqual(ax.get_xlabel(), 'Singular Value Index')
        self.assertEqual(ax.get_ylabel(), 'Singular Value (log scale)')
        self.assertEqual(ax.get_title(), 'Test Singular Values')
        
        # Check that y-axis is logarithmic
        self.assertEqual(ax.get_yscale(), 'log')
        
        # Check that data was plotted
        lines = ax.get_lines()
        self.assertGreater(len(lines), 0)
        
        # Check that the line data matches input
        line_data = lines[0].get_ydata()
        np.testing.assert_array_equal(line_data, self.test_singular_values)
    
    def test_plot_singular_values_with_save(self):
        """Test singular value plotting with file saving."""
        save_path = self.temp_dir / "test_singular_values.png"
        
        fig = self.plot_generator.plot_singular_values(
            self.test_singular_values,
            title="Test Save",
            save_path=save_path
        )
        
        # Check that file was saved
        self.assertTrue(save_path.exists())
        
        # Check file size is reasonable (not empty)
        self.assertGreater(save_path.stat().st_size, 1000)
    
    def test_find_elbow_point(self):
        """Test elbow point detection algorithm."""
        # Test with exponential decay
        elbow_idx = self.plot_generator._find_elbow_point(self.test_singular_values)
        
        # Elbow should be somewhere in the middle, not at extremes
        self.assertGreater(elbow_idx, 1)
        self.assertLess(elbow_idx, len(self.test_singular_values))
        
        # Test with edge cases
        short_values = np.array([10.0, 5.0])
        elbow_short = self.plot_generator._find_elbow_point(short_values)
        self.assertEqual(elbow_short, 1)
        
        # Test with single value
        single_value = np.array([10.0])
        elbow_single = self.plot_generator._find_elbow_point(single_value)
        self.assertEqual(elbow_single, 1)
    
    def test_create_subplot_grid(self):
        """Test subplot grid creation."""
        # Test single subplot
        fig, axes = self.plot_generator.create_subplot_grid(1, 1)
        self.assertIsInstance(fig, plt.Figure)
        self.assertEqual(len(axes), 1)
        
        # Test 2x2 grid
        fig, axes = self.plot_generator.create_subplot_grid(2, 2)
        self.assertEqual(axes.shape, (2, 2))
        
        # Test 1x3 grid (should be 1D array)
        fig, axes = self.plot_generator.create_subplot_grid(1, 3)
        self.assertEqual(axes.shape, (3,))
        
        # Test custom figsize
        fig, axes = self.plot_generator.create_subplot_grid(2, 2, figsize=(12, 8))
        self.assertEqual(fig.get_size_inches().tolist(), [12.0, 8.0])
    
    def test_save_plot(self):
        """Test plot saving functionality."""
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        save_path = self.temp_dir / "test_plot.png"
        
        # Test saving
        self.plot_generator.save_plot(fig, save_path)
        
        # Check file exists and has reasonable size
        self.assertTrue(save_path.exists())
        self.assertGreater(save_path.stat().st_size, 1000)
        
        # Test different format
        pdf_path = self.temp_dir / "test_plot.pdf"
        fig2, ax2 = plt.subplots()
        ax2.plot([1, 2, 3], [1, 4, 2])
        
        self.plot_generator.save_plot(fig2, pdf_path, format='pdf')
        self.assertTrue(pdf_path.exists())
    
    def test_get_color_palette(self):
        """Test color palette generation."""
        # Test different numbers of colors
        colors_3 = self.plot_generator.get_color_palette(3)
        self.assertEqual(len(colors_3), 3)
        
        colors_5 = self.plot_generator.get_color_palette(5)
        self.assertEqual(len(colors_5), 5)
        
        # Check that colors are hex strings
        for color in colors_3:
            self.assertIsInstance(color, str)
            self.assertTrue(color.startswith('#'))
            self.assertEqual(len(color), 7)  # #RRGGBB format
    
    def test_close_all_figures(self):
        """Test closing all figures."""
        # Create multiple figures
        fig1 = plt.figure()
        fig2 = plt.figure()
        fig3 = plt.figure()
        
        # Check figures exist
        self.assertEqual(len(plt.get_fignums()), 3)
        
        # Close all figures
        self.plot_generator.close_all_figures()
        
        # Check all figures are closed
        self.assertEqual(len(plt.get_fignums()), 0)
    
    def test_singular_values_edge_cases(self):
        """Test singular value plotting with edge cases."""
        # Test with very small values
        small_values = np.array([1e-10, 1e-11, 1e-12])
        fig = self.plot_generator.plot_singular_values(small_values)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with single value
        single_value = np.array([42.0])
        fig = self.plot_generator.plot_singular_values(single_value)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with zeros (should handle gracefully)
        with_zeros = np.array([10.0, 5.0, 0.0, 0.0])
        fig = self.plot_generator.plot_singular_values(with_zeros)
        self.assertIsInstance(fig, plt.Figure)
    
    @patch('matplotlib.pyplot.rcParams')
    def test_style_setup_fallback(self, mock_rcparams):
        """Test style setup with LaTeX fallback."""
        # Test that initialization doesn't crash even if LaTeX fails
        pg = PlotGenerator()
        self.assertIsInstance(pg, PlotGenerator)
    
    def test_directory_creation_on_save(self):
        """Test that save_plot creates directories if they don't exist."""
        nested_path = self.temp_dir / "nested" / "directory" / "plot.png"
        
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Save to nested path (should create directories)
        self.plot_generator.save_plot(fig, nested_path)
        
        # Check that file and directories were created
        self.assertTrue(nested_path.exists())
        self.assertTrue(nested_path.parent.exists())
    
    def test_plot_quality_vs_k(self):
        """Test quality vs k plotting functionality."""
        # Create mock results DataFrame
        
        data = {
            'dataset': ['portraits', 'portraits', 'landscapes', 'landscapes'] * 3,
            'k_value': [5, 10, 5, 10] * 3,
            'psnr': [25.5, 30.2, 28.1, 32.5] * 3,
            'ssim': [0.75, 0.85, 0.80, 0.90] * 3,
            'mse': [0.05, 0.03, 0.04, 0.02] * 3
        }
        results_df = pd.DataFrame(data)
        
        # Test PSNR plot
        fig = self.plot_generator.plot_quality_vs_k(results_df, metric='psnr')
        self.assertIsInstance(fig, plt.Figure)
        
        # Check axes properties
        ax = fig.get_axes()[0]
        self.assertEqual(ax.get_xlabel(), 'Number of Singular Values (k)')
        self.assertEqual(ax.get_ylabel(), 'PSNR (dB)')
        
        # Test SSIM plot
        fig = self.plot_generator.plot_quality_vs_k(results_df, metric='ssim')
        ax = fig.get_axes()[0]
        self.assertEqual(ax.get_ylabel(), 'SSIM')
        
        # Test with save path
        save_path = self.temp_dir / "quality_vs_k.png"
        fig = self.plot_generator.plot_quality_vs_k(results_df, save_path=save_path)
        self.assertTrue(save_path.exists())
    
    def test_plot_compression_analysis(self):
        """Test compression analysis scatter plot."""
        
        data = {
            'dataset': ['portraits', 'landscapes', 'textures'] * 4,
            'compression_ratio': [2.5, 3.2, 4.1, 5.8] * 3,
            'psnr': [28.5, 32.1, 35.2, 38.9] * 3,
            'ssim': [0.78, 0.85, 0.91, 0.95] * 3
        }
        results_df = pd.DataFrame(data)
        
        # Test PSNR scatter plot
        fig = self.plot_generator.plot_compression_analysis(results_df, quality_metric='psnr')
        self.assertIsInstance(fig, plt.Figure)
        
        ax = fig.get_axes()[0]
        self.assertEqual(ax.get_xlabel(), 'Compression Ratio')
        self.assertEqual(ax.get_ylabel(), 'PSNR (dB)')
        
        # Test SSIM scatter plot
        fig = self.plot_generator.plot_compression_analysis(results_df, quality_metric='ssim')
        ax = fig.get_axes()[0]
        self.assertEqual(ax.get_ylabel(), 'SSIM')
    
    def test_create_image_grid(self):
        """Test image grid creation."""
        # Create test images
        images = [
            np.random.rand(64, 64),  # Grayscale
            np.random.rand(64, 64, 3),  # RGB
            np.random.rand(64, 64),  # Another grayscale
        ]
        titles = ['Original', 'Compressed', 'Difference']
        
        # Test basic grid creation
        fig = self.plot_generator.create_image_grid(images, titles)
        self.assertIsInstance(fig, plt.Figure)
        
        # Check that correct number of subplots were created
        axes = fig.get_axes()
        # Should have at least 3 axes (one for each image)
        self.assertGreaterEqual(len(axes), 3)
        
        # Test with custom grid dimensions
        fig = self.plot_generator.create_image_grid(images, titles, nrows=1, ncols=3)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test error cases
        with self.assertRaises(ValueError):
            self.plot_generator.create_image_grid([], [])
        
        with self.assertRaises(ValueError):
            self.plot_generator.create_image_grid(images, ['Only one title'])
    
    def test_plot_multiple_metrics(self):
        """Test multiple metrics plotting."""
        
        data = {
            'dataset': ['portraits', 'landscapes'] * 6,
            'k_value': [5, 10, 15, 20, 25, 30] * 2,
            'psnr': [25.5, 30.2, 33.1, 35.5, 37.2, 38.5] * 2,
            'ssim': [0.75, 0.85, 0.90, 0.93, 0.95, 0.96] * 2
        }
        results_df = pd.DataFrame(data)
        
        # Test with multiple metrics
        fig = self.plot_generator.plot_multiple_metrics(results_df, metrics=['psnr', 'ssim'])
        self.assertIsInstance(fig, plt.Figure)
        
        # Should have 2 subplots
        axes = fig.get_axes()
        self.assertEqual(len(axes), 2)
        
        # Test with single metric
        fig = self.plot_generator.plot_multiple_metrics(results_df, metrics=['psnr'])
        axes = fig.get_axes()
        self.assertGreaterEqual(len(axes), 1)
    
    def test_save_to_results_dir(self):
        """Test saving plots to results directory."""
        # Create a simple figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        # Test saving to results directory
        results_dir = self.temp_dir / "results" / "plots"
        saved_path = self.plot_generator.save_to_results_dir(
            fig, "test_plot", results_dir=results_dir
        )
        
        # Check that file was saved in correct location
        expected_path = results_dir / "test_plot.png"
        self.assertEqual(saved_path, expected_path)
        self.assertTrue(saved_path.exists())
        self.assertTrue(results_dir.exists())
        
        # Test with different format
        saved_path = self.plot_generator.save_to_results_dir(
            fig, "test_plot_pdf", format='pdf', results_dir=results_dir
        )
        expected_path = results_dir / "test_plot_pdf.pdf"
        self.assertEqual(saved_path, expected_path)
        self.assertTrue(saved_path.exists())
    
    def test_plot_generation_performance(self):
        """Test that plot generation completes in reasonable time."""
        import time
        
        # Generate a moderately complex plot
        large_singular_values = np.logspace(2, -3, 100)  # 100 singular values
        
        start_time = time.time()
        fig = self.plot_generator.plot_singular_values(large_singular_values)
        end_time = time.time()
        
        # Should complete within 5 seconds
        self.assertLess(end_time - start_time, 5.0)
        self.assertIsInstance(fig, plt.Figure)
    
    def test_memory_cleanup(self):
        """Test that plots are properly cleaned up to prevent memory leaks."""
        initial_figs = len(plt.get_fignums())
        
        # Create multiple plots
        for i in range(5):
            fig = self.plot_generator.plot_singular_values(self.test_singular_values)
            # Don't explicitly close - test cleanup
        
        # Close all figures
        self.plot_generator.close_all_figures()
        
        # Should be back to initial state
        final_figs = len(plt.get_fignums())
        self.assertEqual(final_figs, 0)
    
    def test_plot_customization(self):
        """Test plot customization options."""
        # Test with custom colors
        colors = ['red', 'blue', 'green']
        palette = self.plot_generator.get_color_palette(3)
        self.assertEqual(len(palette), 3)
        
        # Test custom figure size
        custom_pg = PlotGenerator(figsize=(12, 8))
        fig = custom_pg.plot_singular_values(self.test_singular_values)
        self.assertEqual(fig.get_size_inches().tolist(), [12.0, 8.0])
    
    def test_error_handling_in_plots(self):
        """Test error handling in plot generation."""
        # Test with empty data
        empty_values = np.array([])
        fig = self.plot_generator.plot_singular_values(empty_values)
        self.assertIsInstance(fig, plt.Figure)
        
        # Test with invalid data types
        with self.assertRaises((TypeError, ValueError)):
            self.plot_generator.plot_singular_values("not an array")


if __name__ == '__main__':
    unittest.main()