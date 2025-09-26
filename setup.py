#!/usr/bin/env python3
"""
SVD Image Compression - Setup Configuration
Academic project for image compression using Singular Value Decomposition
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "SVD Image Compression - Academic implementation of image compression using Singular Value Decomposition"

# Read version from src/__init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'src', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="svd-image-compression",
    version=get_version(),
    author="Academic Project",
    author_email="student@university.edu",
    description="Academic implementation of image compression using Singular Value Decomposition",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/svd-image-compression",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "pillow>=8.3.0",
        "matplotlib>=3.5.0",
        "pandas>=1.3.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "seaborn>=0.11.0",
        "jupyter>=1.0.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.991",
            "flake8>=5.0.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "web": [
            "streamlit>=1.25.0",
            "plotly>=5.15.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "svd-compress=cli.main:main",
            "svd-batch=cli.batch:main",
            "svd-webapp=webapp.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    zip_safe=False,
    keywords="image-compression svd singular-value-decomposition academic research",
    project_urls={
        "Bug Reports": "https://github.com/username/svd-image-compression/issues",
        "Source": "https://github.com/username/svd-image-compression",
        "Documentation": "https://svd-image-compression.readthedocs.io/",
    },
)