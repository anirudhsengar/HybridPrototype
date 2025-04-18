#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Setup script for the Hybrid Bug Predictor package.

This package combines temporal patterns (FixCache) with code metrics analysis (REPD)
for improved software defect prediction.
"""

from setuptools import setup, find_packages
import os

# Read the long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Package metadata
setup(
    name="hybrid-bug-predictor",
    version="0.1.0",
    description="Hybrid approach to software defect prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="anirudhsengar",
    author_email="anirudhsengar3@gmail.com",
    url="https://github.com/anirudhsengar/HybridPrototype",

    # Package configuration
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",

    # Dependencies
    install_requires=[
        "gitpython>=3.1.0",  # Git repository interaction
        "numpy>=1.20.0",  # Numerical operations
        "pandas>=1.3.0",  # Data manipulation
        "scikit-learn>=1.0.0",  # Machine learning components for REPD
    ],

    # Optional dependencies
    extras_require={
        "visualization": [
            "matplotlib>=3.4.0",  # Plotting
            "seaborn>=0.11.0",  # Enhanced visualizations
        ],
        "dev": [
            "pytest>=6.0.0",  # Testing
            "pytest-cov>=2.10.0",  # Test coverage
            "flake8>=3.8.0",  # Linting
            "black>=20.8b1",  # Code formatting
        ],
        "full": [
            "matplotlib>=3.4.0",  # Plotting
            "seaborn>=0.11.0",  # Enhanced visualizations
            "tqdm>=4.60.0",  # Progress bars
            "pytest>=6.0.0",  # Testing
            "pytest-cov>=2.10.0",  # Test coverage
        ],
    },

    # Entry points
    entry_points={
        "console_scripts": [
            "bugpredictor=bugpredictor.cli:main",  # Command-line interface
        ],
    },

    # Package classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Eclipse Public License 2.0 (EPL-2.0)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],

    # License and keywords
    license="Eclipse Public License 2.0 (EPL-2.0)",
    keywords="bug prediction, defect prediction, software quality, machine learning",
)