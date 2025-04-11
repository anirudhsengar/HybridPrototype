#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FixCache - Enhanced Bug Prediction Tool

This package implements an enhanced version of the FixCache algorithm for predicting
fault-prone files in software repositories. It revives and extends the original BugTools
prototype developed by the Adoptium project.

Author: anirudhsengar
"""

from typing import List, Dict, Any, Optional, Tuple

# Package metadata
__version__ = '0.1.0'
__author__ = 'anirudhsengar'
__license__ = 'Eclipse Public License 2.0'

# Import core components for convenient access
from .algorithm import FixCache
from .repository import RepositoryAnalyzer
from .visualization import visualize_results, plot_cache_optimization

# Utility functions for easy access
from .utils import optimize_cache_size, compare_repositories

# Type aliases for documentation
HitRate = float
CacheSize = float
FilePath = str
CommitSHA = str
ResultDict = Dict[str, Any]

__all__ = [
    # Main classes
    'FixCache',
    'RepositoryAnalyzer',

    # Utility functions
    'optimize_cache_size',
    'compare_repositories',
    'visualize_results',
    'plot_cache_optimization',

    # Package metadata
    '__version__',
    '__author__',
    '__license__',
]

# Make py.typed available for type checkers
# This signals that the package supports type annotations