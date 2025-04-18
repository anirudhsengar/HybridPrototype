#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Bug Predictor

A hybrid approach to software defect prediction combining temporal patterns
(BugCache/FixCache) with code metrics analysis (REPD).

This package implements the integration of two complementary bug prediction
approaches:

1. FixCache: Analyzes version control history to identify temporal patterns
   in bug fixes and build a cache of files likely to contain defects

2. REPD (Reconstruction Error Probability Distribution): Uses code metrics and
   autoencoders to detect anomalous code structures indicating potential defects

The hybrid approach dynamically weights both models based on repository
characteristics, providing more accurate predictions than either method alone.

Author: anirudhsengar
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

# Package metadata
__version__ = '0.1.0'
__author__ = 'Anirudh Sengar'
__license__ = 'Eclipse Public License 2.0'
__copyright__ = 'Copyright 2025 Anirudh Sengar'

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if no handlers are configured
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(_console_handler)
    logger.propagate = False

# Import main components for convenient access
from .repository import Repository, Commit, FileChange, FileMetrics

# Import from fixcache module (adjust imports based on your actual structure)
try:
    from .fixcache.algorithm import FixCache
except ImportError:
    logger.warning("FixCache module could not be imported")
    FixCache = None

# Import from repd module (adjust imports based on your actual structure)
try:
    from .repd.model import REPDModel
    from .repd.structure_mapper import StructureMapper
    from .repd.risk_calculator import RiskCalculator
except ImportError:
    logger.warning("REPD module could not be imported")
    REPDModel = None
    StructureMapper = None
    RiskCalculator = None

# Import hybrid components
from .hybrid.predictor import HybridPredictor
from .hybrid.weighting import DynamicWeightCalculator

# Import visualization utilities
try:
    from .visualization import Visualizer, create_visualizer
except ImportError:
    logger.warning("Visualization module could not be imported")
    Visualizer = None
    create_visualizer = None

# Type aliases for better documentation
FilePath = str
Score = float
ResultDict = Dict[str, Any]

# Define public API
__all__ = [
    # Main classes
    'Repository',
    'Commit',
    'FileChange',
    'FileMetrics',
    'FixCache',
    'REPDModel',
    'StructureMapper',
    'RiskCalculator',
    'HybridPredictor',
    'DynamicWeightCalculator',
    'Visualizer',

    # Helper functions
    'create_visualizer',

    # Package metadata
    '__version__',
    '__author__',
    '__license__',
]


def get_version() -> str:
    """Return the package version."""
    return __version__