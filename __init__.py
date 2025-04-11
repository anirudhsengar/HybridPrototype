#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Bug Predictor

A hybrid approach to software defect prediction combining temporal patterns
(BugCache/FixCache) with code metrics analysis (REPD).

Author: anirudhsengar
"""

from typing import List, Dict, Any, Optional, Tuple

# Package metadata
__version__ = '0.1.0'
__author__ = 'anirudhsengar'
__license__ = 'Eclipse Public License 2.0'

# Import main components for convenient access
from .repository import Repository
from .fixcache.algorithm import FixCache
from .repd.model import REPDModel
from .hybrid.predictor import HybridPredictor

# Type aliases
FilePath = str
Score = float
ResultDict = Dict[str, Any]

__all__ = [
    # Main classes
    'Repository',
    'FixCache',
    'REPDModel',
    'HybridPredictor',

    # Package metadata
    '__version__',
    '__author__',
    '__license__',
]