#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Weight Calculator for Hybrid Bug Predictor

This module calculates optimal weights for combining FixCache
and REPD predictions based on repository characteristics.

Author: anirudhsengar
"""

import logging
from typing import Dict, Set, Any

logger = logging.getLogger(__name__)


class DynamicWeightCalculator:
    """
    Calculates optimal weights for the hybrid prediction approach
    based on repository characteristics.
    """

    def __init__(self, repository):
        """
        Initialize the weight calculator.

        Args:
            repository: Repository object to analyze
        """
        self.repository = repository

    def calculate_weights(
            self,
            fixcache_files: Set[str],
            risk_scores: Dict[str, float],
            default_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate optimal weights based on repository characteristics.

        Args:
            fixcache_files: Set of files predicted by FixCache
            risk_scores: Dictionary of risk scores from REPD
            default_weights: Default weights to use as a baseline

        Returns:
            Dictionary mapping prediction approaches to their weights
        """
        # Start with default weights
        weights = default_weights.copy()

        # If repository has no historical data, favor REPD
        if len(self.repository.bug_fixes) < 10:
            weights["historical"] = 0.3
            weights["risk_based"] = 0.7
            logger.info("Limited bug history available, favoring REPD approach")

        # If repository has many bug fixes, favor historical approach
        elif len(self.repository.bug_fixes) > 100:
            weights["historical"] = 0.7
            weights["risk_based"] = 0.3
            logger.info("Rich bug history available, favoring FixCache approach")

        # Adjust weights based on overlap between approaches
        overlap = self._calculate_approach_overlap(fixcache_files, risk_scores)
        if overlap < 0.2:
            # Low overlap means approaches are complementary, use more balanced weights
            weights["historical"] = 0.5
            weights["risk_based"] = 0.5
            logger.info("Low overlap between approaches, using balanced weights")

        # Normalize weights
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total

        return weights

    def _calculate_approach_overlap(
            self,
            fixcache_files: Set[str],
            risk_scores: Dict[str, float],
            risk_threshold: float = 0.7
    ) -> float:
        """
        Calculate the overlap between FixCache and REPD high-risk predictions.

        Args:
            fixcache_files: Set of files predicted by FixCache
            risk_scores: Dictionary of risk scores from REPD
            risk_threshold: Threshold for high risk in REPD

        Returns:
            Overlap coefficient between the two sets
        """
        # Get high-risk files according to REPD
        high_risk_files = {file for file, score in risk_scores.items()
                           if score >= risk_threshold}

        # Calculate overlap coefficient
        if not fixcache_files or not high_risk_files:
            return 0.0

        intersection = fixcache_files.intersection(high_risk_files)
        min_size = min(len(fixcache_files), len(high_risk_files))

        return len(intersection) / min_size if min_size > 0 else 0.0