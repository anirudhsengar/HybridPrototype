#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Weight Calculator for Hybrid Bug Predictor

This module calculates optimal weights for combining FixCache
and REPD predictions based on repository characteristics.

Author: anirudhsengar
"""

import logging
import math
from typing import Dict, Set, Any, Optional

logger = logging.getLogger(__name__)


class DynamicWeightCalculator:
    """
    Calculates optimal weights for the hybrid prediction approach
    based on repository characteristics.

    This class analyzes various aspects of the repository and predictions
    from both approaches to determine the optimal weighting strategy.
    """

    def __init__(self):
        """Initialize the weight calculator."""
        self.last_weights = None
        self.last_factors = {}

    def calculate_weights(
            self,
            fixcache_files: Set[str],
            repd_scores: Dict[str, float],
            default_weights: Dict[str, float],
            repository_stats: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal weights based on repository characteristics.

        Args:
            fixcache_files: Set of files predicted by FixCache
            repd_scores: Dictionary of risk scores from REPD
            default_weights: Default weights to use as a baseline
            repository_stats: Additional repository statistics

        Returns:
            Dictionary mapping prediction approaches to their weights
        """
        # Start with default weights
        weights = default_weights.copy()
        factors = {}

        # If we don't have repository stats, we can't do much optimization
        if not repository_stats:
            logger.warning("No repository stats provided, using default weights")
            self.last_weights = weights
            return weights

        # Extract relevant statistics
        total_commits = repository_stats.get("total_commits", 0)
        bug_fixes = repository_stats.get("bug_fixes", 0)
        all_files = repository_stats.get("all_files", [])
        total_files = len(all_files) if all_files else 0

        # Factor 1: Bug history strength
        # Repositories with rich bug history should favor the temporal approach
        if total_commits > 0:
            bug_ratio = bug_fixes / total_commits
            bug_history_factor = self._sigmoid(bug_ratio * 10)  # Scale for sigmoid
            factors["bug_history"] = bug_history_factor

            if bug_history_factor > 0.7:
                logger.info(f"Strong bug history detected (ratio: {bug_ratio:.2f}), "
                            f"favoring temporal approach")
                weights["temporal"] = max(weights["temporal"], 0.7)
                weights["structural"] = 1.0 - weights["temporal"]

            elif bug_history_factor < 0.3:
                logger.info(f"Limited bug history detected (ratio: {bug_ratio:.2f}), "
                            f"favoring structural approach")
                weights["structural"] = max(weights["structural"], 0.7)
                weights["temporal"] = 1.0 - weights["structural"]

        # Factor 2: Approach overlap
        # Low overlap suggests complementary approaches, use balanced weights
        overlap_factor = self._calculate_approach_overlap(
            fixcache_files,
            repd_scores
        )
        factors["approach_overlap"] = overlap_factor

        if overlap_factor < 0.2:
            logger.info(f"Low overlap between approaches ({overlap_factor:.2f}), "
                        f"using more balanced weights")
            # Move weights closer to 50/50
            weights["temporal"] = 0.5 + (weights["temporal"] - 0.5) * 0.5
            weights["structural"] = 0.5 + (weights["structural"] - 0.5) * 0.5

        # Factor 3: REPD coverage
        # If REPD has analyzed most files but FixCache has few, adjust accordingly
        if total_files > 0:
            repd_coverage = len(repd_scores) / total_files
            fixcache_coverage = len(fixcache_files) / total_files
            coverage_ratio = repd_coverage / fixcache_coverage if fixcache_coverage > 0 else 10.0

            factors["coverage_ratio"] = coverage_ratio

            if coverage_ratio > 3.0:
                logger.info(f"REPD has significantly better coverage ({repd_coverage:.2f} vs "
                            f"{fixcache_coverage:.2f}), increasing structural weight")
                weights["structural"] += 0.1

            elif coverage_ratio < 0.33:
                logger.info(f"FixCache has significantly better coverage ({fixcache_coverage:.2f} vs "
                            f"{repd_coverage:.2f}), increasing temporal weight")
                weights["temporal"] += 0.1

        # Factor 4: Repository size and age
        # Larger, older repositories often benefit more from temporal approach
        if total_commits > 1000 and total_files > 1000:
            logger.info("Large repository detected, slightly favoring temporal approach")
            weights["temporal"] += 0.05

        # Factor 5: REPD effectiveness
        # If REPD scores are very concentrated (many files with similar scores),
        # it may be less effective at differentiation
        if repd_scores:
            repd_variance = self._calculate_variance(list(repd_scores.values()))
            factors["repd_variance"] = repd_variance

            if repd_variance < 0.01:
                logger.info("Low variance in REPD scores detected, reducing structural weight")
                weights["structural"] -= 0.1

        # Normalize weights to ensure they sum to 1.0
        total = sum(weights.values())
        for key in weights:
            weights[key] /= total

        # Store weights and factors for later retrieval
        self.last_weights = weights
        self.last_factors = factors

        logger.info(f"Final weights: temporal={weights['temporal']:.2f}, "
                    f"structural={weights['structural']:.2f}")

        return weights

    def get_last_weights(self) -> Dict[str, float]:
        """
        Get the most recently calculated weights.

        Returns:
            Dictionary mapping approaches to their weights,
            or None if weights haven't been calculated yet
        """
        return self.last_weights

    def get_weight_factors(self) -> Dict[str, float]:
        """
        Get the factors that influenced the most recent weight calculation.

        Returns:
            Dictionary mapping factor names to their values
        """
        return self.last_factors

    def _calculate_approach_overlap(
            self,
            fixcache_files: Set[str],
            repd_scores: Dict[str, float],
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
        high_risk_files = {file for file, score in repd_scores.items()
                           if score >= risk_threshold}

        # Calculate overlap coefficient
        if not fixcache_files or not high_risk_files:
            return 0.0

        intersection = fixcache_files.intersection(high_risk_files)
        min_size = min(len(fixcache_files), len(high_risk_files))

        return len(intersection) / min_size if min_size > 0 else 0.0

    def _calculate_variance(self, values: list) -> float:
        """
        Calculate the variance of a list of values.

        Args:
            values: List of numerical values

        Returns:
            Variance of the values
        """
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        squared_diffs = [(x - mean) ** 2 for x in values]

        return sum(squared_diffs) / len(values)

    def _sigmoid(self, x: float) -> float:
        """
        Sigmoid function to transform values to range (0,1).

        Args:
            x: Input value

        Returns:
            Sigmoid of x: 1/(1+e^-x)
        """
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0