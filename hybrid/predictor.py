#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Bug Predictor

This module implements a hybrid prediction approach that combines
FixCache temporal patterns with REPD code metrics analysis.

Author: anirudhsengar
Date: 2025-04-18
"""

import os
import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Set

from ..fixcache.algorithm import FixCache
from ..repd.model import REPDModel
from ..repd.structure_mapper import StructureMapper
from ..repd.risk_calculator import RiskCalculator
from .weighting import DynamicWeightCalculator

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Combines FixCache and REPD approaches for improved bug prediction.

    This class integrates temporal patterns from BugCache/FixCache with
    code metrics analysis from REPD to provide more accurate bug predictions
    than either approach alone.
    """

    def __init__(self,
                 repo_path: str,
                 cache_size: float = 0.1,
                 weights: Optional[Dict[str, float]] = None,
                 lookback_commits: Optional[int] = None):
        """
        Initialize the hybrid predictor.

        Args:
            repo_path: Path to the repository to analyze
            cache_size: Cache size as a fraction of total files (default: 0.1)
            weights: Optional preset weights for the hybrid approach
            lookback_commits: Number of commits to analyze (None = all)
        """
        self.repo_path = os.path.abspath(repo_path)
        self.cache_size = cache_size
        self.lookback_commits = lookback_commits

        # Initialize both models
        self.fixcache = FixCache(
            repo_path=self.repo_path,
            cache_size=self.cache_size,
            lookback_commits=self.lookback_commits
        )

        # For REPD, we need structure mapping and risk calculation
        self.structure_mapper = StructureMapper(self.repo_path)
        self.risk_calculator = RiskCalculator(self.repo_path, self.structure_mapper)
        self.repd_model = REPDModel(
            repo_path=self.repo_path,
            structure_mapper=self.structure_mapper,
            risk_calculator=self.risk_calculator
        )

        # Initialize weight calculator
        self.weight_calculator = DynamicWeightCalculator()

        # Default weights if not provided
        self.weights = weights or {
            "temporal": 0.6,  # FixCache results weight
            "structural": 0.4  # REPD results weight
        }

        # Analysis state
        self.has_analyzed_fixcache = False
        self.has_analyzed_repd = False
        self.analysis_timestamp = None

        # Results storage
        self.fixcache_files = set()
        self.repd_scores = {}
        self.hybrid_scores = {}

    def analyze_repository(self) -> bool:
        """
        Analyze the repository using both approaches.

        Returns:
            True if analysis was successful, False otherwise
        """
        logger.info(f"Starting hybrid analysis of repository: {self.repo_path}")
        start_time = time.time()

        # Run FixCache analysis
        logger.info("Running FixCache analysis...")
        fixcache_success = self._run_fixcache_analysis()

        # Run REPD analysis
        logger.info("Running REPD analysis...")
        repd_success = self._run_repd_analysis()

        # Update analysis state
        if fixcache_success and repd_success:
            self.analysis_timestamp = time.time()
            logger.info(f"Hybrid analysis completed in {time.time() - start_time:.2f} seconds")
            return True

        logger.error(f"Hybrid analysis failed: FixCache={fixcache_success}, REPD={repd_success}")
        return False

    def _run_fixcache_analysis(self) -> bool:
        """Run FixCache analysis on the repository."""
        try:
            self.fixcache.analyze()
            self.fixcache.build_cache()
            self.fixcache_files = set(self.fixcache.get_cached_files())
            self.has_analyzed_fixcache = True
            logger.info(f"FixCache analysis found {len(self.fixcache_files)} files in cache")
            return True
        except Exception as e:
            logger.error(f"FixCache analysis failed: {str(e)}")
            return False

    def _run_repd_analysis(self) -> bool:
        """Run REPD analysis on the repository."""
        try:
            self.repd_model.analyze()
            self.repd_scores = self.repd_model.calculate_risk_scores()
            self.has_analyzed_repd = True
            logger.info(f"REPD analysis calculated risk scores for {len(self.repd_scores)} files")
            return True
        except Exception as e:
            logger.error(f"REPD analysis failed: {str(e)}")
            return False

    def predict(self) -> Dict[str, float]:
        """
        Run prediction using both models and combine results.

        Returns:
            Dictionary mapping file paths to hybrid risk scores
        """
        # Ensure repository is analyzed
        if not self.has_analyzed_fixcache or not self.has_analyzed_repd:
            success = self.analyze_repository()
            if not success:
                logger.error("Prediction failed due to analysis failure")
                return {}

        # Get dynamic weights based on repository characteristics
        logger.info("Calculating dynamic weights for prediction")
        dynamic_weights = self.weight_calculator.calculate_weights(
            self.fixcache_files,
            self.repd_scores,
            self.weights,
            self.fixcache.get_repository_stats()
        )

        logger.info(f"Using weights: temporal={dynamic_weights['temporal']:.2f}, "
                    f"structural={dynamic_weights['structural']:.2f}")

        # Combine predictions
        self.hybrid_scores = {}

        # Process all files from both approaches
        all_files = set(self.fixcache.get_all_files()) | set(self.repd_scores.keys())

        for file_path in all_files:
            # Skip files that don't exist anymore
            if not os.path.exists(os.path.join(self.repo_path, file_path)):
                continue

            # Get FixCache prediction (1 if in cache, 0 otherwise)
            temporal_score = 1.0 if file_path in self.fixcache_files else 0.0

            # Get REPD risk score (default to 0 if not available)
            structural_score = self.repd_scores.get(file_path, 0.0)

            # Compute weighted hybrid score
            self.hybrid_scores[file_path] = (
                    dynamic_weights["temporal"] * temporal_score +
                    dynamic_weights["structural"] * structural_score
            )

        logger.info(f"Generated hybrid scores for {len(self.hybrid_scores)} files")
        return self.hybrid_scores

    def get_top_risky_files(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top files most likely to contain bugs.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of (file_path, score) tuples sorted by risk
        """
        # Run prediction if not already done
        if not self.hybrid_scores:
            self.predict()

        # Sort files by score and return top N
        top_files = sorted(
            self.hybrid_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return top_files

    def get_prediction_breakdown(self, file_path: str) -> Dict[str, Any]:
        """
        Get detailed breakdown of prediction components for a file.

        Args:
            file_path: Path to the file (relative to repo root)

        Returns:
            Dictionary with prediction components and scores
        """
        if not self.has_analyzed_fixcache or not self.has_analyzed_repd:
            self.analyze_repository()

        # Get individual scores
        temporal_score = 1.0 if file_path in self.fixcache_files else 0.0
        structural_score = self.repd_scores.get(file_path, 0.0)

        # Get dynamic weights
        weights = self.weight_calculator.calculate_weights(
            self.fixcache_files,
            self.repd_scores,
            self.weights,
            self.fixcache.get_repository_stats()
        )

        # Calculate hybrid score
        hybrid_score = (weights["temporal"] * temporal_score +
                        weights["structural"] * structural_score)

        # Get detailed breakdown from REPD if available
        repd_details = {}
        if hasattr(self.repd_model, 'get_detailed_metrics') and file_path in self.repd_scores:
            repd_details = self.repd_model.get_detailed_metrics(file_path)

        # Compile results
        return {
            "file_path": file_path,
            "hybrid_score": hybrid_score,
            "temporal": {
                "in_cache": file_path in self.fixcache_files,
                "score": temporal_score,
                "weight": weights["temporal"]
            },
            "structural": {
                "score": structural_score,
                "weight": weights["structural"],
                "details": repd_details
            }
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from the hybrid analysis.

        Returns:
            Dictionary of summary statistics
        """
        if not self.has_analyzed_fixcache or not self.has_analyzed_repd:
            self.analyze_repository()

        # Get repository statistics
        repo_stats = self.fixcache.get_repository_stats()

        # Calculate average scores
        avg_hybrid_score = sum(self.hybrid_scores.values()) / len(self.hybrid_scores) if self.hybrid_scores else 0
        avg_repd_score = sum(self.repd_scores.values()) / len(self.repd_scores) if self.repd_scores else 0

        return {
            "repository_path": self.repo_path,
            "analysis_timestamp": self.analysis_timestamp,
            "total_files": len(repo_stats.get("all_files", [])),
            "total_commits": repo_stats.get("total_commits", 0),
            "bug_fixes": repo_stats.get("bug_fixes", 0),
            "cache_size": self.cache_size,
            "files_in_cache": len(self.fixcache_files),
            "avg_hybrid_score": avg_hybrid_score,
            "avg_repd_score": avg_repd_score,
            "weights_used": self.weight_calculator.get_last_weights()
        }