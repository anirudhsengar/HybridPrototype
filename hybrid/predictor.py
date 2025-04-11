#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hybrid Bug Predictor

This module implements a hybrid prediction approach that combines
FixCache temporal patterns with REPD code metrics analysis.

Author: anirudhsengar
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

from ..fixcache.algorithm import FixCache
from ..repd.model import REPDModel
from .weighting import DynamicWeightCalculator

logger = logging.getLogger(__name__)


class HybridPredictor:
    """
    Combines FixCache and REPD approaches for improved bug prediction.
    """

    def __init__(self,
                 repo_path: str,
                 cache_size: float = 0.2,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize the hybrid predictor.

        Args:
            repo_path: Path to the repository to analyze
            cache_size: Cache size as a fraction of total files
            weights: Optional preset weights for the hybrid approach
        """
        self.repo_path = repo_path

        # Initialize repository analyzer (will be shared by both models)
        from ..repository import Repository
        self.repo = Repository(repo_path)
        self.repo_analyzed = False

        # Initialize both models
        self.fixcache = FixCache(repo_path=self.repo, cache_size=cache_size)

        # For REPD, we need to initialize with structure mapper
        from ..repd.structure_mapper import StructureMapper
        self.structure_mapper = StructureMapper(self.repo)
        self.repd_model = REPDModel(self.repo, self.structure_mapper)

        # Initialize weight calculator
        self.weight_calculator = DynamicWeightCalculator(self.repo)

        # Default weights if not provided
        self.weights = weights or {
            "historical": 0.6,  # FixCache results weight
            "risk_based": 0.4  # REPD results weight
        }

    def analyze_repository(self) -> bool:
        """
        Analyze the repository once for both models.

        Returns:
            True if analysis was successful, False otherwise
        """
        if self.repo_analyzed:
            return True

        success = self.repo.analyze()
        if not success:
            return False

        self.repo_analyzed = True
        return True

    def predict(self) -> Dict[str, float]:
        """
        Run both prediction models and combine results.

        Returns:
            Dictionary mapping file paths to hybrid risk scores
        """
        # Ensure repository is analyzed
        if not self.repo_analyzed:
            success = self.analyze_repository()
            if not success:
                return {}

        # Run FixCache prediction
        fixcache_hit_rate = self.fixcache.predict()

        # Get files in the FixCache
        fixcache_files = set(self.fixcache.cache)

        # Run REPD prediction
        risk_scores = self.repd_model.calculate_risk_scores()

        # Calculate dynamic weights if needed
        dynamic_weights = self.weight_calculator.calculate_weights(
            fixcache_files,
            risk_scores,
            self.weights
        )

        # Combine predictions
        hybrid_scores = {}

        # Process all files from both approaches
        all_files = set(self.repo.file_stats.keys()) | set(risk_scores.keys())

        for file_path in all_files:
            # Get FixCache prediction (1 if in cache, 0 otherwise)
            historical_score = 1.0 if file_path in fixcache_files else 0.0

            # Get REPD risk score (default to 0 if not available)
            risk_score = risk_scores.get(file_path, 0.0)

            # Compute weighted hybrid score
            hybrid_scores[file_path] = (
                    dynamic_weights["historical"] * historical_score +
                    dynamic_weights["risk_based"] * risk_score
            )

        return hybrid_scores

    def get_top_risky_files(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top files most likely to contain bugs.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of (file_path, score) tuples sorted by risk
        """
        hybrid_scores = self.predict()
        return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:limit]