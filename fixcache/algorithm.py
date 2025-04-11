#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FixCache Algorithm Implementation

This module implements the core FixCache algorithm for bug prediction
with significant enhancements over the original BugTools implementation.

Author: anirudhsengar
"""

import os
import re
import time
import logging
import datetime
from typing import List, Dict, Set, Tuple, Any, Optional, Union, Callable
from collections import defaultdict, Counter

# Internal imports
from .repository import RepositoryAnalyzer
from .utils import is_code_file, safe_divide

# Setup logging
logger = logging.getLogger(__name__)


class FixCache:
    """
    Enhanced implementation of the FixCache algorithm for bug prediction.

    This class implements the core algorithm described in "Predicting Faults from
    Cached History" by Sunghun Kim et al., with significant enhancements for
    accuracy, robustness, and usability.
    """

    def __init__(
            self,
            repo_path: Union[str, 'RepositoryAnalyzer'],
            cache_size: float = 0.2,
            policy: str = "BUG",
            bug_keywords: Optional[List[str]] = None,
            lookback_commits: Optional[int] = None,
            min_file_count: int = 10,
            window_ratio: float = 0.25,
            cache_seeding: bool = True
    ):
        """
        Initialize the FixCache algorithm with configuration parameters.

        Args:
            repo_path: Path to git repository or RepositoryAnalyzer instance
            cache_size: Size of cache as fraction of total files (0.1 = 10%)
            policy: Cache replacement policy ("BUG", "FIFO", "LRU")
            bug_keywords: List of keywords to identify bug-fixing commits
            lookback_commits: Number of recent commits to analyze (None = all)
            min_file_count: Minimum number of files required for analysis
            window_ratio: Ratio of commits to use for training window
            cache_seeding: Whether to seed cache with most bug-prone files
        """
        # Store the repository path
        self.repo_path = repo_path if isinstance(repo_path, str) else getattr(repo_path, 'repo_path', 'Unknown')

        # Initialize repository analyzer
        if isinstance(repo_path, str):
            self.repo_analyzer = RepositoryAnalyzer(
                repo_path,
                bug_keywords=bug_keywords,
                lookback_commits=lookback_commits
            )
        else:
            self.repo_analyzer = repo_path

        # Configuration parameters
        self.cache_size = cache_size
        self.policy = policy.upper()
        self.min_file_count = min_file_count
        self.window_ratio = window_ratio
        self.cache_seeding = cache_seeding

        # Validate policy
        valid_policies = ["BUG", "FIFO", "LRU"]
        if self.policy not in valid_policies:
            raise ValueError(f"Invalid cache policy. Must be one of: {', '.join(valid_policies)}")

        # Initialize internal state
        self.cache: List[str] = []
        self.cache_max_size: int = 0
        self.hit_count: int = 0
        self.miss_count: int = 0
        self.results: Dict[str, Any] = {}
        self.error_messages: List[str] = []

        # Initialize cache performance tracking
        self.file_access_count: Dict[str, int] = defaultdict(int)
        self.file_hit_count: Dict[str, int] = defaultdict(int)

        # Timestamp of creation for tracking
        self.created_at = datetime.datetime.now().isoformat()

        logger.info(f"Initialized FixCache with cache size {cache_size * 100:.1f}%")

    def analyze_repository(self) -> bool:
        """
        Analyze the git repository and extract commit history.

        Returns:
            True if analysis was successful, False otherwise
        """
        try:
            logger.info("Starting repository analysis")
            start_time = time.time()

            # Run repository analysis
            success = self.repo_analyzer.analyze()
            if not success:
                self.error_messages.extend(self.repo_analyzer.error_messages)
                return False

            # Calculate cache size based on total files
            self.cache_max_size = max(1, int(self.repo_analyzer.total_files * self.cache_size))

            # Check minimum file count
            if self.repo_analyzer.total_files < self.min_file_count:
                error_msg = f"Repository has only {self.repo_analyzer.total_files} files, which is less than the required minimum of {self.min_file_count}"
                logger.error(error_msg)
                self.error_messages.append(error_msg)
                return False

            elapsed_time = time.time() - start_time
            logger.info(f"Repository analysis completed in {elapsed_time:.2f} seconds")
            logger.info(f"Total files: {self.repo_analyzer.total_files}")
            logger.info(f"Found {len(self.repo_analyzer.bug_fixes)} bug-fixing commits")
            logger.info(f"Cache size: {self.cache_size * 100:.1f}% ({self.cache_max_size} files)")

            return True

        except Exception as e:
            error_msg = f"Error during repository analysis: {str(e)}"
            logger.error(error_msg)
            self.error_messages.append(error_msg)
            return False

    def predict(self) -> float:
        """
        Run the FixCache algorithm to predict fault-prone files.

        Returns:
            Hit rate as a percentage (0-100)
        """
        logger.info("Starting FixCache prediction")
        start_time = time.time()

        try:
            # Ensure repository has been analyzed
            if not self.repo_analyzer.is_analyzed:
                logger.warning("Repository has not been analyzed, running analysis now")
                if not self.analyze_repository():
                    return 0.0

            # Check for bug fixes
            if len(self.repo_analyzer.bug_fixes) == 0:
                error_msg = "No bug fixes found, cannot perform prediction"
                logger.warning(error_msg)
                self.error_messages.append(error_msg)
                return 0.0

            # Reset internal state
            self._reset_state()

            # Sort commits chronologically for temporal analysis
            sorted_commits = sorted(
                self.repo_analyzer.bug_fixes,
                key=lambda x: x['timestamp']
            )

            # Apply sliding window approach
            return self._sliding_window_prediction(sorted_commits)

        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg)
            self.error_messages.append(error_msg)
            return 0.0

    def _reset_state(self) -> None:
        """Reset internal state before running prediction."""
        self.cache = []
        self.hit_count = 0
        self.miss_count = 0
        self.file_access_count = defaultdict(int)
        self.file_hit_count = defaultdict(int)

    def _sliding_window_prediction(self, sorted_commits: List[Dict[str, Any]]) -> float:
        """
        Apply sliding window approach for more accurate prediction.

        Args:
            sorted_commits: Chronologically sorted bug-fixing commits

        Returns:
            Hit rate as a percentage (0-100)
        """
        total_commits = len(sorted_commits)

        # Calculate window size based on repository size
        if total_commits <= 10:
            # For very small repositories, use half for training
            window_size = max(1, total_commits // 2)
        else:
            # For larger repositories, use window_ratio
            window_size = max(10, int(total_commits * self.window_ratio))

        logger.info(f"Using sliding window with {window_size} commits for training "
                    f"({window_size / total_commits * 100:.1f}% of total)")

        # Split commits into training and testing sets
        training_commits = sorted_commits[:window_size]
        testing_commits = sorted_commits[window_size:]

        # If no testing commits, use a smaller training window
        if not testing_commits and total_commits > 2:
            window_size = total_commits // 2
            training_commits = sorted_commits[:window_size]
            testing_commits = sorted_commits[window_size:]

        # Initialize cache using training data
        self._initialize_cache(training_commits)

        # Test prediction on remaining commits
        for commit in testing_commits:
            self._process_commit_for_prediction(commit)

        # Calculate hit rate
        hit_rate = self._calculate_hit_rate()
        self._store_results(hit_rate)

        # Log results
        logger.info(f"Prediction completed: hit rate {hit_rate:.2f}%, "
                    f"hits: {self.hit_count}, misses: {self.miss_count}")

        return hit_rate

    def _initialize_cache(self, training_commits: List[Dict[str, Any]]) -> None:
        """
        Initialize cache using training data.

        Args:
            training_commits: List of commits for initial training
        """
        # First: extract file bug frequencies from training data
        file_bugs = defaultdict(int)
        for commit in training_commits:
            for file_path in commit['files_changed']:
                if file_path in self.repo_analyzer.file_stats:
                    file_bugs[file_path] += 1

        # Seed the cache with most bug-prone files if enabled
        if self.cache_seeding and file_bugs:
            most_buggy = sorted(file_bugs.items(), key=lambda x: x[1], reverse=True)
            initial_files = [file for file, _ in most_buggy[:self.cache_max_size]]
            self.cache = initial_files[:self.cache_max_size]
            logger.info(f"Seeded cache with {len(self.cache)} most bug-prone files")

        # Process training commits to update cache
        for commit in training_commits:
            for file_path in commit['files_changed']:
                if (file_path in self.repo_analyzer.file_stats and
                        file_path not in self.cache):

                    # Add file to cache
                    self.cache.append(file_path)

                    # If cache is full, evict according to policy
                    if len(self.cache) > self.cache_max_size:
                        self._evict_from_cache()

    def _process_commit_for_prediction(self, commit: Dict[str, Any]) -> None:
        """
        Process a commit for prediction, updating hit/miss counts.

        Args:
            commit: Commit information dictionary
        """
        for file_path in commit['files_changed']:
            # Skip files not in our file stats (non-code files)
            if file_path not in self.repo_analyzer.file_stats:
                continue

            # Track file access
            self.file_access_count[file_path] += 1

            # Check if file is in cache (hit) or not (miss)
            if file_path in self.cache:
                self.hit_count += 1
                self.file_hit_count[file_path] += 1

                # For LRU policy, move to end of cache
                if self.policy == "LRU":
                    self.cache.remove(file_path)
                    self.cache.append(file_path)
            else:
                self.miss_count += 1

                # Add file to cache
                self.cache.append(file_path)

                # If cache is full, evict according to policy
                if len(self.cache) > self.cache_max_size:
                    self._evict_from_cache()

    def _evict_from_cache(self) -> None:
        """Evict a file from the cache based on the selected policy."""
        if not self.cache:
            return

        if self.policy == "FIFO":
            # First In First Out - remove oldest file
            self.cache.pop(0)

        elif self.policy == "LRU":
            # Least Recently Used - already handled by moving hits to end
            self.cache.pop(0)

        elif self.policy == "BUG":
            # BUG policy - remove file with fewest bug fixes
            min_bugs = float('inf')
            min_index = 0

            for i, file_path in enumerate(self.cache):
                if file_path in self.repo_analyzer.file_stats:
                    bug_fixes = self.repo_analyzer.file_stats[file_path]['bug_fixes']
                    if bug_fixes < min_bugs:
                        min_bugs = bug_fixes
                        min_index = i

            self.cache.pop(min_index)

    def _calculate_hit_rate(self) -> float:
        """
        Calculate the hit rate.

        Returns:
            Hit rate as a percentage (0-100)
        """
        total = self.hit_count + self.miss_count
        return safe_divide(self.hit_count, total) * 100

    def _store_results(self, hit_rate: float) -> None:
        """
        Store prediction results for later reference.

        Args:
            hit_rate: Calculated hit rate
        """
        self.results = {
            'hit_rate': hit_rate,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'total_bug_fixes': len(self.repo_analyzer.bug_fixes),
            'total_files': self.repo_analyzer.total_files,
            'cache_size': self.cache_size,
            'cache_max_size': self.cache_max_size,
            'policy': self.policy,
            'top_files': self.get_top_files(10),
            'bottom_files': self.get_bottom_files(10),
            'file_hit_rates': self._calculate_file_hit_rates(),
            'timestamp': datetime.datetime.now().isoformat(),
            'repo_path': self.repo_analyzer.repo_path,
            'errors': self.error_messages,
            'window_ratio': self.window_ratio,
            'cache_seeding': self.cache_seeding,
        }

    def get_top_files(self, num_files=10):
        """
        Get the top bug-prone files based on the analysis.

        Args:
            num_files (int): Number of files to return

        Returns:
            list: List of tuples (file_path, risk_score) sorted by risk score
        """
        if not hasattr(self, 'file_stats') or not self.file_stats:
            return []

        # Calculate risk scores for all files
        file_risks = []
        for file_path, stats in self.file_stats.items():
            # Simple risk formula: bug_fixes / (days_since_last_fix + 1)
            bug_fixes = stats.get('bug_fixes', 0)
            commit_count = stats.get('commit_count', 0)

            if commit_count == 0:
                continue

            # Risk score formula
            # Prioritizes files with more bug fixes and more recent activity
            risk_score = bug_fixes * (bug_fixes / commit_count)

            file_risks.append((file_path, risk_score))

        # Sort by risk score in descending order
        file_risks.sort(key=lambda x: x[1], reverse=True)

        # Return top N files
        return file_risks[:num_files]

    def get_bottom_files(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get the files least likely to contain bugs.

        Args:
            n: Number of files to return

        Returns:
            List of dictionaries with file info
        """
        # Files not in cache are least likely to have bugs
        non_cache_files = []

        for file_path, stats in self.repo_analyzer.file_stats.items():
            if file_path not in self.cache:
                hit_rate = safe_divide(
                    self.file_hit_count[file_path],
                    self.file_access_count[file_path]
                ) * 100 if file_path in self.file_access_count else 0

                non_cache_files.append({
                    'file_path': file_path,
                    'bug_fixes': stats['bug_fixes'],
                    'hit_rate': hit_rate,
                    'file_type': os.path.splitext(file_path)[1],
                    'last_modified': stats.get('last_bug_fix', 0)
                })

        # Sort by bug fix count (ascending)
        non_cache_files.sort(key=lambda x: x['bug_fixes'])

        # Return bottom N files or all if less than N
        return non_cache_files[:min(n, len(non_cache_files))]

    def _calculate_file_hit_rates(self) -> List[Dict[str, Any]]:
        """
        Calculate hit rates for individual files.

        Returns:
            List of dictionaries with file hit rates
        """
        file_hit_rates = []

        for file_path, access_count in self.file_access_count.items():
            if access_count > 0:
                hit_count = self.file_hit_count[file_path]
                hit_rate = safe_divide(hit_count, access_count) * 100

                file_hit_rates.append({
                    'file_path': file_path,
                    'access_count': access_count,
                    'hit_count': hit_count,
                    'hit_rate': hit_rate,
                    'in_cache': file_path in self.cache
                })

        # Sort by hit rate (descending)
        file_hit_rates.sort(key=lambda x: x['hit_rate'], reverse=True)

        return file_hit_rates

    def save_results(self, output_file: str) -> None:
        """
        Save prediction results to a JSON file.

        Args:
            output_file: Path to output file
        """
        import json

        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)

            logger.info(f"Results saved to {output_file}")

        except Exception as e:
            error_msg = f"Error saving results: {str(e)}"
            logger.error(error_msg)
            self.error_messages.append(error_msg)

    def visualize_results(self, output_file: str = "fixcache_results.png") -> None:
        """
        Visualize the prediction results.

        Args:
            output_file: Path to output image file
        """
        from fixcache.visualization import visualize_results

        if not self.results:
            logger.error("No results to visualize. Run predict() first.")
            return

        try:
            visualize_results(self.results, output_file)
            logger.info(f"Visualization saved to {output_file}")

        except Exception as e:
            error_msg = f"Error visualizing results: {str(e)}"
            logger.error(error_msg)
            self.error_messages.append(error_msg)

    def get_recommended_actions(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get recommended actions based on prediction results.

        Args:
            n: Maximum number of recommendations

        Returns:
            List of recommended actions
        """
        if not self.results:
            logger.warning("No prediction results available. Run predict() first.")
            return []

        recommendations = []

        # Add recommendations for top bug-prone files
        top_files = self.results.get('top_files', [])
        for file_info in top_files[:n]:
            file_path = file_info['file_path']
            bug_fixes = file_info['bug_fixes']

            recommendations.append({
                'type': 'high_risk_file',
                'file_path': file_path,
                'bug_fixes': bug_fixes,
                'message': f"Consider additional testing and review for {os.path.basename(file_path)} ({bug_fixes} historical bugs)",
                'priority': 'high'
            })

        # Add recommendation for optimal cache size if current is suboptimal
        if self.cache_size > 0.1 and self.results['hit_rate'] < 50:
            recommendations.append({
                'type': 'optimization',
                'message': "Consider reducing cache size to 10% for small repositories",
                'priority': 'medium'
            })

        # Add recommendation for repository with few bug fixes
        if len(self.repo_analyzer.bug_fixes) < 20:
            recommendations.append({
                'type': 'data_quality',
                'message': "Limited bug history available. Results may have reduced accuracy.",
                'priority': 'medium'
            })

        return recommendations

    def get_summary(self):
        """
        Get a summary of the analysis results.

        Returns:
            Dict[str, Any]: Summary of analysis results
        """
        total_files = len(self.repo_analyzer.all_files) if hasattr(self.repo_analyzer, 'all_files') else 0
        total_commits = len(self.repo_analyzer.all_commits) if hasattr(self.repo_analyzer, 'all_commits') else 0
        bug_fixes = len(self.repo_analyzer.bug_fixes) if hasattr(self.repo_analyzer, 'bug_fixes') else 0

        if self.hit_count + self.miss_count > 0:
            hit_rate = (self.hit_count / (self.hit_count + self.miss_count)) * 100
        else:
            hit_rate = 0

        summary = {
            'repository_path': self.repo_path,
            'total_files': total_files,
            'total_commits': total_commits,
            'bug_fixes': bug_fixes,
            'cache_size': self.cache_size,
            'policy': self.policy,
            'window_ratio': self.window_ratio,
            'hit_rate': hit_rate,
            'created_at': self.created_at
        }

        # Add top files if we have file statistics
        if hasattr(self, 'get_top_files'):
            summary['top_files'] = self.get_top_files(20)

        return summary


# Factory function for easier instantiation
def create_fixcache(
        repo_path: str,
        cache_size: float = 0.2,
        policy: str = "BUG",
        bug_keywords: Optional[List[str]] = None
) -> FixCache:
    """
    Factory function to create and initialize a FixCache instance.

    Args:
        repo_path: Path to git repository
        cache_size: Size of cache as fraction of total files
        policy: Cache replacement policy
        bug_keywords: List of keywords to identify bug-fixing commits

    Returns:
        Initialized FixCache instance
    """
    fix_cache = FixCache(
        repo_path=repo_path,
        cache_size=cache_size,
        policy=policy,
        bug_keywords=bug_keywords
    )

    # Run repository analysis
    fix_cache.analyze_repository()

    return fix_cache